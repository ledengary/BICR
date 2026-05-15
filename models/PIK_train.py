#!/usr/bin/env python3

"""
Script to train PIK (Probability of "I Know") models.
Implements a feedforward neural network to classify correctness based on
the FIRST token's hidden state representation (after seeing the question, before generating the answer).
Uses Optuna optimization for hyperparameter tuning.
"""

import argparse
import json
import os
import sys
import numpy as np
import random
import pickle
import copy
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List, Union
import logging
from tqdm import tqdm

# Configure logging early
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Parse arguments early to set CUDA_VISIBLE_DEVICES before importing torch
def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train PIK models with Optuna optimization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data configuration
    parser.add_argument("--data-dir", type=str, required=True,
                       help="Directory containing extracted representations (e.g., ../../data/generation_extraction/)")
    parser.add_argument("--model-name", type=str, required=True,
                       help="Model name to train on (e.g., Qwen2.5-VL-3B-Instruct)")
    parser.add_argument("--train-dataset-name", type=str, required=True,
                       help="Train dataset name (e.g., VLCB_train)")
    parser.add_argument("--val-dataset-name", type=str, required=True,
                       help="Validation dataset name (e.g., VLCB_val)")
    parser.add_argument("--output-dir", type=str, default="../trained_models/PIK",
                       help="Output directory for trained models")
    
    # Training configuration
    parser.add_argument("--cuda-devices", type=str, default="0",
                       help="Comma-separated CUDA device IDs to use for training (e.g., '0,1,2'). Uses first device for training.")
    parser.add_argument("--cuda-visible-devices", type=str, default=None,
                       help="CUDA_VISIBLE_DEVICES setting (e.g., '0,1,2,3')")
    
    # Optuna configuration
    parser.add_argument("--n-trials", type=int, default=20,
                       help="Number of Optuna trials to run")
    parser.add_argument("--n-jobs", type=int, default=1,
                       help="Number of parallel jobs for Optuna optimization")
    
    # Validation metric selection
    parser.add_argument("--validation-metric", type=str, default="composite",
                       choices=['auroc', 'ece', 'composite', 'auroc_with_constraint', 'validation_loss', 'composite_with_constraints'],
                       help="Validation metric for optimization")
    parser.add_argument("--composite-auroc-weight", type=float, default=0.6,
                       help="Weight for AUROC in composite score (0-1, default: 0.6 for ranking emphasis)")
    parser.add_argument("--min-sensitivity", type=float, default=0.60,
                       help="Minimum sensitivity threshold for composite_with_constraints (default: 0.60)")
    parser.add_argument("--min-specificity", type=float, default=0.60,
                       help="Minimum specificity threshold for composite_with_constraints (default: 0.60)")
    
    # Validation evaluation frequency
    parser.add_argument("--val-eval-frequency", type=int, default=1,
                       help="Frequency of validation evaluation (evaluate every N epochs)")
    
    # Optuna config file
    parser.add_argument("--optuna-config", type=str, default="../utils/optuna_configs.json",
                       help="Path to optuna configuration JSON file")
    parser.add_argument("--optuna-key", type=str, default="PIK",
                       help="Key to use in optuna config file (e.g., 'PIK')")
    parser.add_argument("--seed", type=int, default=None,
                       help="If set: seeds everything (incl. TPESampler) with this value and treats "
                            "--output-dir as the literal leaf (no {MODEL}/{DATASET} injection).")

    return parser.parse_args()

# Parse arguments immediately
args = parse_arguments()

# Set CUDA_VISIBLE_DEVICES early from cuda_devices argument
os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices
print(f"Set CUDA_VISIBLE_DEVICES={args.cuda_devices}")

# Now import torch and other CUDA-dependent libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import glob
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
optuna.logging.set_verbosity(optuna.logging.INFO)

# Add utils directory to path
sys.path.append(str(Path(__file__).parent / "../utils"))
from general import seed_everything
from eval import calculate_all_metrics

RANDOM_SEED = args.seed if args.seed is not None else 23
seed_everything(RANDOM_SEED)

# Load optuna config
def load_optuna_config(config_path: str, method_key: str) -> Dict[str, Any]:
    """Load optuna configuration from JSON file."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    if method_key not in config['methods']:
        raise ValueError(f"Method {method_key} not found in config file")
    
    method_config = config['methods'][method_key]
    budget_constraints = config['budget_constraints']
    shared_classifier_space = config['shared_classifier_space']
    
    logger.info(f"Loaded config for {method_key}:")
    logger.info(f"Available keys: {list(method_config.keys())}")
    
    return {
        'method_config': method_config,
        'budget_constraints': budget_constraints,
        'shared_classifier_space': shared_classifier_space
    }

# Load optuna config
OPTUNA_CONFIG = load_optuna_config(args.optuna_config, args.optuna_key)

# Load shared classifier space
SHARED_CLASSIFIER_LAYERS = OPTUNA_CONFIG['shared_classifier_space']['classifier_layers']
SHARED_CLASSIFIER_DROPOUT = OPTUNA_CONFIG['shared_classifier_space']['classifier_dropout']

# Update hyperparameter space based on optuna config
HYPERPARAMETER_SPACE = {
    'learning_rate': {
        'type': 'float',
        'low': OPTUNA_CONFIG['method_config']['learning_rate'][0],
        'high': OPTUNA_CONFIG['method_config']['learning_rate'][1],
        'log': True
    },
    'weight_decay': {
        'type': 'float',
        'low': OPTUNA_CONFIG['method_config']['weight_decay'][0],
        'high': OPTUNA_CONFIG['method_config']['weight_decay'][1],
        'log': True
    },
    'dropout': {
        'type': 'categorical',
        'choices': tuple(SHARED_CLASSIFIER_DROPOUT)
    },
    'classifier_layers': {
        'type': 'categorical',
        'choices': tuple(SHARED_CLASSIFIER_LAYERS)
    }
}

logger.info(f"Loaded hyperparameter space:")
logger.info(json.dumps(HYPERPARAMETER_SPACE, indent=2))

# Constant batch size
BATCH_SIZE = 32

def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def calculate_composite_score(auroc: float, ece: float, auroc_weight: float = 0.5) -> float:
    """Calculate composite validation score balancing AUROC and ECE."""
    ece_weight = 1.0 - auroc_weight
    ece_score = 1.0 - ece
    composite_score = auroc_weight * auroc + ece_weight * ece_score
    return composite_score

def calculate_auroc_with_ece_constraint(auroc: float, ece: float) -> float:
    """Calculate AUROC with ECE penalty."""
    penalty_factor = 1.0 - ece
    return auroc * penalty_factor

def pick_tau_by_youden(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Pick optimal threshold by maximizing Youden's J statistic."""
    taus = np.linspace(0.0, 1.0, 1001)
    best_j, best_tau = -1.0, 0.5
    
    for t in taus:
        yhat = (y_prob >= t).astype(int)
        tp = np.sum((yhat == 1) & (y_true == 1))
        tn = np.sum((yhat == 0) & (y_true == 0))
        fp = np.sum((yhat == 1) & (y_true == 0))
        fn = np.sum((yhat == 0) & (y_true == 1))
        sens = tp / (tp + fn + 1e-12)
        spec = tn / (tn + fp + 1e-12)
        j = sens + spec - 1.0
        if j > best_j:
            best_j, best_tau = j, t
    
    return float(best_tau)

def sens_spec_at_tau(y_true: np.ndarray, y_prob: np.ndarray, tau: float) -> Tuple[float, float]:
    """Calculate sensitivity and specificity at a given threshold."""
    yhat = (y_prob >= tau).astype(int)
    tp = np.sum((yhat == 1) & (y_true == 1))
    tn = np.sum((yhat == 0) & (y_true == 0))
    fp = np.sum((yhat == 1) & (y_true == 0))
    fn = np.sum((yhat == 0) & (y_true == 1))
    sens = tp / (tp + fn + 1e-12)
    spec = tn / (tn + fp + 1e-12)
    return float(sens), float(spec)

def calculate_composite_with_constraints(metrics: Dict[str, float], y_true: np.ndarray, y_prob: np.ndarray,
                                      alpha: float = 0.6, min_sens: float = 0.60, min_spec: float = 0.60) -> Tuple[float, bool, Dict[str, float]]:
    """Calculate composite score with sensitivity/specificity constraints."""
    auroc = metrics["auroc"]
    ece = metrics["ece"]
    score = alpha * auroc + (1 - alpha) * (1.0 - ece)
    
    tau = pick_tau_by_youden(y_true, y_prob)
    sens, spec = sens_spec_at_tau(y_true, y_prob, tau)
    feasible = (sens >= min_sens) and (spec >= min_spec)
    
    return score, feasible, {"tau": tau, "sens": sens, "spec": spec}

class PIKDataset:
    """Dataset for PIK training with GPU pre-loading (optimized like ICC3Dataset)."""
    
    def __init__(self, hidden_states: np.ndarray, labels: np.ndarray):
        """Initialize dataset and move to GPU immediately."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Convert to tensors and move to GPU immediately
        self.hidden_states = torch.FloatTensor(hidden_states).to(self.device, non_blocking=True)
        self.labels = torch.FloatTensor(labels.astype(float)).to(self.device, non_blocking=True)
        
        logger.info(f"[PIK] Loaded {len(self.hidden_states)} samples to {self.device}")
        logger.info(f"[PIK] Hidden states shape: {self.hidden_states.shape}, Labels shape: {self.labels.shape}")
    
    def __len__(self):
        return len(self.hidden_states)

class PIKModel(nn.Module):
    """PIK MLP model for classifying question understanding representations.
    
    Uses simple feedforward architecture with configurable hidden layers.
    """
    
    def __init__(self, input_dim: int, hidden_layers: Tuple[int, ...] = (256, 128, 64), dropout: float = 0.0):
        """Initialize PIK model.
        
        Args:
            input_dim: Input dimension
            hidden_layers: Tuple of hidden layer dimensions
            dropout: Dropout probability
        """
        super(PIKModel, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_layers = tuple(hidden_layers) if hidden_layers else ()
        
        if not self.hidden_layers:  # Linear probe
            self.classifier = nn.Linear(input_dim, 1)
        else:
            # Build dynamic sequential model
            layers = []
            current_dim = input_dim
            
            for hidden_dim in self.hidden_layers:
                layers.extend([
                    nn.Linear(current_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                ])
                current_dim = hidden_dim
            
            # Add final output layer
            layers.append(nn.Linear(current_dim, 1))
            
            self.classifier = nn.Sequential(*layers)
    
    def forward(self, x):
        """Forward pass through the model."""
        logits = self.classifier(x)
        return logits.squeeze(-1)

class PIKTrainer:
    """PIK model trainer with Optuna optimization capabilities."""
    
    def __init__(self, device: str = 'cuda'):
        """Initialize trainer."""
        if torch.cuda.is_available() and device.startswith('cuda'):
            if device == 'cuda':
                self.device = 'cuda:0'
            else:
                self.device = device
        else:
            self.device = 'cpu'
        logger.info(f"Using device: {self.device}")
        
        if self.device.startswith('cuda'):
            gpu_idx = int(self.device.split(':')[1]) if ':' in self.device else 0
            logger.info(f"GPU memory: {torch.cuda.get_device_properties(gpu_idx).total_memory / 1024**3:.2f}GB")
        
        self.validation_metric = 'composite'
        self.composite_auroc_weight = 0.5
        self.min_sens = 0.60
        self.min_spec = 0.60
        
        # Store budget constraints
        self.min_params = OPTUNA_CONFIG['budget_constraints']['min_params']
        self.max_params = OPTUNA_CONFIG['budget_constraints']['max_params']
        logger.info(f"Parameter budget constraints: [{self.min_params:,}, {self.max_params:,}]")
    
    def calculate_validation_score(self, val_metrics: Dict[str, Any], y_true: Optional[np.ndarray] = None, 
                                    y_prob: Optional[np.ndarray] = None) -> float:
        """Calculate validation score based on the selected metric."""
        auroc = val_metrics['auroc']
        ece = val_metrics['ece']
        
        if self.validation_metric == 'auroc':
            return auroc
        elif self.validation_metric == 'ece':
            return 1.0 - ece
        elif self.validation_metric == 'composite':
            return calculate_composite_score(auroc, ece, self.composite_auroc_weight)
        elif self.validation_metric == 'auroc_with_constraint':
            return calculate_auroc_with_ece_constraint(auroc, ece)
        elif self.validation_metric == 'validation_loss':
            return -val_metrics.get('loss', 0.0)
        elif self.validation_metric == 'composite_with_constraints':
            if y_true is None or y_prob is None:
                raise ValueError("y_true and y_prob must be provided for composite_with_constraints metric")
            score, feasible, _ = calculate_composite_with_constraints(
                val_metrics, y_true, y_prob,
                alpha=self.composite_auroc_weight,
                min_sens=self.min_sens,
                min_spec=self.min_spec
            )
            return score if feasible else float('-inf')
        else:
            raise ValueError(f"Unknown validation metric: {self.validation_metric}")
    
    def set_validation_metric(self, metric: str, auroc_weight: float = 0.5, 
                            min_sens: float = 0.60, min_spec: float = 0.60):
        """Set validation metric configuration."""
        self.validation_metric = metric
        self.composite_auroc_weight = auroc_weight
        self.min_sens = min_sens
        self.min_spec = min_spec
        logger.info(f"Validation metric set to: {metric}")
        if metric == 'composite':
            logger.info(f"Composite weights: AUROC={auroc_weight:.1f}, ECE={1-auroc_weight:.1f}")
        elif metric == 'composite_with_constraints':
            logger.info(f"Composite weights: AUROC={auroc_weight:.1f}, ECE={1-auroc_weight:.1f}")
            logger.info(f"Sensitivity floor: {min_sens:.2f}")
            logger.info(f"Specificity floor: {min_spec:.2f}")
    
    def load_data_from_samples(self, samples_dir: Path) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Load FIRST token hidden states and labels from samples directory.

        PIK uses the first generated token's hidden state (after seeing the question
        and before producing the response).

        Label convention is fixed: 1 = correct, 0 = incorrect. Class imbalance is
        handled by BCE pos_weight = n_neg / n_pos at training time.
        """
        hidden_states_list = []
        labels_list = []
        sample_ids_list = []

        npz_files = list(samples_dir.glob("*.npz"))
        logger.info(f"Loading data from {len(npz_files)} files in {samples_dir}")

        skipped_no_correctness = 0
        skipped_errors = 0

        for npz_file in tqdm(npz_files, desc="Loading samples"):
            try:
                data = np.load(npz_file, allow_pickle=True)
                hidden_states = data['hidden_states']
                first_hidden_state = hidden_states[0]
                is_correct = data['is_correct']
                if is_correct is None or (isinstance(is_correct, np.ndarray) and is_correct.item() is None):
                    skipped_no_correctness += 1
                    continue
                hash_id = str(data['hash_id'])
                hidden_states_list.append(first_hidden_state)
                labels_list.append(bool(is_correct))
                sample_ids_list.append(hash_id)
            except Exception as e:
                skipped_errors += 1
                if skipped_errors <= 5:
                    logger.warning(f"Error loading {npz_file.name}: {e}")
                continue

        if skipped_no_correctness > 0:
            logger.info(f"Skipped {skipped_no_correctness} samples without correctness assessment")
        if skipped_errors > 0:
            logger.warning(f"Skipped {skipped_errors} samples due to errors")
        if not hidden_states_list:
            raise ValueError(f"No valid data found in {samples_dir}")

        all_hidden_states = np.stack(hidden_states_list, axis=0)
        all_labels = np.array(labels_list, dtype=bool)

        n_pos = int(np.sum(all_labels))
        n_neg = int(len(all_labels) - n_pos)
        logger.info(f"Loaded {len(all_hidden_states)} samples | hidden_dim={all_hidden_states.shape[1]} | "
                    f"correct={n_pos}  incorrect={n_neg}")
        return all_hidden_states, all_labels, sample_ids_list
    
    def calculate_pos_weight_ratio(self, labels: np.ndarray) -> float:
        """Calculate the ratio of class 0 (majority) to class 1 (minority) for pos_weight."""
        n_class_1 = np.sum(labels)  # Class 1 (minority)
        n_class_0 = len(labels) - n_class_1  # Class 0 (majority)
        ratio = n_class_0 / n_class_1 if n_class_1 > 0 else 1.0
        logger.info(f"Class 0 (majority) samples: {n_class_0}, Class 1 (minority) samples: {n_class_1}")
        logger.info(f"Pos weight ratio (class_0/class_1): {ratio:.3f}")
        return ratio
    
    
    def train_model_with_pruning(self, model: PIKModel, train_dataset: PIKDataset, val_dataset: PIKDataset,
                                trial, pos_weight_ratio: float, learning_rate: float, 
                                weight_decay: float, batch_size: int = 32, max_epochs: int = 200, val_eval_frequency: int = 1) -> Tuple[PIKModel, Dict[str, Any]]:
        """Train a PIK model with early stopping and Optuna pruning (optimized with direct tensor indexing)."""
        logger.info(f"Trial {trial.number}: Moving model to device {self.device}...")
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        device = torch.device(self.device)
        model = model.to(device)
        
        # Initialize loss function and optimizer
        pos_weight = torch.tensor(pos_weight_ratio, device=device, dtype=torch.float)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_metrics': [],
            'best_epoch': None,
            'best_operating_point': None
        }
        
        best_val_score = float('-inf')
        best_model_state = None
        best_operating_point = None
        patience = 20
        patience_counter = 0
        
        n_train = len(train_dataset)
        n_val = len(val_dataset)
        
        for epoch in range(max_epochs):
            # Training phase - use direct tensor indexing (like ICC3)
            model.train()
            train_loss = 0.0
            train_samples = 0
            
            # GPU-native shuffle
            perm = torch.randperm(n_train, device=device)
            
            for i in range(0, n_train, batch_size):
                idx = perm[i:i + batch_size]
                
                batch_hidden_states = train_dataset.hidden_states[idx]
                batch_labels = train_dataset.labels[idx]
                
                optimizer.zero_grad()
                
                predictions = model(batch_hidden_states)
                loss = criterion(predictions, batch_labels)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                B = batch_hidden_states.shape[0]
                train_loss += loss.item() * B
                train_samples += B
            
            avg_train_loss = train_loss / train_samples
            
            # Validation phase - use direct tensor indexing
            if epoch % val_eval_frequency == 0:
                model.eval()
                val_loss = 0.0
                val_samples = 0
                all_val_preds = []
                all_val_labels = []
                
                with torch.no_grad():
                    for i in range(0, n_val, batch_size):
                        idx = slice(i, i + batch_size)
                        
                        batch_hidden_states = val_dataset.hidden_states[idx]
                        batch_labels = val_dataset.labels[idx]
                        
                        logits = model(batch_hidden_states)
                        loss = criterion(logits, batch_labels)
                        
                        B = batch_hidden_states.shape[0]
                        val_loss += loss.item() * B
                        val_samples += B
                        
                        probabilities = torch.sigmoid(logits)
                        all_val_preds.append(probabilities.cpu())
                        all_val_labels.append(batch_labels.cpu())
                
                avg_val_loss = val_loss / val_samples
                all_val_preds = torch.cat(all_val_preds).numpy()
                all_val_labels = torch.cat(all_val_labels).numpy()
                
                val_metrics = calculate_all_metrics(all_val_labels, all_val_preds)
                val_metrics['loss'] = avg_val_loss
                
                # Compute operating point metrics only for composite_with_constraints
                operating_point = None
                if self.validation_metric == 'composite_with_constraints':
                    tau = pick_tau_by_youden(all_val_labels, all_val_preds)
                    sens, spec = sens_spec_at_tau(all_val_labels, all_val_preds, tau)
                    operating_point = {"tau": tau, "sens": sens, "spec": spec}
                    
                    # Check feasibility for composite_with_constraints
                    if epoch >= 2:
                        feasible = (sens >= self.min_sens) and (spec >= self.min_spec)
                        if not feasible:
                            logger.info(f"Trial {trial.number}: Epoch {epoch}: Trial pruned (infeasible)")
                            raise optuna.TrialPruned()
                
                # Calculate validation score
                val_score = self.calculate_validation_score(val_metrics, all_val_labels, all_val_preds)
                
                history['train_loss'].append(avg_train_loss)
                history['val_loss'].append(avg_val_loss)
                history['val_metrics'].append(val_metrics)
                
                # Print performance metrics every epoch
                logger.info(f"Epoch {epoch:3d}: Train Loss: {avg_train_loss:.4f}, "
                           f"Val Loss: {avg_val_loss:.4f}, Val AUROC: {val_metrics['auroc']:.4f}, "
                           f"ECE: {val_metrics['ece']:.4f}, Brier: {val_metrics['brier']:.4f}, "
                           f"Val Score: {val_score:.4f}")
                if operating_point is not None:
                    logger.info(f"  Operating point - tau: {operating_point['tau']:.3f}, sens: {operating_point['sens']:.3f}, spec: {operating_point['spec']:.3f}")
            else:
                # Skip validation this epoch, just track training loss
                history['train_loss'].append(avg_train_loss)
                val_score = float('-inf')  # Don't update best model if skipping validation
                logger.info(f"Epoch {epoch:3d}: Train Loss: {avg_train_loss:.4f} (validation skipped)")
            
            # Report to Optuna (only when validation was performed)
            if epoch % val_eval_frequency == 0:
                trial.report(val_score, epoch)
                
                if trial.should_prune():
                    logger.info(f"Trial {trial.number}: Epoch {epoch}: Trial pruned")
                    raise optuna.TrialPruned()
                
                # Early stopping
                if val_score > best_val_score:
                    best_val_score = val_score
                    best_model_state = copy.deepcopy(model.state_dict())
                    patience_counter = 0
                    history['best_epoch'] = epoch
                    if operating_point is not None:
                        best_operating_point = operating_point
                        history['best_operating_point'] = operating_point
                    logger.info(f"  *** New best model! Val Score: {val_score:.4f} ***")
                else:
                    patience_counter += 1
            
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
        
        # Load best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        return model, history
    
    def evaluate_model(self, model: PIKModel, dataset: PIKDataset, batch_size: int = 32, return_arrays: bool = False):
        """Evaluate model on a dataset (optimized with direct tensor indexing)."""
        model.eval()
        all_preds = []
        all_labels = []
        
        n_samples = len(dataset)
        
        with torch.no_grad():
            for i in range(0, n_samples, batch_size):
                idx = slice(i, i + batch_size)
                batch_hidden_states = dataset.hidden_states[idx]
                batch_labels = dataset.labels[idx]
                
                logits = model(batch_hidden_states)
                probabilities = torch.sigmoid(logits)
                all_preds.append(probabilities.cpu())
                all_labels.append(batch_labels.cpu())
        
        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()
        metrics = calculate_all_metrics(all_labels, all_preds)
        
        if return_arrays:
            return metrics, all_labels, all_preds
        return metrics
    
    def save_best_model(self, model: PIKModel, config: Dict[str, Any],
                       metrics: Dict[str, Any], output_dir: Path):
        """Save the best model and its configuration."""
        best_dir = output_dir / "best"
        best_dir.mkdir(parents=True, exist_ok=True)

        torch.save(model.state_dict(), best_dir / "model.pth")

        n_params = count_parameters(model)

        model_info = {
            'input_dim': model.input_dim,
            'hidden_layers': list(model.hidden_layers),
            'config': config,
            'metrics': metrics,
            'parameter_info': {
                'n_params': n_params,
                'budget_constraints': {
                    'min_params': self.min_params,
                    'max_params': self.max_params
                }
            }
        }

        with open(best_dir / "model_info.json", 'w') as f:
            json.dump(model_info, f, indent=2)

        logger.info(f"Best model saved to: {best_dir}")
        logger.info(f"Model parameters: {n_params:,}")
    
    def _sample_hyperparameters(self, trial) -> Dict[str, Any]:
        """Sample hyperparameters using Optuna trial."""
        config = {}
        
        config['learning_rate'] = trial.suggest_float(
            'learning_rate',
            HYPERPARAMETER_SPACE['learning_rate']['low'],
            HYPERPARAMETER_SPACE['learning_rate']['high'],
            log=True
        )
        
        config['weight_decay'] = trial.suggest_float(
            'weight_decay',
            HYPERPARAMETER_SPACE['weight_decay']['low'],
            HYPERPARAMETER_SPACE['weight_decay']['high'],
            log=True
        )
        
        # Sample classifier_layers from shared_classifier_space
        classifier_layers_str = trial.suggest_categorical(
            'classifier_layers',
            HYPERPARAMETER_SPACE['classifier_layers']['choices']
        )
        
        # Store classifier_layers string in config
        config['classifier_layers'] = classifier_layers_str
        
        # Parse classifier_layers string to tuple (e.g., "128,64" -> (128, 64), "0" -> ())
        if classifier_layers_str == "0":
            config['hidden_layers'] = ()
        else:
            config['hidden_layers'] = tuple(int(x.strip()) for x in classifier_layers_str.split(",") if x.strip())
        
        config['dropout'] = trial.suggest_categorical(
            'dropout',
            HYPERPARAMETER_SPACE['dropout']['choices']
        )
        
        return config
    
    def optuna_search(self, train_hidden_states: np.ndarray, train_labels: np.ndarray,
                     val_hidden_states: np.ndarray, val_labels: np.ndarray,
                     output_dir: Path, n_trials: int = 20, n_jobs: int = 1,
                     val_eval_frequency: int = 1):
        """Perform Optuna optimization (in-memory only)."""

        pos_weight_ratio = self.calculate_pos_weight_ratio(train_labels)

        train_dataset = PIKDataset(train_hidden_states, train_labels)
        val_dataset = PIKDataset(val_hidden_states, val_labels)

        models_dir = output_dir / "models"
        models_dir.mkdir(parents=True, exist_ok=True)

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.pos_weight_ratio = pos_weight_ratio
        self.input_dim = train_hidden_states.shape[1]
        self.models_dir = models_dir
        self.output_dir = output_dir
        
        def objective(trial):
            try:
                config = self._sample_hyperparameters(trial)
                logger.info(f"\nTrial {trial.number}: {config}")
                
                model = PIKModel(self.input_dim, config['hidden_layers'], config['dropout'])
                
                n_params = count_parameters(model)
                if n_params > self.max_params or n_params < self.min_params:
                    logger.info(f"Trial {trial.number}: Outside budget [{self.min_params:,}, {self.max_params:,}]")
                    return float('-inf')
                
                logger.info(f"Trial {trial.number}: {n_params:,} parameters")
                
                trained_model, history = self.train_model_with_pruning(
                    model, self.train_dataset, self.val_dataset, trial,
                    self.pos_weight_ratio, config['learning_rate'], config['weight_decay'],
                    batch_size=BATCH_SIZE, val_eval_frequency=val_eval_frequency
                )
                
                val_metrics, val_labels, val_preds = self.evaluate_model(trained_model, self.val_dataset, batch_size=BATCH_SIZE, return_arrays=True)
                val_score = self.calculate_validation_score(val_metrics, val_labels, val_preds)
                
                logger.info(f"Trial {trial.number} metrics:")
                for metric, value in val_metrics.items():
                    logger.info(f"  {metric}: {value:.4f}")
                logger.info(f"  Score ({self.validation_metric}): {val_score:.4f}")
                
                # Save trial
                model_dir = self.models_dir / f"trial_{trial.number:03d}"
                model_dir.mkdir(exist_ok=True)
                
                torch.save(trained_model.state_dict(), model_dir / "model.pth")
                
                model_info = {
                    'input_dim': trained_model.input_dim,
                    'classifier_layers': config['classifier_layers'],
                    'hidden_layers': list(trained_model.hidden_layers),
                    'config': config,
                    'metrics': val_metrics,
                    'trial_number': trial.number,
                    'best_epoch': history.get('best_epoch'),
                    'parameter_info': {'n_params': n_params}
                }

                with open(model_dir / "model_info.json", 'w') as f:
                    json.dump(model_info, f, indent=2)

                with open(model_dir / "results.json", 'w') as f:
                    json.dump({'trial_number': trial.number, 'config': config,
                              'val_metrics': val_metrics, 'val_score': val_score}, f, indent=2)

                self.save_best_model(trained_model, config, val_metrics, model_dir)
                
                return val_score
                
            except optuna.TrialPruned:
                raise
            except Exception as e:
                logger.error(f"Error in trial {trial.number}: {e}")
                import traceback
                logger.error(traceback.format_exc())
                return float('-inf')
            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        sampler = TPESampler(seed=RANDOM_SEED)
        pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10, interval_steps=5)
        
        # Create in-memory study (no storage)
        study = optuna.create_study(
            direction="maximize", sampler=sampler, pruner=pruner
        )
        
        logger.info(f"Starting Optuna with {n_trials} trials")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True, n_jobs=n_jobs)
        
        # Find best trial
        best_trial_dir = None
        best_trial_score = float('-inf')
        
        for trial_dir in sorted(self.models_dir.glob("trial_*")):
            try:
                with open(trial_dir / "results.json", 'r') as f:
                    results = json.load(f)
                    if results['val_score'] > best_trial_score:
                        best_trial_score = results['val_score']
                        best_trial_dir = trial_dir
            except:
                pass
        
        if best_trial_dir is None:
            raise RuntimeError("No valid trials found")
        
        # Copy best model
        best_dir = output_dir / "best"
        if best_dir.exists():
            import shutil
            shutil.rmtree(best_dir)
        best_dir.mkdir(parents=True, exist_ok=True)
        
        import shutil
        for file in best_trial_dir.glob("*"):
            if file.is_file():
                shutil.copy2(file, best_dir / file.name)
        
        with open(best_dir / "model_info.json", 'r') as f:
            best_model_info = json.load(f)
        
        # Save optuna results
        with open(output_dir / "optuna_results.json", 'w') as f:
            json.dump({
                'best_params': study.best_params,
                'best_value': study.best_value,
                'n_trials': len(study.trials)
            }, f, indent=2)
        
        logger.info(f"\nOptimization completed!")
        logger.info(f"Best trial: {study.best_trial.number}")
        logger.info(f"Best score: {study.best_value:.4f}")
        
        best_model = PIKModel(self.input_dim, tuple(best_model_info['hidden_layers']), 
                             best_model_info['config']['dropout'])
        best_model.load_state_dict(torch.load(best_dir / "model.pth"))
        
        return best_model_info['config'], best_model, best_model_info['metrics']

def main():
    logger.info(f"PIK Training Script")
    logger.info(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
    
    if torch.cuda.is_available():
        logger.info(f"CUDA available: {torch.cuda.device_count()} devices")
        for i in range(torch.cuda.device_count()):
            logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # Use cuda:0 (first visible device after setting CUDA_VISIBLE_DEVICES)
    device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    trainer = PIKTrainer(device=device_str)
    trainer.set_validation_metric(args.validation_metric, args.composite_auroc_weight,
                                  min_sens=args.min_sensitivity, min_spec=args.min_specificity)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Training PIK model")
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Train Dataset: {args.train_dataset_name}")
    logger.info(f"Val Dataset: {args.val_dataset_name}")
    logger.info(f"{'='*60}\n")
    
    # Construct paths for train and validation datasets
    # Extract model name (last part after splitting by "/")
    model_name_part = args.model_name.split("/")[-1]
    data_dir = Path(args.data_dir)
    train_samples_dir = data_dir / model_name_part / args.train_dataset_name / "samples"
    val_samples_dir = data_dir / model_name_part / args.val_dataset_name / "samples"
    
    if not train_samples_dir.exists():
        logger.error(f"Train samples directory not found: {train_samples_dir}")
        return
    
    if not val_samples_dir.exists():
        logger.error(f"Validation samples directory not found: {val_samples_dir}")
        return
    
    logger.info(f"Loading train data from {train_samples_dir}")
    train_hs, train_labels, train_ids = trainer.load_data_from_samples(train_samples_dir)

    logger.info(f"Loading validation data from {val_samples_dir}")
    val_hs, val_labels, val_ids = trainer.load_data_from_samples(val_samples_dir)

    if args.seed is not None:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(args.output_dir) / model_name_part / args.train_dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Output: {output_dir}")
    logger.info(f"Trials: {args.n_trials}")

    best_config, best_model, best_metrics = trainer.optuna_search(
        train_hs, train_labels, val_hs, val_labels, output_dir,
        n_trials=args.n_trials, n_jobs=args.n_jobs,
        val_eval_frequency=args.val_eval_frequency,
    )
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Training completed!")
    logger.info(f"{'='*60}\n")

if __name__ == "__main__":
    main()

# Qwen/Qwen3-VL-8B-Instruct
# llava-hf/llava-v1.6-vicuna-13b-hf
# OpenGVLab/InternVL3_5-14B-HF
# google/gemma-3-27b-it
# deepseek-ai/deepseek-vl2

## Example Usage:
# python PIK_train.py \
#   --data-dir ../data/extraction/raw/ \
#   --model-name Qwen/Qwen3-VL-8B-Instruct \
#   --train-dataset-name train \
#   --val-dataset-name validation \
#   --output-dir ../trained_models/PIK_t50/ \
#   --n-trials 50 \
#   --cuda-devices 0 \
#   --validation-metric composite \
#   --composite-auroc-weight 0.6 \
#   --min-sensitivity 0.5 \
#   --min-specificity 0.5 \
#   --n-jobs 1 \
#   --val-eval-frequency 1