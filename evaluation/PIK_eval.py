#!/usr/bin/env python3
"""
Comprehensive PIK evaluation script.
Loads trained PIK models and evaluates them on test data for question understanding classification.
"""

import argparse
import json
import sys
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import logging

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from datasets import load_from_disk

# Add utils directory to path for eval functions
sys.path.append(str(Path(__file__).parent / "../utils"))
from eval import evaluate_by_groups, save_evaluation_results

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PIKModel(nn.Module):
    """PIK MLP model for classifying question understanding representations."""
    
    def __init__(self, input_dim: int, hidden_layers: Union[str, Tuple[int, ...], List[int]], dropout: float = 0.0):
        """Initialize PIK model.
        
        Args:
            input_dim: Input dimension
            hidden_layers: Layer configuration as string ("256,128,64") or sequence of dimensions
            dropout: Dropout probability
        """
        super(PIKModel, self).__init__()
        self.input_dim = input_dim
        
        # Parse hidden layers if string
        if isinstance(hidden_layers, str):
            if hidden_layers.strip() and hidden_layers.strip() != "0":
                self.hidden_layers = tuple(int(x) for x in hidden_layers.split(","))
            else:
                self.hidden_layers = ()  # Linear probe
        else:
            self.hidden_layers = tuple(hidden_layers) if isinstance(hidden_layers, (list, tuple)) else ()
        
        if not self.hidden_layers:
            # Linear probe
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


class PIKEvaluator:
    def __init__(self, model_path: Path, data_dir: Path, model_name: str, test_dataset_name: str, dataset_path: Optional[Path] = None):
        """Initialize the PIK evaluator.
        
        Args:
            model_path: Path to trained model directory
            data_dir: Directory containing extracted representations
            model_name: Model name (e.g., Qwen2.5-VL-3B-Instruct)
            test_dataset_name: Test dataset name (e.g., VLCB_test)
            dataset_path: Optional path to original dataset directory (for accessing dataset property)
        """
        self.model_path = model_path
        self.data_dir = data_dir
        self.model_name = model_name
        self.test_dataset_name = test_dataset_name
        self.dataset_path = dataset_path
        
        # Load trained model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model, self.model_info = self._load_trained_model()
        
        if self.model:
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"PIK model loaded and moved to device: {self.device}")
        
        # Load dataset mapping if dataset_path is provided
        self.hash_id_to_dataset = {}
        if self.dataset_path:
            self._load_dataset_mapping()
    
    def _load_trained_model(self) -> Tuple[Optional[PIKModel], Optional[Dict[str, Any]]]:
        """Load the best trained PIK model."""
        best_model_path = self.model_path / "best"
        
        if not best_model_path.exists():
            logger.error(f"Trained model directory not found: {best_model_path}")
            return None, None
        
        # Load model info
        model_info_path = best_model_path / "model_info.json"
        if not model_info_path.exists():
            logger.error(f"Model info file not found: {model_info_path}")
            return None, None
        
        with open(model_info_path, 'r') as f:
            model_info = json.load(f)
        
        # Create model with correct architecture
        input_dim = model_info['input_dim']
        # Support both new format (classifier_layers) and legacy format (hidden_layers)
        if 'classifier_layers' in model_info:
            # New format: classifier_layers string from shared_classifier_space
            classifier_layers_str = model_info['classifier_layers']
            hidden_layers = classifier_layers_str  # Pass as string, will be parsed
            logger.info(f"Loading PIK model (new format): input_dim={input_dim}, classifier_layers={classifier_layers_str}")
        elif 'hidden_layers' in model_info:
            # Legacy format: tuple/list of hidden layer dimensions
            hidden_layers = model_info['hidden_layers']
            logger.info(f"Loading PIK model (legacy format): input_dim={input_dim}, hidden_layers={hidden_layers}")
        else:
            raise KeyError(f"Model info missing both 'classifier_layers' and 'hidden_layers' keys. Available keys: {list(model_info.keys())}")
        
        dropout = model_info['config']['dropout']
        
        logger.info(f"Loading PIK model with input_dim={input_dim}, hidden_layers={hidden_layers}, dropout={dropout}")
        
        model = PIKModel(input_dim, hidden_layers, dropout)
        
        # Load state dict
        state_dict_path = best_model_path / "model.pth"
        if not state_dict_path.exists():
            logger.error(f"Model state dict not found: {state_dict_path}")
            return None, None
        
        model.load_state_dict(torch.load(state_dict_path, map_location='cpu'))
        
        logger.info(f"Best model config: {model_info['config']}")
        logger.info(f"Best model metrics: {model_info['metrics']}")
        return model, model_info
    
    def _load_dataset_mapping(self):
        """Load dataset and create hash_id -> dataset mapping."""
        try:
            # Try multiple possible paths
            possible_paths = [
                self.dataset_path / self.test_dataset_name,
                self.dataset_path / "raw" / self.test_dataset_name,
                self.dataset_path / self.test_dataset_name.replace("_raw", ""),
            ]
            
            full_dataset_path = None
            for path in possible_paths:
                if path.exists():
                    full_dataset_path = path
                    break
            
            if not full_dataset_path:
                logger.warning(f"Dataset path not found. Tried: {possible_paths}. Dataset property will not be included.")
                return
            
            dataset = load_from_disk(str(full_dataset_path))
            logger.info(f"Loading dataset mapping from {full_dataset_path}")
            
            # Create hash_id -> dataset mapping
            for idx in range(len(dataset)):
                sample = dataset[idx]
                hash_id = sample.get('hash_id')
                dataset_name = sample.get('dataset', 'unknown')
                if hash_id:
                    # Convert to string to match the format used in load_test_data()
                    self.hash_id_to_dataset[str(hash_id)] = dataset_name
            
            logger.info(f"Loaded dataset mapping for {len(self.hash_id_to_dataset)} samples")
            if len(self.hash_id_to_dataset) == 0:
                logger.warning("Dataset mapping is empty! Check if dataset has 'hash_id' and 'dataset' fields.")
        except Exception as e:
            logger.warning(f"Failed to load dataset mapping: {e}. Dataset property will not be included.")
            import traceback
            logger.warning(traceback.format_exc())
            self.hash_id_to_dataset = {}
    
    def load_test_data(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Load test hidden states and labels from samples directory.
        
        Returns:
            hidden_states: [N, hidden_dim] - first token hidden states
            labels: [N] - correctness labels (as stored)
            sample_ids: [N] - sample IDs for tracking
        """
        # Extract model name part (last part after splitting by "/")
        model_name_part = self.model_name.split("/")[-1]
        # Construct test samples path
        samples_dir = self.data_dir / model_name_part / self.test_dataset_name / "samples"
        
        if not samples_dir.exists():
            logger.error(f"Test samples directory not found: {samples_dir}")
            return None, None, None
        
        hidden_states_list = []
        labels_list = []
        sample_ids_list = []
        
        npz_files = list(samples_dir.glob("*.npz"))
        logger.info(f"Loading test data from {len(npz_files)} files in {samples_dir}")
        
        skipped_no_correctness = 0
        skipped_errors = 0
        
        for npz_file in tqdm(npz_files, desc="Loading test samples"):
            try:
                data = np.load(npz_file, allow_pickle=True)
                
                # PIK: Extract FIRST token hidden state
                hidden_states = data['hidden_states']  # Shape: [num_tokens, hidden_dim]
                first_hidden_state = hidden_states[0]  # FIRST token's hidden state
                
                # Extract correctness label
                is_correct = data['is_correct']
                
                # Skip if no correctness assessment
                if is_correct is None or (isinstance(is_correct, np.ndarray) and is_correct.item() is None):
                    skipped_no_correctness += 1
                    continue
                
                # Extract sample ID - handle numpy arrays/scalars
                hash_id_raw = data['hash_id']
                if isinstance(hash_id_raw, np.ndarray):
                    hash_id = str(hash_id_raw.item() if hash_id_raw.shape == () else hash_id_raw)
                else:
                    hash_id = str(hash_id_raw)
                
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
            logger.error(f"No valid data found in {samples_dir}")
            return None, None, None
        
        # Stack all data — labels are always {0=incorrect, 1=correct}.
        all_hidden_states = np.stack(hidden_states_list, axis=0)
        all_labels = np.array(labels_list, dtype=bool)

        logger.info(f"Loaded {len(all_hidden_states)} test samples, "
                    f"hidden_states shape={all_hidden_states.shape}")
        logger.info(f"correct={int(np.sum(all_labels == 1))}  "
                    f"incorrect={int(np.sum(all_labels == 0))}")
        
        return all_hidden_states, all_labels, sample_ids_list
    
    def apply_pik_classification(self, hidden_states: np.ndarray) -> np.ndarray:
        """Apply PIK classification to hidden states.
        
        Args:
            hidden_states: [N, hidden_dim] array of hidden states
            
        Returns:
            confidence_scores: [N] array of confidence scores (probability of class 1)
        """
        if not self.model:
            logger.error("PIK model not loaded")
            return None
        
        # Convert to tensor
        hidden_states_tensor = torch.FloatTensor(hidden_states).to(self.device)
        
        # Get predictions in batches
        batch_size = 32
        all_confidences = []
        
        with torch.no_grad():
            for i in range(0, len(hidden_states_tensor), batch_size):
                batch = hidden_states_tensor[i:i+batch_size]
                logits = self.model(batch)
                confidences = torch.sigmoid(logits)
                all_confidences.append(confidences.cpu().numpy())
        
        # Concatenate all batches
        confidence_scores = np.concatenate(all_confidences)
        
        return confidence_scores
    
    def evaluate(self, output_dir: Path) -> Dict[str, Any]:
        """Run evaluation on test data.
        
        Args:
            output_dir: Directory to save evaluation results
            
        Returns:
            Dictionary containing evaluation results
        """
        logger.info("Starting PIK evaluation...")
        
        # Load test data
        hidden_states, labels, sample_ids = self.load_test_data()
        
        if hidden_states is None:
            logger.error("Failed to load test data")
            return None
        
        # Apply PIK classification
        logger.info("Applying PIK classification...")
        confidence_scores = self.apply_pik_classification(hidden_states)
        
        if confidence_scores is None:
            logger.error("Failed to apply PIK classification")
            return None
        
        logger.info(f"PIK Classification Summary:")
        logger.info(f"  Average confidence: {np.mean(confidence_scores):.3f}")
        logger.info(f"  Confidence std: {np.std(confidence_scores):.3f}")
        logger.info(f"  Confidence range: [{np.min(confidence_scores):.3f}, {np.max(confidence_scores):.3f}]")
        
        # Prepare evaluation records
        evaluation_records = []
        matched_count = 0
        for i, sample_id in enumerate(sample_ids):
            # Get dataset property from mapping if available
            dataset_name = self.hash_id_to_dataset.get(sample_id, 'unknown')
            if dataset_name != 'unknown':
                matched_count += 1
            evaluation_records.append({
                'sample_id': sample_id,
                'ground_truth_correctness': int(labels[i]),
                'confidence_score': float(confidence_scores[i]),
                'dataset': dataset_name
            })
        
        if matched_count == 0 and len(self.hash_id_to_dataset) > 0:
            logger.warning(f"No hash_id matches found! Mapping has {len(self.hash_id_to_dataset)} entries but none matched.")
            # Log first few sample_ids and mapping keys for debugging
            logger.warning(f"First 5 sample_ids: {sample_ids[:5]}")
            logger.warning(f"First 5 mapping keys: {list(self.hash_id_to_dataset.keys())[:5]}")
        elif matched_count < len(sample_ids) and len(self.hash_id_to_dataset) > 0:
            logger.warning(f"Only {matched_count}/{len(sample_ids)} samples matched dataset mapping.")
        
        # Save evaluation records
        output_dir.mkdir(parents=True, exist_ok=True)
        labels_path = output_dir / "test_labels.json"
        with open(labels_path, 'w', encoding='utf-8') as f:
            json.dump(evaluation_records, f, indent=2, ensure_ascii=False)
        logger.info(f"Evaluation records saved to {labels_path}")
        
        # Calculate evaluation metrics
        # Note: For PIK, we don't have dataset/category groupings from npz files
        # So we'll just compute overall metrics
        from eval import calculate_all_metrics
        
        metrics = calculate_all_metrics(labels.astype(float), confidence_scores)
        
        # Create results structure
        results = {
            'overall': {
                'n_samples': len(labels),
                'n_total_samples': len(labels),
                **metrics
            },
            'metadata': {
                'model_name': self.model_name,
                'model_name_part': self.model_name.split("/")[-1],
                'test_dataset_name': self.test_dataset_name,
                'total_records': len(labels),
                'evaluation_timestamp': str(np.datetime64('now')),
                'pik_statistics': {
                    'avg_confidence': float(np.mean(confidence_scores)),
                    'confidence_std': float(np.std(confidence_scores)),
                    'min_confidence': float(np.min(confidence_scores)),
                    'max_confidence': float(np.max(confidence_scores))
                },
                'pik_model_info': {
                    'model_path': str(self.model_path),
                    'input_dim': self.model.input_dim,
                    'hidden_layers': list(self.model.hidden_layers),
                }
            }
        }
        
        # Save results
        results_path = output_dir / "test_results.json"
        save_evaluation_results(results, results_path)
        
        # Print summary
        self.print_results_summary(results)
        
        return results
    
    def print_results_summary(self, results: Dict[str, Any]) -> None:
        """Print a summary of evaluation results."""
        print(f"\nTEST PIK EVALUATION SUMMARY:")
        print("=" * 60)
        
        # Overall results
        if 'overall' in results:
            overall = results['overall']
            print(f"Overall Performance:")
            print(f"  Samples: {overall['n_samples']}")
            print(f"  Accuracy: {overall['accuracy']:.4f}")
            print(f"  Precision: {overall['precision']:.4f}")
            print(f"  Recall: {overall['recall']:.4f}")
            print(f"  F1-Score: {overall['f1']:.4f}")
            print(f"  Sensitivity: {overall['sensitivity']:.4f}")
            print(f"  Specificity: {overall['specificity']:.4f}")
            print(f"  ECE: {overall['ece']:.4f}")
            print(f"  Brier Score: {overall['brier']:.4f}")
            print(f"  AUROC: {overall['auroc']:.4f}")
            print(f"  AUCPR: {overall['aucpr']:.4f}")
        
        # PIK statistics
        if 'metadata' in results and 'pik_statistics' in results['metadata']:
            stats = results['metadata']['pik_statistics']
            print(f"\nPIK Classification Statistics:")
            print(f"  Average confidence: {stats['avg_confidence']:.3f}")
            print(f"  Confidence std: {stats['confidence_std']:.3f}")
            print(f"  Confidence range: [{stats['min_confidence']:.3f}, {stats['max_confidence']:.3f}]")
        
        
        print("=" * 60)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate PIK models for question understanding classification",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input configuration
    parser.add_argument("--data-dir", type=str, required=True,
                       help="Directory containing extracted representations (e.g., ../data/generation_extraction/)")
    parser.add_argument("--model-name", type=str, required=True,
                       help="Model name to evaluate (e.g., llava-hf/llava-v1.6-vicuna-13b-hf)")
    parser.add_argument("--test-dataset-name", type=str, required=True,
                       help="Test dataset name (e.g., test)")
    parser.add_argument("--train-dataset-name", type=str, required=True,
                       help="Train dataset name used for training (e.g., train)")
    parser.add_argument("--trained-model-path", type=str, required=True,
                       help="Base path to trained PIK models directory (e.g., ../trained_models/PIK)")
    parser.add_argument("--dataset-path", type=str, default=None,
                       help="Path to original dataset directory (for accessing dataset property, e.g., ../data/VLCB/raw)")
    
    # Output configuration
    parser.add_argument("--output-dir", type=str, required=True,
                       help="Base directory to save evaluation results (e.g., ../results/PIK)")
    parser.add_argument("--seed", type=int, default=None,
                       help="If set: treat --trained-model-path and --output-dir as literal leaves (no {MODEL}/{DATASET} injection).")

    return parser.parse_args()


def main():
    args = parse_arguments()
    
    # Extract model name part (last part after splitting by "/")
    model_name_part = args.model_name.split("/")[-1]
    
    # Convert base paths
    data_dir = Path(args.data_dir)
    trained_model_base = Path(args.trained_model_path)
    output_base = Path(args.output_dir)
    dataset_path = Path(args.dataset_path) if args.dataset_path else None
    
    # Construct full paths
    if args.seed is not None:
        trained_model_path = trained_model_base
        output_dir = output_base / args.test_dataset_name
    else:
        # Trained model path: {base}/{model_name_part}/{train_dataset_name}/best/
        trained_model_path = trained_model_base / model_name_part / args.train_dataset_name
        # Output directory: {base}/{model_name_part}/{test_dataset_name}/
        output_dir = output_base / model_name_part / args.test_dataset_name
    
    logger.info(f"PIK Evaluation Script")
    logger.info(f"Model name: {args.model_name} (using: {model_name_part})")
    logger.info(f"Train dataset: {args.train_dataset_name}")
    logger.info(f"Test dataset: {args.test_dataset_name}")
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Trained model path: {trained_model_path}")
    logger.info(f"Output directory: {output_dir}")
    if dataset_path:
        logger.info(f"Dataset path: {dataset_path}")
    else:
        logger.info("Dataset path not provided - dataset property will not be included in results")
    
    # Initialize evaluator
    evaluator = PIKEvaluator(
        model_path=trained_model_path,
        data_dir=data_dir,
        model_name=args.model_name,
        test_dataset_name=args.test_dataset_name,
        dataset_path=dataset_path
    )
    
    # Run evaluation
    results = evaluator.evaluate(output_dir)
    
    if results:
        print(f"\nEvaluation complete! Results saved to: {output_dir}")
        print("\nGenerated files:")
        print("- test_labels.json: Structured records with ground truth and confidence scores")
        print("- test_results.json: Comprehensive evaluation metrics")
    else:
        print("\nEvaluation failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()

# Example usage:
# python PIK_eval.py \
#   --data-dir ../data/generation_extraction/ \
#   --model-name Qwen/Qwen3-VL-32B-Instruct \
#   --trained-model-path ../trained_models/PIK \
#   --output-dir ../results/PIK \
#   --test-dataset-name test \
#   --train-dataset-name train