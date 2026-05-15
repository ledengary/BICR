#!/usr/bin/env python3
"""
BICR (Blind-Image Contrastive Ranking) — Training Script

Trains a shared MLP probe on h_base with L_rank regularization via h_blank.
Loss: L_bce + beta_brier * L_brier + lambda_rank * L_rank

Aligned with PIK / SAPLMA / II / ICC3 paper-locked conventions:
  - max_epochs   = 200
  - patience     = 20
  - validation_metric = composite (AUROC + (1 - ECE)) by default; auroc / ece /
    composite_with_constraints / auroc_with_constraint / validation_loss also supported.
  - TPESampler  + MedianPruner with per-epoch trial.report() & pruning.
  - shared_classifier_space (classifier_layers, classifier_dropout) read from
    utils/optuna_configs.json. BICR-specific params (lambda_rank, margin, beta_brier,
    learning_rate, weight_decay) are inline in _sample_hyperparameters().
  - Per-trial artifacts written to {output_dir}/models/trial_NNN/, best promoted
    to {output_dir}/best/. optuna_results.json saved at output_dir.

Data: Reads h_base, h_blank, mask_blank from extraction/BICR.

Usage:
    # Full BICR training
    python models/BICR/BICR_train.py --gpu 0 \\
        --model-name Qwen/Qwen3-VL-8B-Instruct --seed 23

    # Ablation variant
    python models/BICR/BICR_train.py --gpu 0 \\
        --model-name Qwen/Qwen3-VL-8B-Instruct --seed 23 --ablation no_rank
"""
import sys, os, json, gc, argparse, copy
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
import numpy as np
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='BICR Training (paper-locked conventions)')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index')
    parser.add_argument('--model-name', type=str, required=True,
                        help='Full model name (e.g., Qwen/Qwen3-VL-8B-Instruct)')
    parser.add_argument('--seed', type=int, required=True, help='Random seed')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='ICC extraction directory (default: data/extraction/BICR)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Checkpoint output directory '
                             '(default: trained_models/SPARROW/BICR[/_abl/{ablation}]/{model}/seed_{seed})')

    # Optuna
    parser.add_argument('--n-trials', type=int, default=50, help='Optuna trials')
    parser.add_argument('--n-jobs', type=int, default=1, help='Parallel Optuna jobs')

    # Training schedule (paper-locked defaults)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--max-epochs', type=int, default=200)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--val-eval-frequency', type=int, default=1,
                        help='Evaluate validation every N epochs')

    # Validation metric
    parser.add_argument('--validation-metric', type=str, default='composite',
                        choices=['auroc', 'ece', 'composite', 'auroc_with_constraint',
                                 'validation_loss', 'composite_with_constraints'],
                        help='Validation metric for optimization')
    parser.add_argument('--composite-auroc-weight', type=float, default=0.6,
                        help='Weight for AUROC in composite score')
    parser.add_argument('--min-sensitivity', type=float, default=0.60,
                        help='Minimum sensitivity for composite_with_constraints')
    parser.add_argument('--min-specificity', type=float, default=0.60,
                        help='Minimum specificity for composite_with_constraints')

    # Optuna config (only shared_classifier_space + budget_constraints are read; BICR
    # is not a key in optuna_configs.json — its params are sampled inline below.)
    parser.add_argument('--optuna-config', type=str, default=None,
                        help='Path to optuna_configs.json (default: utils/optuna_configs.json)')

    # Ablation
    parser.add_argument('--ablation', type=str, default=None,
                        choices=['no_rank', 'no_brier', 'bce_only'],
                        help='Ablation variant (default: full model)')
    return parser.parse_args()


args = parse_args()

# Early CUDA binding — match PIK/II convention so a single torch.cuda.set_device(0) works.
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
print(f"Set CUDA_VISIBLE_DEVICES={args.gpu}")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
optuna.logging.set_verbosity(optuna.logging.INFO)

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / 'utils'))
from eval import calculate_all_metrics, save_evaluation_results  # noqa: E402
from general import seed_everything  # noqa: E402

RANDOM_SEED = args.seed
seed_everything(RANDOM_SEED)


# ============================================================================
# Optuna config (shared classifier space + budget)
# ============================================================================

def load_optuna_config(config_path: Optional[str]) -> Dict[str, Any]:
    """Load shared_classifier_space + budget_constraints. BICR is sampled inline."""
    if config_path is None:
        config_path = ROOT / 'utils' / 'optuna_configs.json'
    with open(config_path, 'r') as f:
        cfg = json.load(f)
    return {
        'shared_classifier_space': cfg['shared_classifier_space'],
        'budget_constraints': cfg['budget_constraints'],
    }


OPTUNA_CONFIG = load_optuna_config(args.optuna_config)
SHARED_CLF_LAYERS = OPTUNA_CONFIG['shared_classifier_space']['classifier_layers']
SHARED_CLF_DROPOUT = OPTUNA_CONFIG['shared_classifier_space']['classifier_dropout']
MIN_PARAMS = OPTUNA_CONFIG['budget_constraints']['min_params']
MAX_PARAMS = OPTUNA_CONFIG['budget_constraints']['max_params']
logger.info(f"Parameter budget: [{MIN_PARAMS:,}, {MAX_PARAMS:,}]")
logger.info(f"Shared classifier layers: {SHARED_CLF_LAYERS}")
logger.info(f"Shared classifier dropout: {SHARED_CLF_DROPOUT}")


# ============================================================================
# Data loading
# ============================================================================

def load_split(data_dir: Path, model_short: str, split: str) -> Dict[str, Any]:
    """Load h_base, h_blank, mask_blank, labels, sample_ids for a split.

    Label convention is fixed: 1 = correct LVLM response, 0 = incorrect.
    Class imbalance is handled by BCE pos_weight = n_neg / n_pos at training time.
    """
    sdir = data_dir / model_short / split / 'samples'
    files = sorted(f for f in os.listdir(sdir) if f.endswith('.npz'))
    h_base_l, h_blank_l, mask_blank_l, labels, sample_ids = [], [], [], [], []
    for f in tqdm(files, desc=f'{model_short[:8]}/{split}', leave=False):
        d = dict(np.load(sdir / f, allow_pickle=True))
        ic = d.get('is_correct')
        if hasattr(ic, 'item'):
            ic = ic.item()
        if ic is None:
            continue
        h_base_l.append(d['h_base'].astype(np.float32))
        h_blank_l.append(d['h_blank'].astype(np.float32))
        mb = d.get('mask_blank', 1)
        if hasattr(mb, 'item'):
            mb = mb.item()
        mask_blank_l.append(float(mb))
        labels.append(int(bool(ic)))
        sid = d.get('sample_id', f[:-4])
        if hasattr(sid, 'item'):
            sid = str(sid.item())
        elif isinstance(sid, np.ndarray):
            sid = str(sid)
        sample_ids.append(sid)

    labels_arr = np.array(labels, dtype=np.float32)
    n_pos = int((labels_arr == 1).sum())
    n_neg = int((labels_arr == 0).sum())
    logger.info(f"  {split}: n={len(labels_arr)} (correct={n_pos}, incorrect={n_neg})")

    return {
        'h_base': np.stack(h_base_l),
        'h_blank': np.stack(h_blank_l),
        'mask_blank': np.array(mask_blank_l, dtype=np.float32),
        'labels': labels_arr,
        'sample_ids': sample_ids,
    }


# ============================================================================
# Model
# ============================================================================

class BICRModel(nn.Module):
    """MLP probe shared across base and blank views."""

    def __init__(self, h_dim: int, clf_layers_str: str, dropout: float = 0.0):
        super().__init__()
        self.h_dim = h_dim
        self.clf_layers_str = clf_layers_str
        self.dropout_rate = dropout

        if clf_layers_str == "0":
            self.hidden_layers: List[int] = []
            self.net = nn.Linear(h_dim, 1)
        else:
            self.hidden_layers = [int(x) for x in clf_layers_str.split(',') if x.strip()]
            dims = [h_dim] + self.hidden_layers
            blocks: List[nn.Module] = []
            for i in range(len(dims) - 1):
                blocks += [nn.Linear(dims[i], dims[i + 1]), nn.ReLU()]
                if dropout > 0:
                    blocks.append(nn.Dropout(dropout))
            blocks.append(nn.Linear(dims[-1], 1))
            self.net = nn.Sequential(*blocks)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h).squeeze(-1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def count_params_for_config(h_dim: int, clf_layers_str: str) -> int:
    """Count params without instantiating the model."""
    if clf_layers_str == "0":
        return h_dim + 1
    hidden = [int(x) for x in clf_layers_str.split(',') if x.strip()]
    dims = [h_dim] + hidden
    total = 0
    for i in range(len(dims) - 1):
        total += dims[i] * dims[i + 1] + dims[i + 1]
    total += dims[-1] + 1
    return total


# ============================================================================
# Validation metric helpers (PIK/ICC3 style)
# ============================================================================

def calculate_composite_score(auroc: float, ece: float, w: float = 0.6) -> float:
    return w * auroc + (1.0 - w) * (1.0 - ece)


def calculate_auroc_with_ece_constraint(auroc: float, ece: float) -> float:
    return auroc * (1.0 - ece)


def pick_tau_by_youden(y_true: np.ndarray, y_prob: np.ndarray) -> float:
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
    yhat = (y_prob >= tau).astype(int)
    tp = np.sum((yhat == 1) & (y_true == 1))
    tn = np.sum((yhat == 0) & (y_true == 0))
    fp = np.sum((yhat == 1) & (y_true == 0))
    fn = np.sum((yhat == 0) & (y_true == 1))
    sens = tp / (tp + fn + 1e-12)
    spec = tn / (tn + fp + 1e-12)
    return float(sens), float(spec)


def calculate_validation_score(metrics: Dict[str, float],
                                metric_name: str,
                                auroc_weight: float,
                                min_sens: float,
                                min_spec: float,
                                y_true: Optional[np.ndarray] = None,
                                y_prob: Optional[np.ndarray] = None) -> float:
    auroc = metrics.get('auroc', 0.0)
    ece = metrics.get('ece', 1.0)
    if metric_name == 'auroc':
        return auroc
    if metric_name == 'ece':
        return 1.0 - ece
    if metric_name == 'composite':
        return calculate_composite_score(auroc, ece, auroc_weight)
    if metric_name == 'auroc_with_constraint':
        return calculate_auroc_with_ece_constraint(auroc, ece)
    if metric_name == 'validation_loss':
        return -metrics.get('loss', 0.0)
    if metric_name == 'composite_with_constraints':
        if y_true is None or y_prob is None:
            return calculate_composite_score(auroc, ece, auroc_weight)
        tau = pick_tau_by_youden(y_true, y_prob)
        sens, spec = sens_spec_at_tau(y_true, y_prob, tau)
        feasible = (sens >= min_sens) and (spec >= min_spec)
        score = calculate_composite_score(auroc, ece, auroc_weight)
        return score if feasible else float('-inf')
    raise ValueError(f"Unknown validation metric: {metric_name}")


# ============================================================================
# Training (single trial)
# ============================================================================

def compute_pos_weight(labels: np.ndarray) -> float:
    n_pos = float((labels == 1).sum())
    n_neg = float((labels == 0).sum())
    return n_neg / n_pos if n_pos > 0 else 1.0


def evaluate_model(model: BICRModel, h_base_t: torch.Tensor, y_t: torch.Tensor,
                   batch_size: int) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, float]:
    """Evaluate on h_base only — single-view inference."""
    model.eval()
    n = h_base_t.shape[0]
    all_probs: List[torch.Tensor] = []
    all_logits: List[torch.Tensor] = []
    with torch.no_grad():
        for i in range(0, n, batch_size):
            sl = slice(i, i + batch_size)
            logit = model(h_base_t[sl])
            all_logits.append(logit.cpu())
            all_probs.append(torch.sigmoid(logit).cpu())
    probs = torch.cat(all_probs).numpy()
    logits = torch.cat(all_logits).numpy()
    labels = y_t.cpu().numpy()
    metrics = calculate_all_metrics(labels, probs)
    # Mean BCE on val (unweighted) for reporting
    yt_np = labels.astype(np.float32)
    z = logits.astype(np.float64)
    p = 1.0 / (1.0 + np.exp(-z))
    eps = 1e-7
    val_loss = float(-(yt_np * np.log(p + eps) + (1.0 - yt_np) * np.log(1.0 - p + eps)).mean())
    return metrics, labels, probs, val_loss


def train_one_trial(model: BICRModel,
                    data_tr: Dict[str, Any],
                    data_va: Dict[str, Any],
                    config: Dict[str, Any],
                    trial,
                    device: torch.device,
                    use_rank: bool,
                    use_brier: bool,
                    args_ref) -> Tuple[BICRModel, Dict[str, Any]]:
    model = model.to(device)
    opt = optim.Adam(model.parameters(), lr=config['learning_rate'],
                     weight_decay=config['weight_decay'])

    hb_tr = torch.FloatTensor(data_tr['h_base']).to(device)
    hk_tr = torch.FloatTensor(data_tr['h_blank']).to(device)
    mb_tr = torch.FloatTensor(data_tr['mask_blank']).to(device)
    yt = torch.FloatTensor(data_tr['labels']).to(device)
    hb_va = torch.FloatTensor(data_va['h_base']).to(device)
    yv = torch.FloatTensor(data_va['labels']).to(device)

    pos_w_val = compute_pos_weight(data_tr['labels'])
    pos_weight = torch.tensor([pos_w_val], device=device)

    n_train = hb_tr.shape[0]
    history: Dict[str, Any] = {
        'train_loss': [], 'val_loss': [], 'val_metrics': [],
        'best_epoch': None, 'best_operating_point': None,
    }
    best_score = float('-inf')
    best_state = None
    best_op = None
    patience_ctr = 0

    for epoch in range(args_ref.max_epochs):
        # ── Training (full-batch — matches the original BICR scheme) ─────────
        model.train()
        perm = torch.randperm(n_train, device=device)

        # Use mini-batches consistent with PIK/ICC3
        epoch_loss = 0.0
        epoch_samples = 0
        for i in range(0, n_train, args_ref.batch_size):
            idx = perm[i:i + args_ref.batch_size]
            hb = hb_tr[idx]
            hk = hk_tr[idx]
            mb = mb_tr[idx]
            y = yt[idx]
            B = hb.shape[0]

            opt.zero_grad()
            logit_base = model(hb)
            p_base = torch.sigmoid(logit_base)
            loss = F.binary_cross_entropy_with_logits(logit_base, y, pos_weight=pos_weight)

            if use_brier:
                loss = loss + config['beta_brier'] * torch.mean((p_base - y) ** 2)

            if use_rank:
                p_blank = torch.sigmoid(model(hk))
                rank_blank = F.relu(config['margin'] - (p_base - p_blank))
                rank_blank = (rank_blank * y * mb).sum() / ((y * mb).sum() + 1e-8)
                loss = loss + config['lambda_rank'] * rank_blank

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            epoch_loss += loss.item() * B
            epoch_samples += B

        avg_train = epoch_loss / max(epoch_samples, 1)

        # ── Validation ───────────────────────────────────────────────────────
        if epoch % args_ref.val_eval_frequency == 0:
            val_metrics, all_y, all_p, val_loss = evaluate_model(
                model, hb_va, yv, args_ref.batch_size)
            val_metrics['loss'] = val_loss

            tau = pick_tau_by_youden(all_y, all_p)
            sens, spec = sens_spec_at_tau(all_y, all_p, tau)
            operating_point = {'tau': tau, 'sens': sens, 'spec': spec}

            if args_ref.validation_metric == 'composite_with_constraints' and epoch >= 2:
                feasible = (sens >= args_ref.min_sensitivity) and (spec >= args_ref.min_specificity)
                if not feasible:
                    logger.info(f"  Trial {trial.number}: Epoch {epoch}: Pruned (infeasible)")
                    raise optuna.TrialPruned()

            val_score = calculate_validation_score(
                val_metrics, args_ref.validation_metric,
                args_ref.composite_auroc_weight,
                args_ref.min_sensitivity, args_ref.min_specificity,
                all_y, all_p,
            )

            history['train_loss'].append(avg_train)
            history['val_loss'].append(val_loss)
            history['val_metrics'].append(val_metrics)

            trial.report(val_score, epoch)
            if trial.should_prune():
                logger.info(f"  Trial {trial.number}: Epoch {epoch}: Pruned by Optuna")
                raise optuna.TrialPruned()

            logger.info(f"  Epoch {epoch:3d} | TrLoss={avg_train:.4f} VaLoss={val_loss:.4f} "
                        f"AUROC={val_metrics['auroc']:.4f} ECE={val_metrics['ece']:.4f} "
                        f"Brier={val_metrics['brier']:.4f} Score={val_score:.4f}")

            if val_score > best_score:
                best_score = val_score
                best_state = copy.deepcopy(model.state_dict())
                best_op = operating_point
                patience_ctr = 0
                history['best_epoch'] = epoch
                history['best_operating_point'] = operating_point
                logger.info(f"    *** new best: score={val_score:.4f} ***")
            else:
                patience_ctr += 1
        else:
            history['train_loss'].append(avg_train)
            history['val_loss'].append(None)
            history['val_metrics'].append(None)

        if patience_ctr >= args_ref.patience:
            logger.info(f"  Early stopping at epoch {epoch} (patience={args_ref.patience})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    history['best_score'] = best_score
    history['best_operating_point'] = best_op
    return model, history


# ============================================================================
# Hyperparameter sampling (BICR-specific, inline)
# ============================================================================

def sample_hyperparameters(trial, use_rank: bool, use_brier: bool) -> Dict[str, Any]:
    config: Dict[str, Any] = {}
    config['classifier_layers'] = trial.suggest_categorical(
        'classifier_layers', tuple(SHARED_CLF_LAYERS))
    config['dropout'] = trial.suggest_categorical(
        'dropout', tuple(SHARED_CLF_DROPOUT))
    config['learning_rate'] = trial.suggest_float(
        'learning_rate', 1e-5, 1e-3, log=True)
    config['weight_decay'] = trial.suggest_float(
        'weight_decay', 1e-6, 1e-3, log=True)
    # BICR loss-term coefficients
    config['beta_brier'] = (
        trial.suggest_float('beta_brier', 0.0, 0.5) if use_brier else 0.0
    )
    config['lambda_rank'] = (
        trial.suggest_float('lambda_rank', 0.01, 0.3) if use_rank else 0.0
    )
    config['margin'] = (
        trial.suggest_float('margin', 0.05, 0.25) if use_rank else 0.1
    )
    return config


# ============================================================================
# Save helpers
# ============================================================================

def save_trial(model: BICRModel, config: Dict[str, Any], val_metrics: Dict[str, float],
               val_score: float, trial_num: int, n_params: int, best_epoch: Optional[int],
               operating_point: Optional[Dict[str, float]], trial_dir: Path,
               args_ref) -> None:
    trial_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), trial_dir / 'model.pth')
    info = {
        'trial_number': trial_num,
        'method': 'BICR',
        'ablation': args_ref.ablation,
        'model_name': args_ref.model_name,
        'seed': args_ref.seed,
        'input_dim': model.h_dim,
        'classifier_layers': model.clf_layers_str,
        'hidden_layers': list(model.hidden_layers),
        'dropout': model.dropout_rate,
        'config': config,
        'metrics': val_metrics,
        'val_score': val_score,
        'best_epoch': best_epoch,
        'best_operating_point': operating_point,
        'n_parameters': n_params,
        'parameter_info': {
            'n_params': n_params,
            'budget_constraints': {'min_params': MIN_PARAMS, 'max_params': MAX_PARAMS},
        },
        'validation_metric': args_ref.validation_metric,
        'composite_auroc_weight': args_ref.composite_auroc_weight,
    }
    with open(trial_dir / 'model_info.json', 'w') as f:
        json.dump(info, f, indent=2)
    with open(trial_dir / 'results.json', 'w') as f:
        json.dump({'trial_number': trial_num, 'config': config,
                   'val_metrics': val_metrics, 'val_score': val_score}, f, indent=2)


def promote_best(models_dir: Path, output_dir: Path) -> Path:
    import shutil
    best_dir = output_dir / 'best'
    best_score = float('-inf')
    best_src: Optional[Path] = None
    for td in sorted(models_dir.glob('trial_*')):
        rpath = td / 'results.json'
        if not rpath.exists():
            continue
        try:
            with open(rpath) as f:
                r = json.load(f)
            if r.get('val_score', float('-inf')) > best_score:
                best_score = r['val_score']
                best_src = td
        except Exception:
            pass
    if best_src is None:
        raise RuntimeError('No valid trials found — cannot promote best model.')
    if best_dir.exists():
        shutil.rmtree(best_dir)
    best_dir.mkdir(parents=True, exist_ok=True)
    for f in best_src.glob('*'):
        if f.is_file():
            shutil.copy2(f, best_dir / f.name)
    logger.info(f"Best trial (score={best_score:.4f}) promoted -> {best_dir}")
    return best_dir


# ============================================================================
# Main
# ============================================================================

def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model_short = args.model_name.split('/')[-1]
    data_dir = Path(args.data_dir) if args.data_dir else ROOT / 'data' / 'extraction' / 'BICR'

    use_rank = args.ablation not in ('no_rank', 'bce_only')
    use_brier = args.ablation not in ('no_brier', 'bce_only')

    # Output paths
    if args.output_dir:
        output_dir = Path(args.output_dir)
    elif args.ablation:
        output_dir = ROOT / 'trained_models' / 'SPARROW' / 'BICR_abl' / args.ablation / model_short / f'seed_{args.seed}'
    else:
        output_dir = ROOT / 'trained_models' / 'SPARROW' / 'BICR' / model_short / f'seed_{args.seed}'

    if (output_dir / 'best' / 'model.pth').exists():
        logger.info(f"SKIP: checkpoint exists at {output_dir / 'best'}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir = output_dir / 'models'
    models_dir.mkdir(parents=True, exist_ok=True)

    variant = args.ablation or 'full'
    logger.info(f"BICR Training — {model_short}, seed {args.seed}, variant={variant}")
    logger.info(f"  use_rank={use_rank}, use_brier={use_brier}")
    logger.info(f"  max_epochs={args.max_epochs}, patience={args.patience}, "
                f"val_eval_frequency={args.val_eval_frequency}")
    logger.info(f"  validation_metric={args.validation_metric} "
                f"(auroc_weight={args.composite_auroc_weight})")
    logger.info(f"  device={device}")

    # Load data — labels are fixed (1=correct, 0=incorrect)
    tr = load_split(data_dir, model_short, 'train')
    va = load_split(data_dir, model_short, 'validation')
    h_dim = tr['h_base'].shape[1]
    n_pos = int((tr['labels'] == 1).sum())
    n_neg = int((tr['labels'] == 0).sum())
    logger.info(f"  Train: n={len(tr['labels'])} (pos={n_pos}, neg={n_neg}), "
                f"val: n={len(va['labels'])}, h_dim={h_dim}")

    # Optuna search
    def objective(trial):
        try:
            config = sample_hyperparameters(trial, use_rank, use_brier)
            logger.info(f"\n  Trial {trial.number}: {config}")

            # Param budget check (cheap, before instantiation)
            n_params = count_params_for_config(h_dim, config['classifier_layers'])
            if n_params < MIN_PARAMS or n_params > MAX_PARAMS:
                logger.info(f"  Trial {trial.number}: {n_params:,} params outside budget "
                            f"[{MIN_PARAMS:,}, {MAX_PARAMS:,}] — skip")
                return float('-inf')

            model = BICRModel(h_dim, config['classifier_layers'], config['dropout'])
            n_params = model.count_parameters()
            logger.info(f"  Trial {trial.number}: {n_params:,} parameters")

            trained_model, history = train_one_trial(
                model, tr, va, config, trial, device, use_rank, use_brier, args)

            hb_va = torch.FloatTensor(va['h_base']).to(device)
            yv = torch.FloatTensor(va['labels']).to(device)
            val_metrics, all_y, all_p, val_loss = evaluate_model(
                trained_model, hb_va, yv, args.batch_size)
            val_metrics['loss'] = val_loss
            val_score = calculate_validation_score(
                val_metrics, args.validation_metric,
                args.composite_auroc_weight,
                args.min_sensitivity, args.min_specificity,
                all_y, all_p)

            logger.info(f"  Trial {trial.number} -> AUROC={val_metrics['auroc']:.4f} "
                        f"ECE={val_metrics['ece']:.4f} Brier={val_metrics['brier']:.4f} "
                        f"Score={val_score:.4f} params={n_params:,}")

            trial_dir = models_dir / f'trial_{trial.number:03d}'
            save_trial(trained_model, config, val_metrics, val_score, trial.number,
                       n_params, history.get('best_epoch'),
                       history.get('best_operating_point'), trial_dir, args)

            del trained_model, hb_va, yv
            return val_score

        except optuna.TrialPruned:
            raise
        except Exception as e:
            import traceback
            logger.error(f"  Trial {trial.number} error: {e}\n{traceback.format_exc()}")
            return float('-inf')
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    sampler = TPESampler(seed=RANDOM_SEED)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10, interval_steps=5)
    study = optuna.create_study(direction='maximize', sampler=sampler, pruner=pruner)

    logger.info(f"\nStarting Optuna ({args.n_trials} trials)…")
    study.optimize(objective, n_trials=args.n_trials, n_jobs=args.n_jobs,
                   show_progress_bar=True)

    # Promote best
    best_dir = promote_best(models_dir, output_dir)

    with open(output_dir / 'optuna_results.json', 'w') as f:
        json.dump({
            'best_params': study.best_params,
            'best_value': study.best_value,
            'n_trials': len(study.trials),
            'validation_metric': args.validation_metric,
            'composite_auroc_weight': args.composite_auroc_weight,
        }, f, indent=2)

    with open(best_dir / 'model_info.json') as f:
        best_info = json.load(f)

    logger.info("\n" + "=" * 60)
    logger.info("BICR training complete.")
    logger.info(f"  AUROC : {best_info['metrics']['auroc']:.4f}")
    logger.info(f"  ECE   : {best_info['metrics']['ece']:.4f}")
    logger.info(f"  Brier : {best_info['metrics']['brier']:.4f}")
    logger.info(f"  Score : {best_info.get('val_score', float('nan')):.4f}")
    logger.info(f"  Epoch : {best_info.get('best_epoch')}")
    logger.info(f"  Params: {best_info['n_parameters']:,}")
    logger.info(f"  Output: {best_dir}")
    logger.info("=" * 60)

    del tr, va
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()


# Qwen/Qwen3-VL-8B-Instruct
# llava-hf/llava-v1.6-vicuna-13b-hf
# OpenGVLab/InternVL3_5-14B-HF
# google/gemma-3-27b-it
# deepseek-ai/deepseek-vl2

## Example Usage:
# python models/BICR/BICR_train.py \
#   --gpu 0 \
#   --model-name Qwen/Qwen3-VL-8B-Instruct \
#   --seed 23 \
#   --n-trials 50 \
#   --max-epochs 200 \
#   --patience 20 \
#   --validation-metric composite \
#   --composite-auroc-weight 0.6
