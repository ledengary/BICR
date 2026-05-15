#!/usr/bin/env python3

"""
SAPLMA_paper_train.py

SAPLMA paper architecture (Linear 256 -> 128 -> 64 -> 1 with ReLU, no dropout)
trained under VLCB's paper-locked schedule, aligned with PIK / II / BICR / CCPS:

  - Optimizer       : Adam (no weight decay, paper)
  - Loss            : BCEWithLogitsLoss(pos_weight=n_neg/n_pos)  (class-balanced)
  - Max epochs      : 200
  - Patience        : 20      (early stop on validation composite score)
  - Validation metric: composite (auroc_weight=0.6 default) ; auroc / ece /
                       composite_with_constraints / auroc_with_constraint /
                       validation_loss also supported.
  - Label convention: 1 = correct, 0 = incorrect.
  - Output layout   : {output-dir}/best/model.pth + best/model_info.json
                       (single best checkpoint, like PIK / II / BICR).

This departs from the original SAPLMA recipe (5 epochs / 3 averaged runs / no
class weighting / no early stopping) so SAPLMA is comparable head-to-head with
the other trained methods in the benchmark.
"""

import argparse
import copy
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Train SAPLMA (paper architecture) under the paper-locked SPARROW schedule",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--train-dataset-name", type=str, required=True)
    parser.add_argument("--val-dataset-name", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="../trained_models/SAPLMA_paper")

    parser.add_argument("--cuda-devices", type=str, default="0")
    parser.add_argument("--batch-size", type=int, default=32)

    # Paper-locked schedule
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--val-eval-frequency", type=int, default=1)

    # Validation metric
    parser.add_argument("--validation-metric", type=str, default="composite",
                        choices=['auroc', 'ece', 'composite', 'auroc_with_constraint',
                                 'validation_loss', 'composite_with_constraints'])
    parser.add_argument("--composite-auroc-weight", type=float, default=0.6)
    parser.add_argument("--min-sensitivity", type=float, default=0.60)
    parser.add_argument("--min-specificity", type=float, default=0.60)

    # Optimizer
    parser.add_argument("--learning-rate", type=float, default=1e-3,
                        help="Adam default; paper uses Adam without explicit lr (defaults to 1e-3).")
    parser.add_argument("--weight-decay", type=float, default=0.0,
                        help="0.0 to match paper; the schedule itself adds class weighting + early stop.")

    parser.add_argument("--seed", type=int, default=None,
                        help="If set: single-run mode; --output-dir is the literal leaf "
                             "(no {MODEL}/{DATASET} injection).")
    return parser.parse_args()


args = parse_arguments()

os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices
print(f"Set CUDA_VISIBLE_DEVICES={args.cuda_devices}")

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.optim as optim  # noqa: E402

sys.path.append(str(Path(__file__).parent / "../utils"))
from general import seed_everything  # noqa: E402
from eval import calculate_all_metrics  # noqa: E402


# ------------------------------------------------------------------
# Paper architecture
# ------------------------------------------------------------------

ARCHITECTURE = (256, 128, 64)


class SAPLMADataset:
    def __init__(self, hidden_states: np.ndarray, labels: np.ndarray):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hidden_states = torch.FloatTensor(hidden_states).to(self.device, non_blocking=True)
        self.labels = torch.FloatTensor(labels.astype(float)).to(self.device, non_blocking=True)
        logger.info(f"[SAPLMA] Loaded {len(self.hidden_states)} samples to {self.device}")
        logger.info(f"[SAPLMA] Hidden states shape: {self.hidden_states.shape}")

    def __len__(self):
        return len(self.hidden_states)


class SAPLMAModel(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_layers = ARCHITECTURE
        layers: List[nn.Module] = []
        dims = [input_dim] + list(ARCHITECTURE)
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dims[-1], 1))
        self.classifier = nn.Sequential(*layers)

    def forward(self, x):
        return self.classifier(x).squeeze(-1)


# ------------------------------------------------------------------
# Validation helpers (PIK/II/BICR/ICC3 style)
# ------------------------------------------------------------------

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


def calculate_validation_score(metrics: Dict[str, float], metric_name: str,
                                auroc_weight: float, min_sens: float, min_spec: float,
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


# ------------------------------------------------------------------
# Trainer
# ------------------------------------------------------------------

class SAPLMATrainer:
    def __init__(self):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")

    def load_data_from_samples(
        self, samples_dir: Path,
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Load LAST token hidden state + label from .npz samples.

        Label convention is fixed: 1 = correct, 0 = incorrect. Class imbalance
        is handled by BCE pos_weight = n_neg / n_pos.
        """
        hidden_states_list: List[np.ndarray] = []
        labels_list: List[bool] = []
        ids_list: List[str] = []
        npz_files = list(samples_dir.glob("*.npz"))
        logger.info(f"Loading {len(npz_files)} samples from {samples_dir}")
        skipped_no_label = 0
        for npz_file in tqdm(npz_files):
            data = np.load(npz_file, allow_pickle=True)
            ic = data["is_correct"]
            if ic is None or (isinstance(ic, np.ndarray) and ic.item() is None):
                skipped_no_label += 1
                continue
            hidden_states_list.append(data["hidden_states"][-1])
            labels_list.append(bool(ic))
            hid = data["hash_id"] if "hash_id" in data.files else npz_file.stem
            if isinstance(hid, np.ndarray):
                hid = hid.item() if hid.shape == () else hid
            ids_list.append(str(hid))
        if skipped_no_label:
            logger.info(f"Skipped {skipped_no_label} samples without correctness label")
        if not hidden_states_list:
            raise ValueError(f"No valid data in {samples_dir}")
        hs = np.stack(hidden_states_list)
        ys = np.array(labels_list, dtype=bool)

        n_pos = int(ys.sum()); n_neg = int(len(ys) - n_pos)
        logger.info(f"Loaded {len(hs)} samples | hidden_dim={hs.shape[1]} | "
                    f"correct={n_pos}  incorrect={n_neg}")
        return hs, ys, ids_list

    @staticmethod
    def calculate_pos_weight_ratio(labels: np.ndarray) -> float:
        n1 = float((labels == 1).sum())
        n0 = float((labels == 0).sum())
        ratio = n0 / n1 if n1 > 0 else 1.0
        logger.info(f"pos_weight = n_neg/n_pos = {n0:.0f}/{n1:.0f} = {ratio:.3f}")
        return ratio

    def evaluate(self, model: SAPLMAModel, dataset: SAPLMADataset, batch_size: int
                 ) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, float]:
        model.eval()
        n = len(dataset)
        all_logits: List[torch.Tensor] = []
        with torch.no_grad():
            for i in range(0, n, batch_size):
                sl = slice(i, i + batch_size)
                logits = model(dataset.hidden_states[sl])
                all_logits.append(logits.cpu())
        logits_np = torch.cat(all_logits).numpy()
        labels_np = dataset.labels.cpu().numpy()
        probs = 1.0 / (1.0 + np.exp(-logits_np.astype(np.float64)))
        eps = 1e-7
        val_loss = float(-(labels_np * np.log(probs + eps) +
                           (1.0 - labels_np) * np.log(1.0 - probs + eps)).mean())
        metrics = calculate_all_metrics(labels_np, probs.astype(np.float32))
        return metrics, labels_np, probs.astype(np.float32), val_loss

    def train(self, model: SAPLMAModel, train_ds: SAPLMADataset, val_ds: SAPLMADataset,
              pos_weight_ratio: float, args_ref) -> Tuple[SAPLMAModel, Dict[str, Any]]:
        device = torch.device(self.device)
        model = model.to(device)
        pos_weight = torch.tensor([pos_weight_ratio], device=device, dtype=torch.float)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = optim.Adam(model.parameters(), lr=args_ref.learning_rate,
                               weight_decay=args_ref.weight_decay)

        n_train = len(train_ds)
        history: Dict[str, Any] = {
            'train_loss': [], 'val_loss': [], 'val_metrics': [],
            'best_epoch': None, 'best_operating_point': None,
        }
        best_score = float('-inf')
        best_state = None
        best_op = None
        patience_ctr = 0

        for epoch in range(args_ref.max_epochs):
            model.train()
            perm = torch.randperm(n_train, device=device)
            epoch_loss = 0.0
            n_seen = 0
            for i in range(0, n_train, args_ref.batch_size):
                idx = perm[i:i + args_ref.batch_size]
                x = train_ds.hidden_states[idx]
                y = train_ds.labels[idx]
                B = x.shape[0]
                optimizer.zero_grad()
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item() * B
                n_seen += B
            avg_train = epoch_loss / max(n_seen, 1)

            if epoch % args_ref.val_eval_frequency != 0:
                history['train_loss'].append(avg_train)
                history['val_loss'].append(None)
                history['val_metrics'].append(None)
                continue

            metrics, all_y, all_p, val_loss = self.evaluate(model, val_ds, args_ref.batch_size)
            metrics['loss'] = val_loss

            tau = pick_tau_by_youden(all_y, all_p)
            sens, spec = sens_spec_at_tau(all_y, all_p, tau)
            operating_point = {'tau': tau, 'sens': sens, 'spec': spec}

            if args_ref.validation_metric == 'composite_with_constraints' and epoch >= 2:
                feasible = (sens >= args_ref.min_sensitivity) and (spec >= args_ref.min_specificity)
                if not feasible:
                    logger.info(f"Epoch {epoch}: Pruned (infeasible sens/spec)")
                    # Continue training but treat this as non-improving
                    score = float('-inf')
                else:
                    score = calculate_validation_score(metrics, args_ref.validation_metric,
                                                       args_ref.composite_auroc_weight,
                                                       args_ref.min_sensitivity,
                                                       args_ref.min_specificity,
                                                       all_y, all_p)
            else:
                score = calculate_validation_score(metrics, args_ref.validation_metric,
                                                   args_ref.composite_auroc_weight,
                                                   args_ref.min_sensitivity,
                                                   args_ref.min_specificity,
                                                   all_y, all_p)

            history['train_loss'].append(avg_train)
            history['val_loss'].append(val_loss)
            history['val_metrics'].append(metrics)

            logger.info(f"Epoch {epoch:3d} | TrLoss={avg_train:.4f} VaLoss={val_loss:.4f} "
                        f"AUROC={metrics['auroc']:.4f} ECE={metrics['ece']:.4f} "
                        f"Brier={metrics['brier']:.4f} Score={score:.4f}")

            if score > best_score:
                best_score = score
                best_state = copy.deepcopy(model.state_dict())
                best_op = operating_point
                patience_ctr = 0
                history['best_epoch'] = epoch
                history['best_operating_point'] = operating_point
                logger.info(f"  *** new best: score={score:.4f} ***")
            else:
                patience_ctr += 1

            if patience_ctr >= args_ref.patience:
                logger.info(f"Early stopping at epoch {epoch} (patience={args_ref.patience})")
                break

        if best_state is not None:
            model.load_state_dict(best_state)
        history['best_score'] = best_score
        history['best_operating_point'] = best_op
        return model, history


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    seed = args.seed if args.seed is not None else 23
    seed_everything(seed)
    logger.info(f"SAPLMA paper-arch under SPARROW schedule (seed={seed}, "
                f"single_run={args.seed is not None})")
    logger.info(f"  max_epochs={args.max_epochs}  patience={args.patience}  "
                f"validation_metric={args.validation_metric} "
                f"(auroc_w={args.composite_auroc_weight})")

    model_name_part = args.model_name.split("/")[-1]
    data_dir = Path(args.data_dir)
    train_samples_dir = data_dir / model_name_part / args.train_dataset_name / "samples"
    val_samples_dir = data_dir / model_name_part / args.val_dataset_name / "samples"

    if args.seed is not None:
        # SPARROW single-seed mode: --output-dir is the literal leaf
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(args.output_dir) / model_name_part / args.train_dataset_name
    best_dir = output_dir / "best"
    best_dir.mkdir(parents=True, exist_ok=True)

    trainer = SAPLMATrainer()
    train_hs, train_labels, _ = trainer.load_data_from_samples(train_samples_dir)
    val_hs, val_labels, _ = trainer.load_data_from_samples(val_samples_dir)

    pos_weight_ratio = trainer.calculate_pos_weight_ratio(train_labels)
    train_ds = SAPLMADataset(train_hs, train_labels)
    val_ds = SAPLMADataset(val_hs, val_labels)

    input_dim = train_hs.shape[1]
    model = SAPLMAModel(input_dim)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model: {ARCHITECTURE}  params={n_params:,}")

    model, history = trainer.train(model, train_ds, val_ds, pos_weight_ratio, args)

    # Final val metrics on best checkpoint
    val_metrics, all_y, all_p, val_loss = trainer.evaluate(model, val_ds, args.batch_size)
    val_metrics['loss'] = val_loss
    logger.info("\n===== Final validation metrics (best checkpoint) =====")
    for k, v in val_metrics.items():
        logger.info(f"  {k}: {v:.4f}")

    # Save
    torch.save(model.state_dict(), best_dir / "model.pth")
    info = {
        "method": "SAPLMA_paper",
        "architecture": list(ARCHITECTURE),
        "input_dim": input_dim,
        "hidden_layers": list(ARCHITECTURE),
        "classifier_layers": ",".join(str(h) for h in ARCHITECTURE),
        "dropout": 0.0,
        "n_parameters": n_params,
        "config": {
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "batch_size": args.batch_size,
            "max_epochs": args.max_epochs,
            "patience": args.patience,
            "validation_metric": args.validation_metric,
            "composite_auroc_weight": args.composite_auroc_weight,
            "pos_weight_ratio": pos_weight_ratio,
            "optimizer": "Adam",
            "loss": "BCEWithLogitsLoss(pos_weight)",
        },
        "metrics": val_metrics,
        "best_epoch": history.get('best_epoch'),
        "best_operating_point": history.get('best_operating_point'),
        "seed": seed,
    }
    with open(best_dir / "model_info.json", "w") as f:
        json.dump(info, f, indent=2)
    logger.info(f"Saved best checkpoint to {best_dir}")


if __name__ == "__main__":
    main()


# Qwen/Qwen3-VL-8B-Instruct
# llava-hf/llava-v1.6-vicuna-13b-hf
# OpenGVLab/InternVL3_5-14B-HF
# google/gemma-3-27b-it
# deepseek-ai/deepseek-vl2

## Example Usage:
# python SAPLMA_paper_train.py \
#   --cuda-devices 0 \
#   --data-dir ../data/extraction/raw/ \
#   --model-name deepseek-ai/deepseek-vl2 \
#   --train-dataset-name train \
#   --val-dataset-name validation \
#   --output-dir ../trained_models/SAPLMA_paper/deepseek-vl2/seed_23 \
#   --batch-size 32 \
#   --max-epochs 200 --patience 20 \
#   --validation-metric composite --composite-auroc-weight 0.6 \
#   --seed 23
