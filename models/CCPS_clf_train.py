#!/usr/bin/env python3
"""
CCPS_clf_paper_train.py
=======================
CCPS Stage 2 (paper architecture, SPARROW-aligned schedule).

Loads the Stage 1 encoder produced by CCPS_proj_paper_train.py, attaches the
paper-exact classifier head, and JOINTLY fine-tunes encoder + classifier.

Classifier (COE, Appendix E):
    Linear(16 -> 32) + ReLU
    Linear(32 -> 2)

Training (SPARROW-aligned, replaces the paper's 5000-step / no-early-stop loop):
    Encoder NOT frozen (joint fine-tuning per paper).
    Adam, LR=1e-4, weight_decay=0.0, batch_size=32.
    max_epochs=200, patience=20, early stopping on composite (AUROC + ECE).
    CrossEntropyLoss(weight=[1.0, n_neg/n_pos]) — 2-class equivalent of pos_weight.
    Label convention: 1 = correct, 0 = incorrect.

Output:
    {output-dir}/best/
        classifier_model.pt   (full state dict incl. embedding_model.*)
        model_info.json
        scaler.pkl            (copied from Stage 1)

Run AFTER CCPS_proj_paper_train.py.
"""

import argparse
import os
import sys
from pathlib import Path


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="CCPS Stage 2 (paper-fixed): classifier joint fine-tuning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--feature-dir", type=str, required=True)
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--train-dataset-name", type=str, required=True)
    parser.add_argument("--val-dataset-name", type=str, required=True)
    # EITHER give --encoder-dir pointing directly to the Stage 1 best/ folder,
    # OR give --encoder-output-dir and we derive the path.
    parser.add_argument("--encoder-output-dir", type=str,
                        default="../trained_models/CCPS_proj_paper")
    parser.add_argument("--encoder-dir", type=str, default=None,
                        help="Stage 1 best/ folder (overrides --encoder-output-dir)")
    parser.add_argument("--output-dir", type=str, default="../trained_models/CCPS_clf_paper")
    parser.add_argument("--cuda-devices", type=str, default="0")
    parser.add_argument("--seed", type=int, default=23)

    # SPARROW-aligned schedule (defaults match constants below)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--val-eval-frequency", type=int, default=1)
    parser.add_argument("--validation-metric", type=str, default="composite",
                        choices=['auroc', 'ece', 'composite', 'auroc_with_constraint',
                                 'validation_loss', 'composite_with_constraints'])
    parser.add_argument("--composite-auroc-weight", type=float, default=0.6)
    parser.add_argument("--min-sensitivity", type=float, default=0.60)
    parser.add_argument("--min-specificity", type=float, default=0.60)
    return parser.parse_args()


args = parse_arguments()
os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices
print(f"Set CUDA_VISIBLE_DEVICES={args.cuda_devices}")

import copy
import json
import pickle
import shutil
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent / "../utils"))
from general import seed_everything
from eval import calculate_all_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Architecture constants (paper Appendix E, OE variant)
# ---------------------------------------------------------------------------
HIDDEN_DIMS       = [64, 32]           # encoder (for reconstruction)
KERNEL_SIZES      = [3, 3]
EMBED_DIM         = 16
CLF_HIDDEN_DIMS   = [32]                # classifier: 16 -> 32 -> 2
FREEZE_ENCODER    = False               # JOINT fine-tune per paper
DROPOUT           = 0.0

# SPARROW-aligned schedule (replaces paper's 5000-step / no-early-stop / AdamW / no-class-weight setup).
LEARNING_RATE     = 1e-4
WEIGHT_DECAY      = 0.0                 # Adam (no weight decay), matching PIK / SAPLMA / II / BICR
BATCH_SIZE        = 32
MAX_EPOCHS        = 200
PATIENCE          = 20
VAL_EVAL_FREQUENCY = 1

EXCLUDE_COLUMNS = {"hash_id", "is_correct", "token_idx", "token_id", "token_str",
                   "sample_id", "token_text", "dataset", "split"}


# ---------------------------------------------------------------------------
# Models (must match Stage 1's architecture)
# ---------------------------------------------------------------------------
class ConvEmbeddingNet(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.activation = nn.ReLU()
        self.conv_layers = nn.ModuleList()
        self.conv_layers.append(
            nn.Conv1d(input_dim, HIDDEN_DIMS[0],
                      kernel_size=KERNEL_SIZES[0], padding=KERNEL_SIZES[0] // 2)
        )
        for i in range(1, len(HIDDEN_DIMS)):
            self.conv_layers.append(
                nn.Conv1d(HIDDEN_DIMS[i - 1], HIDDEN_DIMS[i],
                          kernel_size=KERNEL_SIZES[i], padding=KERNEL_SIZES[i] // 2)
            )
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(HIDDEN_DIMS[-1], EMBED_DIM)

    def forward(self, x, seq_lengths=None):
        x = x.transpose(1, 2)
        for conv in self.conv_layers:
            x = self.activation(conv(x))
        x = self.pool(x).squeeze(-1)
        x = self.fc(x)
        return x


class ConvClassifierWithEmbedding(nn.Module):
    def __init__(self, embedding_model: ConvEmbeddingNet):
        super().__init__()
        self.embedding_model = embedding_model
        layers = [
            nn.Linear(EMBED_DIM, CLF_HIDDEN_DIMS[0]),
            nn.ReLU(),
            nn.Linear(CLF_HIDDEN_DIMS[0], 2),
        ]
        self.classifier = nn.Sequential(*layers)

    def forward(self, x, seq_lengths=None):
        emb = self.embedding_model(x, seq_lengths)
        return self.classifier(emb), emb


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class SequenceClassificationDataset(Dataset):
    def __init__(self, samples: List[Dict], max_seq_length: int):
        self.samples = samples
        self.max_seq_length = max_seq_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        feats = s["features_scaled"]
        seq_len = min(s["seq_length"], self.max_seq_length)
        d = feats.shape[1]
        padded = np.zeros((self.max_seq_length, d))
        padded[:seq_len] = feats[:seq_len]
        return (torch.tensor(padded, dtype=torch.float32),
                torch.tensor(s["label"], dtype=torch.long),
                torch.tensor(seq_len, dtype=torch.long))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_features(feature_dir: Path, model_name: str, dataset_name: str) -> pd.DataFrame:
    model_name_part = model_name.split("/")[-1]
    p = feature_dir / model_name_part / dataset_name / "features.pkl"
    if not p.exists():
        raise FileNotFoundError(f"Features not found: {p}")
    df = pd.read_pickle(p)
    logger.info(f"Loaded {len(df)} tokens, {df['hash_id'].nunique()} samples from {p}")
    return df


def prepare_sequential_data(df: pd.DataFrame, max_seq_length: int,
                            eps_high: float = 20.0) -> List[Dict]:
    feat_cols = [c for c in df.columns if c not in EXCLUDE_COLUMNS]
    out = []
    for hash_id in tqdm(df["hash_id"].unique(), desc="Preparing sequences"):
        sdf = df[df["hash_id"] == hash_id].sort_values("token_idx")
        feats = sdf[feat_cols].copy()
        if "epsilon_to_flip_token" in feats.columns:
            feats["epsilon_to_flip_token"] = feats["epsilon_to_flip_token"].replace(
                [np.inf, -np.inf], eps_high)
        feats = feats.replace([np.inf, -np.inf], np.nan).fillna(0)
        out.append({
            "hash_id": hash_id,
            "features": feats.values,
            "seq_length": len(feats),
            "label": int(sdf["is_correct"].iloc[0]),
        })
    return out


def apply_scaler(samples: List[Dict], scaler: StandardScaler) -> List[Dict]:
    for s in samples:
        s["features_scaled"] = scaler.transform(s["features"])
    return samples


# ---------------------------------------------------------------------------
# Validation helpers (PIK/II/SAPLMA/BICR style)
# ---------------------------------------------------------------------------

def calculate_composite_score(auroc: float, ece: float, w: float = 0.6) -> float:
    return w * auroc + (1.0 - w) * (1.0 - ece)


def pick_tau_by_youden(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    taus = np.linspace(0.0, 1.0, 1001)
    best_j, best_tau = -1.0, 0.5
    for t in taus:
        yhat = (y_prob >= t).astype(int)
        tp = ((yhat == 1) & (y_true == 1)).sum()
        tn = ((yhat == 0) & (y_true == 0)).sum()
        fp = ((yhat == 1) & (y_true == 0)).sum()
        fn = ((yhat == 0) & (y_true == 1)).sum()
        sens = tp / (tp + fn + 1e-12)
        spec = tn / (tn + fp + 1e-12)
        j = sens + spec - 1.0
        if j > best_j:
            best_j, best_tau = j, t
    return float(best_tau)


def sens_spec_at_tau(y_true: np.ndarray, y_prob: np.ndarray, tau: float):
    yhat = (y_prob >= tau).astype(int)
    tp = ((yhat == 1) & (y_true == 1)).sum()
    tn = ((yhat == 0) & (y_true == 0)).sum()
    fp = ((yhat == 1) & (y_true == 0)).sum()
    fn = ((yhat == 0) & (y_true == 1)).sum()
    sens = tp / (tp + fn + 1e-12)
    spec = tn / (tn + fp + 1e-12)
    return float(sens), float(spec)


def calculate_validation_score(metrics, metric_name, auroc_weight, min_sens, min_spec,
                                y_true=None, y_prob=None):
    auroc = metrics.get('auroc', 0.0)
    ece = metrics.get('ece', 1.0)
    if metric_name == 'auroc':
        return auroc
    if metric_name == 'ece':
        return 1.0 - ece
    if metric_name == 'composite':
        return calculate_composite_score(auroc, ece, auroc_weight)
    if metric_name == 'auroc_with_constraint':
        return auroc * (1.0 - ece)
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


# ---------------------------------------------------------------------------
# Training + eval
# ---------------------------------------------------------------------------
def evaluate(classifier: ConvClassifierWithEmbedding,
             loader: DataLoader,
             device: torch.device,
             criterion: Optional[nn.Module] = None) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """Evaluate Stage 2 classifier. confidence_score = softmax(logits)[:, 1] = P(label=1) = P(correct)."""
    classifier.eval()
    preds_list: List[np.ndarray] = []
    labs_list: List[np.ndarray] = []
    total_loss = 0.0
    n_seen = 0
    with torch.no_grad():
        for feats, labels, seq_lens in loader:
            feats = feats.to(device)
            labels_dev = labels.to(device)
            logits, _ = classifier(feats, seq_lens)
            if criterion is not None:
                loss = criterion(logits, labels_dev)
                total_loss += loss.item() * feats.shape[0]
                n_seen += feats.shape[0]
            probs = F.softmax(logits, dim=1)[:, 1]
            preds_list.append(probs.cpu().numpy())
            labs_list.append(labels.cpu().numpy())
    preds = np.concatenate(preds_list).astype(np.float32)
    labs = np.concatenate(labs_list).astype(np.float32)
    metrics = calculate_all_metrics(labs, preds)
    if criterion is not None:
        metrics['loss'] = total_loss / max(n_seen, 1)
    return metrics, labs, preds


def train_joint(classifier: ConvClassifierWithEmbedding,
                train_loader: DataLoader,
                val_loader: DataLoader,
                pos_weight_ratio: float,
                device: torch.device,
                args_ref) -> Tuple[ConvClassifierWithEmbedding, Dict[str, Any]]:
    """SPARROW-aligned Stage 2: epoch-based with early stopping on composite."""
    classifier = classifier.to(device).train()
    # Joint fine-tuning: encoder parameters are trainable.
    for p in classifier.embedding_model.parameters():
        p.requires_grad = not FREEZE_ENCODER

    # Class-weighted CrossEntropyLoss is the 2-class equivalent of pos_weight in BCE.
    # weights = [1.0, pos_weight_ratio] up-weights the positive class proportional to imbalance.
    class_weights = torch.tensor([1.0, pos_weight_ratio], device=device, dtype=torch.float)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, classifier.parameters()),
        lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    history: Dict[str, Any] = {
        'train_loss': [], 'val_loss': [], 'val_metrics': [],
        'best_epoch': None, 'best_operating_point': None,
    }
    best_score = float('-inf')
    best_state = None
    best_op = None
    patience_ctr = 0

    for epoch in range(args_ref.max_epochs):
        classifier.train()
        # Re-enable encoder grads each epoch (eval flips train→eval but not requires_grad)
        for p in classifier.embedding_model.parameters():
            p.requires_grad = not FREEZE_ENCODER

        epoch_loss = 0.0
        n_seen = 0
        for feats, labels, seq_lens in train_loader:
            feats, labels = feats.to(device), labels.to(device)
            optimizer.zero_grad()
            logits, _ = classifier(feats, seq_lens)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), 1.0)
            optimizer.step()
            B = feats.shape[0]
            epoch_loss += loss.item() * B
            n_seen += B
        avg_train = epoch_loss / max(n_seen, 1)

        if epoch % args_ref.val_eval_frequency != 0:
            history['train_loss'].append(avg_train)
            history['val_loss'].append(None)
            history['val_metrics'].append(None)
            continue

        metrics, all_y, all_p = evaluate(classifier, val_loader, device, criterion=criterion)

        tau = pick_tau_by_youden(all_y, all_p)
        sens, spec = sens_spec_at_tau(all_y, all_p, tau)
        operating_point = {'tau': tau, 'sens': sens, 'spec': spec}

        if args_ref.validation_metric == 'composite_with_constraints' and epoch >= 2:
            feasible = (sens >= args_ref.min_sensitivity) and (spec >= args_ref.min_specificity)
            if not feasible:
                score = float('-inf')
            else:
                score = calculate_validation_score(
                    metrics, args_ref.validation_metric,
                    args_ref.composite_auroc_weight,
                    args_ref.min_sensitivity, args_ref.min_specificity,
                    all_y, all_p)
        else:
            score = calculate_validation_score(
                metrics, args_ref.validation_metric,
                args_ref.composite_auroc_weight,
                args_ref.min_sensitivity, args_ref.min_specificity,
                all_y, all_p)

        history['train_loss'].append(avg_train)
        history['val_loss'].append(metrics.get('loss'))
        history['val_metrics'].append(metrics)

        logger.info(f"Epoch {epoch:3d} | TrLoss={avg_train:.4f} VaLoss={metrics.get('loss', 0.0):.4f} "
                    f"AUROC={metrics['auroc']:.4f} ECE={metrics['ece']:.4f} "
                    f"Brier={metrics['brier']:.4f} Score={score:.4f}")

        if score > best_score:
            best_score = score
            best_state = copy.deepcopy(classifier.state_dict())
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
        classifier.load_state_dict(best_state)
    history['best_score'] = best_score
    history['best_operating_point'] = best_op
    return classifier, history


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    seed_everything(args.seed)
    logger.info("=" * 70)
    logger.info("CCPS Stage 2 (paper-fixed) — Classifier Joint Fine-tuning")
    logger.info("=" * 70)
    logger.info(f"Model: {args.model_name}  |  Seed: {args.seed}")
    logger.info(f"Classifier: Linear({EMBED_DIM}->{CLF_HIDDEN_DIMS[0]}) + "
                f"Linear({CLF_HIDDEN_DIMS[0]}->2)")
    logger.info(f"Optimizer: Adam, LR={LEARNING_RATE}, WD={WEIGHT_DECAY}, BS={BATCH_SIZE}")
    logger.info(f"Schedule: max_epochs={args.max_epochs} patience={args.patience} "
                f"val_metric={args.validation_metric} (auroc_w={args.composite_auroc_weight})")
    logger.info(f"Encoder frozen: {FREEZE_ENCODER}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    feature_dir     = Path(args.feature_dir)
    model_name_part = args.model_name.split("/")[-1]
    # SPARROW convention: --output-dir is the literal leaf (e.g. .../seed_{S}/).
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve Stage 1 encoder location
    if args.encoder_dir is not None:
        encoder_dir = Path(args.encoder_dir)
    else:
        # SPARROW convention: --encoder-output-dir is the literal Stage 1 leaf; best/ is a child.
        encoder_dir = Path(args.encoder_output_dir) / "best"

    if not encoder_dir.exists():
        raise FileNotFoundError(f"Stage 1 encoder dir not found: {encoder_dir}. "
                                f"Run CCPS_proj_paper_train.py first.")
    logger.info(f"Loading encoder from: {encoder_dir}")

    # Load Stage 1 metadata + preprocessing artifacts
    with open(encoder_dir / "model_info.json", "r") as f:
        encoder_info = json.load(f)
    stage1_parent = encoder_dir.parent
    with open(stage1_parent / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(stage1_parent / "seq_length_info.json", "r") as f:
        seq_info = json.load(f)
    max_seq_length = seq_info["max_seq_length"]
    feature_dim    = encoder_info["feature_dim"]

    # Sanity: paper config must match
    assert encoder_info["hidden_dims"]  == HIDDEN_DIMS,  "Encoder hidden_dims mismatch"
    assert encoder_info["kernel_sizes"] == KERNEL_SIZES, "Encoder kernel_sizes mismatch"
    assert encoder_info["config"]["embed_dim"] == EMBED_DIM, "Encoder embed_dim mismatch"

    # Build encoder + load Stage 1 weights
    encoder = ConvEmbeddingNet(input_dim=feature_dim)
    encoder.load_state_dict(torch.load(encoder_dir / "encoder_model.pt", map_location="cpu"))

    # Data
    train_df = load_features(feature_dir, args.model_name, args.train_dataset_name)
    val_df   = load_features(feature_dir, args.model_name, args.val_dataset_name)
    train_samples = prepare_sequential_data(train_df, max_seq_length)
    val_samples   = prepare_sequential_data(val_df,   max_seq_length)
    train_samples = apply_scaler(train_samples, scaler)
    val_samples   = apply_scaler(val_samples,   scaler)

    train_ds     = SequenceClassificationDataset(train_samples, max_seq_length)
    val_ds       = SequenceClassificationDataset(val_samples,   max_seq_length)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0, pin_memory=True)

    # pos_weight = n_neg / n_pos on the (no-swap) labels
    train_labels_arr = np.array([s["label"] for s in train_samples], dtype=int)
    n_pos = int((train_labels_arr == 1).sum())
    n_neg = int((train_labels_arr == 0).sum())
    pos_weight_ratio = (n_neg / n_pos) if n_pos > 0 else 1.0
    logger.info(f"pos_weight = n_neg/n_pos = {n_neg}/{n_pos} = {pos_weight_ratio:.3f} "
                f"(applied as CrossEntropyLoss class_weights=[1.0, {pos_weight_ratio:.3f}])")

    # Build classifier on top of (trainable) encoder and joint fine-tune
    classifier = ConvClassifierWithEmbedding(embedding_model=encoder)
    classifier, history = train_joint(classifier, train_loader, val_loader,
                                      pos_weight_ratio, device, args)

    # Final val metrics on best checkpoint
    val_metrics, _, _ = evaluate(classifier, val_loader, device)
    logger.info("\nValidation metrics on best checkpoint:")
    for k, v in val_metrics.items():
        logger.info(f"  {k}: {v:.4f}")

    # Save (layout mirrors CCPS_clf_train.py so downstream eval code works)
    best_dir = output_dir / "best"
    best_dir.mkdir(parents=True, exist_ok=True)
    torch.save(classifier.state_dict(), best_dir / "classifier_model.pt")

    model_info = {
        "encoder_info": encoder_info,
        "classifier_config": {
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "batch_size": BATCH_SIZE,
            "max_epochs": args.max_epochs,
            "patience": args.patience,
            "val_eval_frequency": args.val_eval_frequency,
            "validation_metric": args.validation_metric,
            "composite_auroc_weight": args.composite_auroc_weight,
            "classifier_dropout": DROPOUT,
            "freeze_encoder": FREEZE_ENCODER,
            "optimizer": "Adam",
            "loss": "CrossEntropyLoss(weight=[1, n_neg/n_pos])",
            "pos_weight_ratio": pos_weight_ratio,
        },
        "classifier_hidden_dims": CLF_HIDDEN_DIMS,
        "val_metrics": val_metrics,
        "best_epoch": history.get("best_epoch"),
        "best_operating_point": history.get("best_operating_point"),
        "max_seq_length": max_seq_length,
        "feature_dim": feature_dim,
        "freeze_encoder": FREEZE_ENCODER,
        "seed": args.seed,
    }
    with open(best_dir / "model_info.json", "w") as f:
        json.dump(model_info, f, indent=2)

    shutil.copy(stage1_parent / "scaler.pkl", best_dir / "scaler.pkl")

    logger.info(f"\nSaved Stage 2 model to: {best_dir}")
    logger.info("CCPS paper-fixed training complete.")


if __name__ == "__main__":
    main()

# Example:
# python models/CCPS_clf_paper_train.py \
#   --cuda-devices 0 \
#   --feature-dir ../data/CCPS_features/ \
#   --model-name Qwen/Qwen3-VL-8B-Instruct \
#   --train-dataset-name train \
#   --val-dataset-name validation \
#   --encoder-dir ./trained_models/SPARROW/CCPS_proj/Qwen3-VL-8B-Instruct/seed_23/Qwen3-VL-8B-Instruct/train/best/ \
#   --output-dir ./trained_models/SPARROW/CCPS/Qwen3-VL-8B-Instruct/seed_23/ \
#   --seed 23
