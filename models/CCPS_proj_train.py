#!/usr/bin/env python3
"""
CCPS_proj_paper_train.py
========================
Paper-exact CCPS Stage 1: Contrastive encoder (projector) pre-training.

Uses the OE-variant best architecture from Appendix E of the CCPS paper:
    Encoder (EOE):
        Conv1d(75 -> 64, kernel=3) + ReLU
        Conv1d(64 -> 32, kernel=3) + ReLU
        AdaptiveMaxPool1d(1)
        Linear(32 -> 16)

Training (Stage 1 only):
    Contrastive (max-margin) loss, margin=1.0
    AdamW, LR=1e-4, weight_decay=0.1, batch_size=32
    5000 steps (no Optuna, no early stopping, no dropout)

Output:
    {output-dir}/{MODEL}/{TRAIN_DATASET}/best/
        encoder_model.pt
        model_info.json
    {output-dir}/{MODEL}/{TRAIN_DATASET}/
        scaler.pkl
        seq_length_info.json

Run BEFORE CCPS_clf_train.py.
"""

import argparse
import os
import sys
from pathlib import Path


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="CCPS Stage 1 (paper-fixed): contrastive encoder training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--feature-dir", type=str, required=True)
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--train-dataset-name", type=str, required=True)
    parser.add_argument("--val-dataset-name", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="../trained_models/CCPS_proj_paper")
    parser.add_argument("--cuda-devices", type=str, default="0")
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--max-seq-length", type=int, default=64)
    return parser.parse_args()


args = parse_arguments()
os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices
print(f"Set CUDA_VISIBLE_DEVICES={args.cuda_devices}")

import json
import pickle
import logging
from typing import Dict, List, Optional, Tuple

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

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Paper-exact constants (Appendix E, OE variant)
# ---------------------------------------------------------------------------
HIDDEN_DIMS        = [64, 32]
KERNEL_SIZES       = [3, 3]
EMBED_DIM          = 16
LEARNING_RATE      = 1e-4
WEIGHT_DECAY       = 0.1
BATCH_SIZE         = 32
CONTRASTIVE_STEPS  = 5000
CONTRASTIVE_MARGIN = 1.0
DROPOUT            = 0.0
ACTIVATION         = "relu"

EXCLUDE_COLUMNS = {"hash_id", "is_correct", "token_idx", "token_id", "token_str",
                   "sample_id", "token_text", "dataset", "split"}


# ---------------------------------------------------------------------------
# Model
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


class ContrastiveLoss(nn.Module):
    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(self, out1, out2, label):
        dist = F.pairwise_distance(out1, out2)
        return torch.mean(
            (1 - label) * torch.pow(dist, 2)
            + label * torch.pow(torch.clamp(self.margin - dist, min=0.0), 2)
        )


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class SequenceContrastiveDataset(Dataset):
    def __init__(self, samples: List[Dict], max_seq_length: int):
        self.samples = samples
        self.max_seq_length = max_seq_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        anchor = self.samples[idx]
        anchor_feat = anchor["features_scaled"]
        anchor_lab  = anchor["label"]
        anchor_len  = min(anchor["seq_length"], self.max_seq_length)

        other_idx = np.random.choice(len(self.samples))
        while other_idx == idx:
            other_idx = np.random.choice(len(self.samples))
        other = self.samples[other_idx]
        other_feat = other["features_scaled"]
        other_lab  = other["label"]
        other_len  = min(other["seq_length"], self.max_seq_length)

        d = anchor_feat.shape[1]
        ap = np.zeros((self.max_seq_length, d))
        op = np.zeros((self.max_seq_length, d))
        ap[:anchor_len] = anchor_feat[:anchor_len]
        op[:other_len]  = other_feat[:other_len]

        pair_label = 1 if anchor_lab != other_lab else 0
        return (torch.tensor(ap, dtype=torch.float32),
                torch.tensor(op, dtype=torch.float32),
                torch.tensor(pair_label, dtype=torch.float32),
                torch.tensor(anchor_len, dtype=torch.long),
                torch.tensor(other_len,  dtype=torch.long))


# ---------------------------------------------------------------------------
# Data loading / preprocessing
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


def fit_scaler(train_samples: List[Dict]) -> StandardScaler:
    all_feats = np.vstack([s["features"] for s in train_samples])
    scaler = StandardScaler()
    scaler.fit(all_feats)
    logger.info(f"Fit scaler on {all_feats.shape[0]} tokens × {all_feats.shape[1]} features")
    return scaler


def apply_scaler(samples: List[Dict], scaler: StandardScaler) -> List[Dict]:
    for s in samples:
        s["features_scaled"] = scaler.transform(s["features"])
    return samples


# ---------------------------------------------------------------------------
# Training (fixed-step, no early stopping)
# ---------------------------------------------------------------------------
def train_contrastive(encoder: ConvEmbeddingNet,
                      train_loader: DataLoader,
                      device: torch.device) -> ConvEmbeddingNet:
    encoder = encoder.to(device).train()
    criterion = ContrastiveLoss(margin=CONTRASTIVE_MARGIN)
    optimizer = optim.AdamW(encoder.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    step = 0
    pbar = tqdm(total=CONTRASTIVE_STEPS, desc="Contrastive pre-training")
    while step < CONTRASTIVE_STEPS:
        for anchor, other, label, alen, olen in train_loader:
            if step >= CONTRASTIVE_STEPS:
                break
            anchor, other, label = anchor.to(device), other.to(device), label.to(device)
            optimizer.zero_grad()
            loss = criterion(encoder(anchor, alen), encoder(other, olen), label)
            loss.backward()
            optimizer.step()
            step += 1
            pbar.update(1)
    pbar.close()
    return encoder


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    seed_everything(args.seed)
    logger.info("=" * 70)
    logger.info("CCPS Stage 1 (paper-fixed) — Contrastive Encoder Training")
    logger.info("=" * 70)
    logger.info(f"Model: {args.model_name}  |  Seed: {args.seed}")
    logger.info(f"Encoder: Conv1d(input->64,k3) + Conv1d(64->32,k3) + Pool + Linear(32->{EMBED_DIM})")
    logger.info(f"Optimizer: AdamW, LR={LEARNING_RATE}, WD={WEIGHT_DECAY}, BS={BATCH_SIZE}")
    logger.info(f"Contrastive steps: {CONTRASTIVE_STEPS}  |  margin: {CONTRASTIVE_MARGIN}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    feature_dir     = Path(args.feature_dir)
    model_name_part = args.model_name.split("/")[-1]
    # SPARROW convention: --output-dir is the literal leaf (e.g. .../seed_{S}/).
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load features
    train_df = load_features(feature_dir, args.model_name, args.train_dataset_name)
    val_df   = load_features(feature_dir, args.model_name, args.val_dataset_name)

    # Prepare, label-map, scale
    train_samples = prepare_sequential_data(train_df, args.max_seq_length)
    val_samples   = prepare_sequential_data(val_df,   args.max_seq_length)

    scaler = fit_scaler(train_samples)
    train_samples = apply_scaler(train_samples, scaler)
    val_samples   = apply_scaler(val_samples,   scaler)

    # Save preprocessing artifacts (Stage 2 reads these)
    with open(output_dir / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open(output_dir / "seq_length_info.json", "w") as f:
        json.dump({"max_seq_length": args.max_seq_length}, f, indent=2)

    feature_dim = train_samples[0]["features_scaled"].shape[1]
    logger.info(f"Feature dim: {feature_dim}")

    # Train
    train_ds     = SequenceContrastiveDataset(train_samples, args.max_seq_length)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=True)

    encoder = ConvEmbeddingNet(input_dim=feature_dim)
    encoder = train_contrastive(encoder, train_loader, device)

    # Save (match existing CCPS_proj_train.py layout so downstream clf + eval work)
    best_dir = output_dir / "best"
    best_dir.mkdir(parents=True, exist_ok=True)
    torch.save(encoder.state_dict(), best_dir / "encoder_model.pt")

    encoder_info = {
        "config": {
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "batch_size": BATCH_SIZE,
            "contrastive_steps": CONTRASTIVE_STEPS,
            "contrastive_margin": CONTRASTIVE_MARGIN,
            "encoder_dropout": DROPOUT,          # kept under this key for downstream compat
            "embed_dim": EMBED_DIM,
            "activation": ACTIVATION,
            "optimizer": "AdamW",
            "variant": "OE (Appendix E)",
        },
        "feature_dim": feature_dim,
        "hidden_dims": HIDDEN_DIMS,
        "kernel_sizes": KERNEL_SIZES,
        "seed": args.seed,
    }
    with open(best_dir / "model_info.json", "w") as f:
        json.dump(encoder_info, f, indent=2)

    logger.info(f"Saved encoder to: {best_dir}")
    logger.info("Stage 1 done. Next: run CCPS_clf_paper_train.py pointing at this encoder.")


if __name__ == "__main__":
    main()

# Example:
# python models/CCPS_proj_paper_train.py \
#   --cuda-devices 0 \
#   --feature-dir ../data/CCPS_features/ \
#   --model-name Qwen/Qwen3-VL-8B-Instruct \
#   --train-dataset-name train \
#   --val-dataset-name validation \
#   --output-dir ./trained_models/SPARROW/CCPS_proj/Qwen3-VL-8B-Instruct/seed_23/ \
#   --seed 23
