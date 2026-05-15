#!/usr/bin/env python3
"""
CCPS_paper_eval.py

Evaluation for the two-stage paper-fixed CCPS variant produced by
    models/CCPS_proj_paper_train.py  (Stage 1: contrastive encoder)
  → models/CCPS_clf_paper_train.py   (Stage 2: joint fine-tune)

The paper-fixed classifier head is:
    Linear(EMBED_DIM -> CLF_HIDDEN_DIMS[0]) + ReLU + Linear(CLF_HIDDEN_DIMS[0] -> 2)
There is NO dropout in the classifier, so the saved state dict has keys
`classifier.0.*` and `classifier.2.*` (not `classifier.3.*` as in the
generic Optuna-tuned variant CCPS_eval.py loads).

We therefore define the paper-fixed model here directly instead of reusing
CCPS_eval's generic ConvClassifierWithEmbedding.

SPARROW convention: --trained-model-path and --output-dir are literal leaves
when --seed is passed (no {MODEL}/{DATASET} subpath injection).
"""

import argparse
import json
import logging
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def parse_arguments():
    p = argparse.ArgumentParser(description="Evaluate paper-fixed CCPS (two-stage) models")
    p.add_argument("--feature-dir", type=str, required=True)
    p.add_argument("--model-name", type=str, required=True)
    p.add_argument("--test-dataset-name", type=str, required=True)
    p.add_argument("--train-dataset-name", type=str, required=True)
    p.add_argument("--trained-model-path", type=str, required=True)
    p.add_argument("--dataset-path", type=str, default=None)
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--cuda-devices", type=str, default="0")
    p.add_argument("--seed", type=int, default=None,
                   help="If set: treat --trained-model-path and --output-dir as literal leaves.")
    return p.parse_args()


args = parse_arguments()
os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices

import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from datasets import load_from_disk  # noqa: E402
from tqdm import tqdm  # noqa: E402

sys.path.append(str(Path(__file__).parent / "../utils"))
from eval import calculate_all_metrics, save_evaluation_results  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

EXCLUDE_COLUMNS = ["hash_id", "is_correct", "token_idx", "token_id", "token_str"]


# ============================================================================
# Paper-fixed architecture (must match CCPS_proj_paper_train.py /
#                                    CCPS_clf_paper_train.py exactly)
# ============================================================================

class ConvEmbeddingNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int],
                 kernel_sizes: List[int], embed_dim: int, dropout: float = 0.2,
                 activation: str = "relu"):
        super().__init__()
        act_map = {"relu": nn.ReLU(), "gelu": nn.GELU(), "silu": nn.SiLU()}
        self.activation = act_map.get(activation, nn.ReLU())
        self.conv_layers = nn.ModuleList()
        self.conv_layers.append(
            nn.Conv1d(input_dim, hidden_dims[0],
                      kernel_size=kernel_sizes[0], padding=kernel_sizes[0] // 2)
        )
        for i in range(1, len(hidden_dims)):
            k = kernel_sizes[min(i, len(kernel_sizes) - 1)]
            self.conv_layers.append(
                nn.Conv1d(hidden_dims[i - 1], hidden_dims[i],
                          kernel_size=k, padding=k // 2)
            )
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(hidden_dims[-1], embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, seq_lengths=None):
        x = x.transpose(1, 2)
        for c in self.conv_layers:
            x = self.activation(c(x))
            x = self.dropout(x)
        x = self.pool(x).squeeze(-1)
        return self.fc(x)


class PaperCCPSClassifier(nn.Module):
    """Paper-fixed: Linear(embed -> h) + ReLU + Linear(h -> 2). No dropout."""

    def __init__(self, embedding_model: ConvEmbeddingNet, embed_dim: int,
                 clf_hidden: int):
        super().__init__()
        self.embedding_model = embedding_model
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, clf_hidden),
            nn.ReLU(),
            nn.Linear(clf_hidden, 2),
        )

    def forward(self, x, seq_lengths=None):
        emb = self.embedding_model(x, seq_lengths)
        return self.classifier(emb), emb


# ============================================================================
# Loader
# ============================================================================

def load_paper_ccps(model_path: Path, device: torch.device):
    best = model_path / "best"
    if not best.exists():
        # SPARROW uses the leaf directly when --seed is set
        best = model_path
    with open(best / "model_info.json") as f:
        info = json.load(f)

    encoder_info = info["encoder_info"]
    embed_dim = encoder_info["config"]["embed_dim"]
    hidden_dims = encoder_info["hidden_dims"]
    kernel_sizes = encoder_info["kernel_sizes"]
    feature_dim = encoder_info["feature_dim"]
    encoder_dropout = encoder_info["config"].get("encoder_dropout", 0.2)
    clf_hidden_dims = info["classifier_hidden_dims"]
    assert len(clf_hidden_dims) == 1, \
        f"Paper variant expects CLF_HIDDEN_DIMS=[h], got {clf_hidden_dims}"
    clf_hidden = clf_hidden_dims[0]
    max_seq_length = info.get("max_seq_length", 64)

    encoder = ConvEmbeddingNet(
        input_dim=feature_dim, hidden_dims=hidden_dims,
        kernel_sizes=kernel_sizes, embed_dim=embed_dim,
        dropout=encoder_dropout,
    )
    model = PaperCCPSClassifier(encoder, embed_dim, clf_hidden)

    state_dict = torch.load(best / "classifier_model.pt", map_location="cpu")
    model.load_state_dict(state_dict, strict=True)

    with open(best / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    model.to(device).eval()
    return model, info, scaler, max_seq_length, feature_dim


# ============================================================================
# Eval
# ============================================================================

def load_test_features(feature_dir: Path, model_name: str,
                       test_dataset_name: str):
    model_name_part = model_name.split("/")[-1]
    path = feature_dir / model_name_part / test_dataset_name / "features.pkl"
    df = pd.read_pickle(path)
    logger.info(f"Loaded {len(df)} token rows from {path}")

    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLUMNS]
    ids = df["hash_id"].unique()

    samples = []
    for hid in tqdm(ids, desc="Preparing test sequences"):
        sub = df[df["hash_id"] == hid].sort_values("token_idx")
        feats = sub[feature_cols].copy()
        if "epsilon_to_flip_token" in feats.columns:
            feats["epsilon_to_flip_token"] = feats["epsilon_to_flip_token"].replace(
                [np.inf, -np.inf], 20.0)
        feats = feats.replace([np.inf, -np.inf], np.nan).fillna(0)
        samples.append({
            "hash_id": hid,
            "features": feats.values,
            "seq_length": len(feats),
            "label": int(sub["is_correct"].iloc[0]),
        })
    return samples, list(ids)


def load_hash_to_dataset(dataset_path: Optional[Path], test_dataset_name: str):
    if dataset_path is None:
        return {}
    for p in (dataset_path / test_dataset_name,
              dataset_path / "raw" / test_dataset_name):
        if p.exists():
            ds = load_from_disk(str(p))
            logger.info(f"Loading dataset mapping from {p}")
            return {str(ds[i].get("hash_id")): ds[i].get("dataset", "unknown")
                    for i in range(len(ds))}
    logger.warning(f"No dataset mapping found at {dataset_path}")
    return {}


def run_eval(model, samples, scaler, max_seq_length,
             device, batch_size=32):
    for s in samples:
        s["features_scaled"] = scaler.transform(s["features"])

    all_conf = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(samples), batch_size):
            batch = samples[i:i + batch_size]
            X = np.zeros((len(batch), max_seq_length,
                          batch[0]["features_scaled"].shape[1]))
            for j, s in enumerate(batch):
                L = min(s["seq_length"], max_seq_length)
                X[j, :L] = s["features_scaled"][:L]
            xt = torch.tensor(X, dtype=torch.float32, device=device)
            logits, _ = model(xt)
            all_conf.extend(F.softmax(logits, dim=1)[:, 1].cpu().numpy().tolist())
    return np.array(all_conf, dtype=float)


def main():
    model_name_part = args.model_name.split("/")[-1]
    feature_dir = Path(args.feature_dir)
    trained_model_base = Path(args.trained_model_path)
    output_base = Path(args.output_dir)
    dataset_path = Path(args.dataset_path) if args.dataset_path else None

    if args.seed is not None:
        trained_model_path = trained_model_base
        output_dir = output_base / args.test_dataset_name
    else:
        trained_model_path = trained_model_base / model_name_part / args.train_dataset_name
        output_dir = output_base / model_name_part / args.test_dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Loading paper-fixed CCPS from: {trained_model_path}")
    model, info, scaler, max_seq_length, _ = load_paper_ccps(
        trained_model_path, device)

    samples, ids = load_test_features(feature_dir, args.model_name,
                                       args.test_dataset_name)
    hash_to_dataset = load_hash_to_dataset(dataset_path, args.test_dataset_name)

    conf = run_eval(model, samples, scaler, max_seq_length, device)
    labels = np.array([s["label"] for s in samples], dtype=float)

    records = [{
        "sample_id": hid,
        "ground_truth_correctness": int(labels[i]),
        "confidence_score": float(conf[i]),
        "dataset": hash_to_dataset.get(str(hid), "unknown"),
    } for i, hid in enumerate(ids)]
    with open(output_dir / "test_labels.json", "w") as f:
        json.dump(records, f, indent=2)

    metrics = calculate_all_metrics(labels, conf)
    results = {
        "overall": {"n_samples": int(len(labels)), **metrics},
        "metadata": {
            "model_name": args.model_name,
            "model_name_part": model_name_part,
            "test_dataset_name": args.test_dataset_name,
            "evaluation_timestamp": str(np.datetime64("now")),
            "ccps_statistics": {
                "avg_confidence": float(conf.mean()),
                "confidence_std": float(conf.std()),
                "min_confidence": float(conf.min()),
                "max_confidence": float(conf.max()),
            },
            "ccps_model_info": {
                "model_path": str(trained_model_path),
                "feature_dim": info["feature_dim"],
                "embed_dim": info["encoder_info"]["config"]["embed_dim"],
                "classifier_hidden_dims": info["classifier_hidden_dims"],
                "max_seq_length": max_seq_length,
            },
        },
    }
    save_evaluation_results(results, output_dir / "test_results.json")
    logger.info(f"AUROC {metrics['auroc']:.4f}  ECE {metrics['ece']:.4f}  "
                f"Brier {metrics['brier']:.4f}  Accuracy {metrics['accuracy']:.4f}")
    print(f"\nEvaluation complete! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
