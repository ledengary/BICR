#!/usr/bin/env python3
"""
InternalInspector (I²) Evaluation Script
==========================================
Loads a trained II model from a best/ directory produced by II_train.py
and evaluates it on a test split extracted by II_extraction_v2.py.

Label convention is fixed: 1 = correct, 0 = incorrect. confidence_score in
the saved test_labels.json is P(correct).

Outputs (mirrors PIK_eval.py):
  test_labels.json   – per-sample records with confidence scores + predictions
  test_results.json  – aggregate + per-dataset metrics
"""

# ── early CUDA_VISIBLE_DEVICES ──────────────────────────────────────────────
import os, argparse

def _pre_parse():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--cuda_devices", type=str, default="0")
    known, _ = p.parse_known_args()
    return known

_pre = _pre_parse()
os.environ["CUDA_VISIBLE_DEVICES"] = _pre.cuda_devices
print(f"Set CUDA_VISIBLE_DEVICES={_pre.cuda_devices}")

# ── standard imports ─────────────────────────────────────────────────────────
import json, logging, sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent / "../utils"))
from general import seed_everything
from eval import calculate_all_metrics, save_evaluation_results

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

RANDOM_SEED = 23
seed_everything(RANDOM_SEED)

VALID_STATE_COMBINATIONS = [
    "activation", "attention", "ff",
    "attn_ff", "attn_act", "ff_act",
    "all",
]


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Args                                                                     ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate a trained InternalInspector (I²) model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model_base_dir",     type=str, default="../trained_models/II",
                   help="Base directory for trained models (e.g., ../trained_models/II)")
    p.add_argument("--model_name",         type=str, required=True,
                   help="VLM model name, e.g. Qwen/Qwen3-VL-8B-Instruct")
    p.add_argument("--train_dataset_name", type=str, required=True,
                   help="Training dataset name used during training (e.g., train)")
    p.add_argument("--data_dir",           type=str, required=True,
                   help="Root of II_extraction output")
    p.add_argument("--test_dataset_name",  type=str, required=True)
    p.add_argument("--dataset_path",       type=str, default=None,
                   help="(Optional) Path to original dataset directory "
                        "(for per-dataset breakdown, e.g., ../data/VLCB/raw)")
    p.add_argument("--state_combination",  type=str, default=None,
                   choices=VALID_STATE_COMBINATIONS,
                   help="State combination. If None, auto-detected from trained model dir.")
    p.add_argument("--output_dir",         type=str, required=True,
                   help="Base directory to save evaluation results (e.g., ../results/II)")
    p.add_argument("--batch_size",         type=int, default=64)
    p.add_argument("--cuda_devices",       type=str, default="0")
    p.add_argument("--num_workers",        type=int, default=8,
                   help="Number of parallel workers for loading npz files")
    p.add_argument("--threshold",          type=float, default=None,
                   help="Decision threshold. If None, Youden-J is computed on the test set.")
    p.add_argument("--seed", type=int, default=None,
                   help="If set: treat --model_base_dir as the literal {best/}'s parent and "
                        "--output_dir as a literal leaf (no {MODEL}/{DATASET} injection).")
    return p.parse_args()


args = parse_args()


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Architecture  (must exactly match II_train.py)                           ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class _ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1      = nn.Conv2d(in_ch,  out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1        = nn.BatchNorm2d(out_ch)
        self.conv2      = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2        = nn.BatchNorm2d(out_ch)
        self.relu       = nn.ReLU(inplace=True)
        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class CNNEncoder(nn.Module):
    def __init__(self, in_channels: int, embed_dim: int):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.layer1 = nn.Sequential(_ResBlock(64,  64),  _ResBlock(64,  64))
        self.layer2 = nn.Sequential(_ResBlock(64,  128, stride=2), _ResBlock(128, 128))
        self.layer3 = nn.Sequential(_ResBlock(128, 256, stride=2), _ResBlock(256, 256))
        self.layer4 = nn.Sequential(_ResBlock(256, 512, stride=2), _ResBlock(512, 512))
        self.pool   = nn.AdaptiveAvgPool2d((1, 1))
        self.proj   = nn.Linear(512, embed_dim)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x); x = self.layer2(x)
        x = self.layer3(x); x = self.layer4(x)
        return self.proj(self.pool(x).flatten(1))


class MLPClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_layers: Tuple[int, ...], dropout: float):
        super().__init__()
        layers: List[nn.Module] = []
        curr = input_dim
        for h in hidden_layers:
            layers += [nn.Linear(curr, h), nn.ReLU(inplace=True), nn.Dropout(dropout)]
            curr = h
        layers.append(nn.Linear(curr, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class InternalInspectorModel(nn.Module):
    def __init__(self, in_channels: int, embed_dim: int,
                 hidden_layers: Tuple[int, ...], dropout: float):
        super().__init__()
        self.encoder    = CNNEncoder(in_channels=in_channels, embed_dim=embed_dim)
        self.classifier = MLPClassifier(input_dim=embed_dim,
                                        hidden_layers=hidden_layers, dropout=dropout)

    def forward(self, x):
        z     = self.encoder(x)
        z_n   = F.normalize(z, p=2, dim=-1)
        logit = self.classifier(z)
        return z_n, logit

    def predict_confidence(self, x):
        _, logit = self.forward(x)
        return torch.sigmoid(logit)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Model loader                                                             ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def load_trained_model(model_dir: Path,
                       device: torch.device) -> Tuple[InternalInspectorModel, Dict]:
    info_path  = model_dir / "model_info.json"
    model_path = model_dir / "model.pth"
    if not info_path.exists():
        raise FileNotFoundError(f"model_info.json not found in {model_dir}")
    if not model_path.exists():
        raise FileNotFoundError(f"model.pth not found in {model_dir}")

    with open(info_path) as f:
        info = json.load(f)

    hidden_layers = tuple(int(x) for x in info["hidden_layers"])
    model = InternalInspectorModel(
        in_channels=info["in_channels"],
        embed_dim=info["embed_dim"],
        hidden_layers=hidden_layers,
        dropout=info["dropout"],
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device).eval()

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model loaded: {model_dir.name}  "
                f"state_combo={info['state_combination']}  "
                f"in_channels={info['in_channels']}  "
                f"params={n_params:,}")
    return model, info


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Data loading                                                             ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def build_input_tensor(data, state_combo: str) -> np.ndarray:
    act  = data["activation_states"].astype(np.float32)
    attn = data["attention_states"].astype(np.float32)
    ff   = data["ff_states"].astype(np.float32)
    if state_combo == "activation":  return act[:, :, np.newaxis]
    if state_combo == "attention":   return attn[:, :, np.newaxis]
    if state_combo == "ff":          return ff[:, :, np.newaxis]
    if state_combo == "attn_ff":     return np.stack([attn, ff],  axis=-1)
    if state_combo == "attn_act":    return np.stack([attn, act], axis=-1)
    if state_combo == "ff_act":      return np.stack([ff,  act],  axis=-1)
    if state_combo == "all":         return np.stack([act, attn, ff], axis=-1)
    raise ValueError(f"Unknown state_combination: {state_combo}")


def _load_single_npz(args: Tuple[Path, str]) -> Optional[Tuple[np.ndarray, bool, str]]:
    """Parallel worker: returns (input_tensor, label, hash_id) or None if skipped."""
    file_path, state_combo = args
    try:
        data = np.load(file_path, allow_pickle=True)
        ic   = data["is_correct"]
        if ic is None or (isinstance(ic, np.ndarray) and ic.item() is None):
            return None
        input_tensor = build_input_tensor(data, state_combo)
        # Raw is_correct as bool — the SPARROW swap (if any) is applied
        # downstream once all samples are loaded.
        label = bool(ic)
        hid   = data["hash_id"]
        hash_id = str(hid.item() if isinstance(hid, np.ndarray) and hid.shape == () else hid)
        return (input_tensor, label, hash_id)
    except Exception:
        return None


def load_test_samples(samples_dir: Path,
                      state_combo: str,
                      num_workers: int = 8,
                      ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load test samples with fixed label convention (1 = correct, 0 = incorrect)."""
    npz_files = sorted(samples_dir.glob("*.npz"))
    logger.info(f"  Found {len(npz_files)} test .npz files in {samples_dir}")

    inputs_list, labels_list, ids_list = [], [], []
    skipped_count = 0

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(_load_single_npz, (f, state_combo)): f
                   for f in npz_files}
        for future in tqdm(as_completed(futures), total=len(futures),
                           desc="  Loading test samples", leave=False):
            result = future.result()
            if result is None:
                skipped_count += 1
            else:
                input_tensor, label, hash_id = result
                inputs_list.append(input_tensor)
                labels_list.append(label)
                ids_list.append(hash_id)

    if skipped_count:
        logger.info(f"  Skipped {skipped_count} samples (no label or load error).")
    if not inputs_list:
        raise ValueError(f"No valid test samples found in {samples_dir}")

    all_inputs = np.stack(inputs_list, axis=0).astype(np.float32)
    all_labels = np.array(labels_list, dtype=bool)

    # ── NaN / Inf guard ───────────────────────────────────────────────────
    nan_mask = np.isnan(all_inputs)
    inf_mask = np.isinf(all_inputs)
    if nan_mask.any() or inf_mask.any():
        logger.warning(f"  Found {nan_mask.sum()} NaN + {inf_mask.sum()} Inf in inputs — zeroing.")
        all_inputs[nan_mask] = 0.0
        all_inputs[inf_mask] = 0.0

    n_pos = int(np.sum(all_labels))
    n_neg = int(len(all_labels) - n_pos)
    logger.info(f"  Loaded {len(all_inputs)} samples | shape={all_inputs.shape} | "
                f"correct={n_pos}  incorrect={n_neg}")
    return all_inputs, all_labels, ids_list


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Per-dataset mapping                                                      ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def build_hash_to_dataset(dataset_path: Optional[str],
                          test_dataset_name: str) -> Dict[str, str]:
    if dataset_path is None:
        return {}
    from datasets import load_from_disk

    base = Path(dataset_path)
    possible_paths = [
        base / test_dataset_name,
        base / "raw" / test_dataset_name,
        base / test_dataset_name.replace("_raw", ""),
    ]

    full_path = None
    for p in possible_paths:
        if p.exists():
            full_path = p
            break

    if full_path is None:
        logger.warning(f"Dataset path not found. Tried: {possible_paths}. "
                       "Dataset property will not be included.")
        return {}

    try:
        logger.info(f"Loading dataset mapping from {full_path} …")
        ds = load_from_disk(str(full_path))
        mapping: Dict[str, str] = {}
        hash_ids      = ds["hash_id"]
        dataset_names = (ds["dataset"] if "dataset" in ds.column_names
                         else ["unknown"] * len(ds))
        for hid, dname in zip(hash_ids, dataset_names):
            if hid:
                mapping[str(hid)] = dname
        logger.info(f"Loaded dataset mapping for {len(mapping)} samples "
                    f"({len(set(mapping.values()))} unique datasets)")
        return mapping
    except Exception as e:
        logger.warning(f"Failed to load dataset mapping: {e}. "
                       "Dataset property will not be included.")
        return {}


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Threshold                                                                ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def find_best_threshold_youden(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    best_j, best_tau = -1.0, 0.5
    for t in np.linspace(0.0, 1.0, 1001):
        yhat = (y_prob >= t).astype(int)
        tp = np.sum((yhat == 1) & (y_true == 1))
        tn = np.sum((yhat == 0) & (y_true == 0))
        fp = np.sum((yhat == 1) & (y_true == 0))
        fn = np.sum((yhat == 0) & (y_true == 1))
        j  = tp / (tp + fn + 1e-12) + tn / (tn + fp + 1e-12) - 1.0
        if j > best_j:
            best_j, best_tau = j, t
    return float(best_tau)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Inference                                                                ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def run_inference(model: InternalInspectorModel,
                  inputs_np: np.ndarray,
                  batch_size: int,
                  device: torch.device,
                  ) -> Tuple[np.ndarray, np.ndarray]:
    # [N, L, d, C] → [N, C, L, d] channels-first for CNN
    inputs_np = np.transpose(inputs_np, (0, 3, 1, 2))
    all_confs, all_logits = [], []
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(inputs_np), batch_size), desc="Inference", leave=False):
            batch = torch.from_numpy(inputs_np[i:i + batch_size]).float().to(device)

            if torch.isnan(batch).any() or torch.isinf(batch).any():
                logger.warning(
                    f"  Batch {i//batch_size}: "
                    f"{torch.isnan(batch).sum().item()} NaN + "
                    f"{torch.isinf(batch).sum().item()} Inf in input — zeroing.")
                batch = torch.where(
                    torch.isnan(batch) | torch.isinf(batch),
                    torch.zeros_like(batch), batch)

            _, logit = model(batch)

            if torch.isnan(logit).any():
                logger.warning(f"  Batch {i//batch_size}: "
                               f"{torch.isnan(logit).sum().item()} NaN in logits — zeroing.")
                logit = torch.where(torch.isnan(logit), torch.zeros_like(logit), logit)

            conf_batch = torch.sigmoid(logit).cpu().numpy()

            if np.isnan(conf_batch).any():
                logger.warning(f"  Batch {i//batch_size}: "
                               f"{np.isnan(conf_batch).sum()} NaN in confidences — zeroing.")
                conf_batch = np.where(np.isnan(conf_batch),
                                      np.zeros_like(conf_batch), conf_batch)

            all_confs.append(conf_batch)
            all_logits.append(logit.cpu().numpy())

    confidences = np.concatenate(all_confs).astype(np.float32)
    logits      = np.concatenate(all_logits).astype(np.float32)

    nan_mask = np.isnan(confidences)
    inf_mask = np.isinf(confidences)
    if nan_mask.any() or inf_mask.any():
        logger.warning(f"  Final confidences: {nan_mask.sum()} NaN + "
                       f"{inf_mask.sum()} Inf — zeroing.")
        confidences[nan_mask] = 0.0
        confidences[inf_mask] = 0.0

    return confidences, logits


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Per-dataset metrics                                                      ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def compute_per_dataset_metrics(sample_ids: List[str],
                                all_labels: np.ndarray,
                                confidences: np.ndarray,
                                predictions: np.ndarray,
                                hash_to_dataset: Dict[str, str]) -> Dict:
    if not hash_to_dataset:
        return {}
    groups: Dict[str, List[int]] = {}
    for idx, hid in enumerate(sample_ids):
        groups.setdefault(hash_to_dataset.get(hid, "unknown"), []).append(idx)
    results: Dict[str, Any] = {}
    for sd, indices in sorted(groups.items()):
        sd_labels = all_labels[indices]
        sd_confs  = confidences[indices]
        sd_preds  = predictions[indices]
        if len(np.unique(sd_labels)) < 2:
            results[sd] = {"n_samples": len(sd_labels),
                           "n_correct": int(np.sum(sd_labels)),
                           "error": "single_class_in_subset"}
            continue
        try:
            m = calculate_all_metrics(sd_labels, sd_confs)
            m["n_samples"] = len(sd_labels)
            m["n_correct"] = int(np.sum(sd_labels))
            tp = int(np.sum((sd_preds==1)&(sd_labels==1)))
            tn = int(np.sum((sd_preds==0)&(sd_labels==0)))
            fp = int(np.sum((sd_preds==1)&(sd_labels==0)))
            fn = int(np.sum((sd_preds==0)&(sd_labels==1)))
            m.update(dict(tp=tp, tn=tn, fp=fp, fn=fn))
            results[sd] = m
        except Exception as e:
            results[sd] = {"error": str(e), "n_samples": len(sd_labels)}
    return results


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Model path construction                                                  ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def find_model_directory(model_base_dir: Path,
                         model_name: str,
                         train_dataset_name: str,
                         state_combination: Optional[str] = None,
                         ) -> Tuple[Path, Optional[str]]:
    """
    Construct the model directory path matching II_train.py output structure:
    {model_base_dir}/{model_name_part}/{train_dataset_name}/{state_combination}/best/

    Returns (model_dir, detected_state_combination).
    detected_state_combination is None when state_combination was given as arg.
    """
    model_name_part = model_name.split("/")[-1]
    base_path = model_base_dir / model_name_part / train_dataset_name

    if not base_path.exists():
        raise FileNotFoundError(
            f"Model base path not found: {base_path}\n"
            f"Expected: {{model_base_dir}}/{{model_name}}/{{train_dataset_name}}"
            f"/{{state_combination}}/best/")

    if state_combination is not None:
        model_dir = base_path / state_combination / "best"
        if not model_dir.exists():
            available = [d.name for d in base_path.iterdir() if d.is_dir()]
            raise FileNotFoundError(
                f"Model directory not found: {model_dir}\n"
                f"Available state combinations: {available}")
        return model_dir, None

    # Auto-detect: search for best/ directories
    best_dirs = [
        (sub.name, sub / "best")
        for sub in base_path.iterdir()
        if sub.is_dir() and (sub / "best" / "model.pth").exists()
    ]

    if not best_dirs:
        available = [d.name for d in base_path.iterdir() if d.is_dir()]
        raise FileNotFoundError(
            f"No valid best/ directories found in {base_path}\n"
            f"Available subdirs: {available}")

    if len(best_dirs) > 1:
        logger.warning(
            f"Multiple state combinations found: {[n for n, _ in best_dirs]}. "
            f"Using: {best_dirs[0][0]}. Pass --state_combination to be explicit.")

    detected = best_dirs[0][0]
    return best_dirs[0][1], detected


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Results printing                                                         ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def print_results_summary(results: Dict[str, Any],
                          per_dataset: Dict[str, Any]) -> None:
    print(f"\nTEST II EVALUATION SUMMARY:")
    print("=" * 60)
    if "overall" in results:
        o = results["overall"]
        print(f"Overall Performance:")
        print(f"  Samples      : {o['n_samples']}")
        print(f"  Accuracy     : {o['accuracy']:.4f}")
        print(f"  Precision    : {o['precision']:.4f}")
        print(f"  Recall       : {o['recall']:.4f}")
        print(f"  F1-Score     : {o['f1']:.4f}")
        print(f"  Sensitivity  : {o['sensitivity']:.4f}")
        print(f"  Specificity  : {o['specificity']:.4f}")
        print(f"  ECE          : {o['ece']:.4f}")
        print(f"  Brier Score  : {o['brier']:.4f}")
        print(f"  AUROC        : {o['auroc']:.4f}")
        print(f"  AUCPR        : {o['aucpr']:.4f}")
    if "metadata" in results and "ii_statistics" in results["metadata"]:
        s = results["metadata"]["ii_statistics"]
        print(f"\nConfidence Statistics:")
        print(f"  Mean  : {s['avg_confidence']:.3f}")
        print(f"  Std   : {s['confidence_std']:.3f}")
        print(f"  Range : [{s['min_confidence']:.3f}, {s['max_confidence']:.3f}]")
    print(f"\nLabel convention: 1 = correct, 0 = incorrect.")
    if per_dataset:
        print(f"\nPer-Dataset AUROC:")
        for sd, m in sorted(per_dataset.items()):
            if "auroc" in m:
                print(f"  {sd:<30s}: AUROC={m['auroc']:.4f}  "
                      f"ECE={m.get('ece', float('nan')):.4f}  "
                      f"n={m['n_samples']}")
    print("=" * 60)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Main                                                                     ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def main():
    logger.info("InternalInspector (I²) Evaluation")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"  device : {device}")

    # ── find model ─────────────────────────────────────────────────────────
    model_base_dir = Path(args.model_base_dir)
    if args.seed is not None:
        # SPARROW mode: --model_base_dir already points at the seed_{S} folder;
        # best/ is its immediate child.
        model_dir = model_base_dir / "best"
        if not model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")
        detected_state_combo = None
    else:
        model_dir, detected_state_combo = find_model_directory(
            model_base_dir=model_base_dir,
            model_name=args.model_name,
            train_dataset_name=args.train_dataset_name,
            state_combination=args.state_combination,
        )
    logger.info(f"  model_dir : {model_dir}")

    # ── load model ─────────────────────────────────────────────────────────
    model, model_info = load_trained_model(model_dir, device)

    # ── resolve state combination ──────────────────────────────────────────
    if args.state_combination is not None:
        state_combo = args.state_combination
    elif detected_state_combo is not None:
        state_combo = detected_state_combo
    else:
        state_combo = model_info.get("state_combination", "all")

    stored = model_info.get("state_combination")
    if stored and stored != state_combo:
        logger.warning(f"  state_combination mismatch: using '{state_combo}' "
                       f"but model_info says '{stored}'.")
    logger.info(f"  state_combination : {state_combo}")

    # ── paths ──────────────────────────────────────────────────────────────
    model_name_part = args.model_name.split("/")[-1]
    test_samples    = (Path(args.data_dir) / model_name_part
                       / args.test_dataset_name / "samples")
    if not test_samples.exists():
        raise FileNotFoundError(f"Test samples directory not found: {test_samples}")

    if args.seed is not None:
        output_dir = Path(args.output_dir) / args.test_dataset_name
    else:
        output_dir = Path(args.output_dir) / model_name_part / args.test_dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"  model_name        : {args.model_name}")
    logger.info(f"  train_dataset     : {args.train_dataset_name}")
    logger.info(f"  test_dataset      : {args.test_dataset_name}")
    logger.info(f"  data_dir          : {args.data_dir}")
    logger.info(f"  output_dir        : {output_dir}")

    # ── load test data ─────────────────────────────────────────────────────
    test_inputs_np, test_labels, sample_ids = load_test_samples(
        test_samples, state_combo, num_workers=args.num_workers)

    # ── per-dataset mapping ───────────────────────────────────────────────
    hash_to_dataset = build_hash_to_dataset(args.dataset_path, args.test_dataset_name)

    # ── inference ──────────────────────────────────────────────────────────
    logger.info("Running inference …")
    confidences, logits = run_inference(model, test_inputs_np, args.batch_size, device)

    # ── threshold ──────────────────────────────────────────────────────────
    if args.threshold is not None:
        threshold = args.threshold
        logger.info(f"  Using provided threshold: {threshold:.4f}")
    else:
        threshold = find_best_threshold_youden(test_labels, confidences)
        logger.info(f"  Youden-J threshold: {threshold:.4f}")

    predictions = (confidences >= threshold).astype(int)

    # ── aggregate metrics ──────────────────────────────────────────────────
    agg = calculate_all_metrics(test_labels, confidences, threshold=threshold)
    tp  = int(np.sum((predictions==1)&(test_labels==1)))
    tn  = int(np.sum((predictions==0)&(test_labels==0)))
    fp  = int(np.sum((predictions==1)&(test_labels==0)))
    fn  = int(np.sum((predictions==0)&(test_labels==1)))
    agg.update(dict(threshold=threshold, tp=tp, tn=tn, fp=fp, fn=fn))

    # ── per-dataset ────────────────────────────────────────────────────────
    per_dataset = compute_per_dataset_metrics(
        sample_ids, test_labels, confidences, predictions, hash_to_dataset)

    # ── per-sample records → test_labels.json ─────────────────────────────
    # confidence_score = P(correct); label is 1 if the LVLM response was correct.
    evaluation_records = []
    matched_count = 0
    for idx, sample_id in enumerate(sample_ids):
        gt_label_1   = bool(test_labels[idx])
        conf_score   = float(confidences[idx])
        pred_label_1 = bool(predictions[idx])

        dataset_name = hash_to_dataset.get(sample_id, "unknown")
        if dataset_name != "unknown":
            matched_count += 1

        evaluation_records.append({
            "sample_id":                sample_id,
            "ground_truth_correctness": int(gt_label_1),
            "confidence_score":         conf_score,
            "dataset":                  dataset_name,
            "logit":                    float(logits[idx]),
            "predicted_correct":        pred_label_1,
            "prediction_match":         gt_label_1 == pred_label_1,
        })

    if len(hash_to_dataset) > 0:
        if matched_count == 0:
            logger.warning(
                f"No hash_id matches found! Mapping has {len(hash_to_dataset)} entries "
                f"but none matched.\n"
                f"  First 5 sample_ids : {sample_ids[:5]}\n"
                f"  First 5 map keys   : {list(hash_to_dataset.keys())[:5]}")
        elif matched_count < len(sample_ids):
            logger.warning(f"Only {matched_count}/{len(sample_ids)} samples matched dataset mapping.")

    labels_path = output_dir / "test_labels.json"
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(evaluation_records, f, indent=2, ensure_ascii=False)
    logger.info(f"Evaluation records saved → {labels_path}")

    # ── test_results.json ──────────────────────────────────────────────────
    results = {
        "overall": {
            "n_samples":       len(test_labels),
            "n_total_samples": len(test_labels),
            **agg,
        },
        "metadata": {
            "model_name":            args.model_name,
            "model_name_part":       model_name_part,
            "test_dataset_name":     args.test_dataset_name,
            "total_records":         len(evaluation_records),
            "evaluation_timestamp":  datetime.now().isoformat(),
            "model_dir":             str(model_dir),
            "state_combination":     state_combo,
            "threshold":             threshold,
            "ii_statistics": {
                "avg_confidence": float(np.mean(confidences)),
                "confidence_std": float(np.std(confidences)),
                "min_confidence": float(np.min(confidences)),
                "max_confidence": float(np.max(confidences)),
            },
            "ii_model_info": {
                "model_path":    str(model_dir),
                "in_channels":   model_info.get("in_channels"),
                "embed_dim":     model_info.get("embed_dim"),
                "hidden_layers": model_info.get("hidden_layers"),
                "dropout":       model_info.get("dropout"),
            },
            "per_dataset_metrics": per_dataset,
        },
    }

    results_path = output_dir / "test_results.json"
    save_evaluation_results(results, results_path)
    logger.info(f"Evaluation results saved → {results_path}")

    print_results_summary(results, per_dataset)
    print(f"\nEvaluation complete! Results saved to: {output_dir}")
    print("\nGenerated files:")
    print("  test_labels.json  – per-sample records with ground truth and confidence scores")
    print("  test_results.json – comprehensive evaluation metrics")


if __name__ == "__main__":
    main()


# Qwen/Qwen3-VL-8B-Instruct
# llava-hf/llava-v1.6-vicuna-13b-hf
# OpenGVLab/InternVL3_5-14B-HF
# google/gemma-3-27b-it
# deepseek-ai/deepseek-vl2

# ── Example ───────────────────────────────────────────────────────────────────
# python II_eval.py \
#   --model_base_dir      ../trained_models/II \
#   --model_name          Qwen/Qwen3-VL-8B-Instruct \
#   --train_dataset_name  train \
#   --data_dir            ../data/II_extraction_v2/ \
#   --test_dataset_name   test \
#   --dataset_path        ../data/VLCB/raw \
#   --output_dir          ../results/II \
#   --num_workers         16 \
#   --batch_size          64 \
#   --cuda_devices        7

# ── Ablation eval ─────────────────────────────────────────────────────────────
# python II_eval.py \
#   --model_base_dir      ../trained_models/II \
#   --model_name          Qwen/Qwen3-VL-8B-Instruct \
#   --train_dataset_name  train \
#   --state_combination   ff \
#   --data_dir            ../data/II_extraction_v2/ \
#   --test_dataset_name   test \
#   --dataset_path        ../data/VLCB/raw \
#   --output_dir          ../results/II \
#   --batch_size          64 \
#   --cuda_devices        0