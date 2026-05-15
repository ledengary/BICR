#!/usr/bin/env python3
"""
InternalInspector (I²) Training Script
========================================
Uses the exact best-performing configuration reported in the paper:
  • Encoder    : ResNet18-style CNN
  • Classifier : 3-layer MLP  (hidden: 256, 128, 64 | ReLU | dropout=0.1)
  • Loss        : L_contrastive + L_cls  (supervised contrastive + unweighted BCE)
  • Optimizer   : Adam  lr=0.001, weight_decay=1e-4  (L2 reg)
  • Contrastive temperature: 0.1
  • All layers, early stopping on composite val metric (AUROC + ECE)

Label convention is fixed: 1 = correct, 0 = incorrect. BCE loss uses
pos_weight = n_neg / n_pos (departs from the paper's unweighted Eq. 4 to be
comparable head-to-head with PIK / SAPLMA / BICR / CCPS).

State combinations (--state_combination, default="all"):
  activation          →  [L, d, 1]   activation states only
  attention           →  [L, d, 1]   attention states only
  ff                  →  [L, d, 1]   feed-forward states only
  attn_ff             →  [L, d, 2]   attention + ff
  attn_act            →  [L, d, 2]   attention + activation
  ff_act              →  [L, d, 2]   ff + activation
  all                 →  [L, d, 3]   all three  ← paper best / default

Loads only the .npz samples present on disk – works for any split size.
Saves best/model.pth + best/model_info.json (compatible with II_eval.py).
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
import copy, json, logging, sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent / "../utils"))
from general import seed_everything
from eval import calculate_all_metrics

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

RANDOM_SEED = 23
seed_everything(RANDOM_SEED)

# ── paper's exact best hyper-parameters (fixed, not tuned) ───────────────────
LR                = 0.001
WEIGHT_DECAY      = 1e-4
DROPOUT           = 0.1
CONTRASTIVE_TEMP  = 0.1
CLASSIFIER_HIDDEN = (256, 128, 64)
EMBED_DIM         = 128

# ── valid state combinations and their channel counts ────────────────────────
VALID_STATE_COMBINATIONS = [
    "activation", "attention", "ff",
    "attn_ff", "attn_act", "ff_act",
    "all",
]
STATE_COMBINATION_CHANNELS = {
    "activation": 1, "attention": 1, "ff": 1,
    "attn_ff": 2,    "attn_act": 2,  "ff_act": 2,
    "all": 3,
}


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Args                                                                     ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def parse_args():
    p = argparse.ArgumentParser(
        description="Train InternalInspector (I²) – paper best config",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data_dir",               type=str, required=True,
                   help="Root of II_extraction output")
    p.add_argument("--model_name",             type=str, required=True,
                   help="VLM model name, e.g. Qwen/Qwen3-VL-8B-Instruct")
    p.add_argument("--train_dataset_name",     type=str, required=True)
    p.add_argument("--val_dataset_name",       type=str, required=True)
    p.add_argument("--output_dir",             type=str, default="../trained_models/II")
    p.add_argument("--state_combination",      type=str, default="all",
                   choices=VALID_STATE_COMBINATIONS,
                   help="Which internal state types to use as input channels. "
                        "Default 'all' uses all three (paper best). "
                        "Use single types or pairs for ablation studies.")
    p.add_argument("--cuda_devices",           type=str, default="0")
    p.add_argument("--batch_size",             type=int, default=32)
    p.add_argument("--max_epochs",             type=int, default=200)
    p.add_argument("--patience",               type=int, default=20)
    p.add_argument("--composite_auroc_weight", type=float, default=0.6)
    p.add_argument("--seed", type=int, default=None,
                   help="If set: seeds everything with this value and treats --output_dir as the literal leaf "
                        "(no {MODEL}/{DATASET}/{STATE_COMBINATION} injection).")
    return p.parse_args()


args = parse_args()
if args.seed is not None:
    seed_everything(args.seed)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Data loading                                                             ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def build_input_tensor(data, state_combo: str) -> np.ndarray:
    """
    Build [L, d, C] input array from a loaded .npz sample.
    States are stored as float16 on disk, loaded as float32.
    """
    act  = data["activation_states"].astype(np.float32)   # [L, d]
    attn = data["attention_states"].astype(np.float32)    # [L, d]
    ff   = data["ff_states"].astype(np.float32)           # [L, d]

    if state_combo == "activation":  return act[:, :, np.newaxis]           # [L, d, 1]
    if state_combo == "attention":   return attn[:, :, np.newaxis]
    if state_combo == "ff":          return ff[:, :, np.newaxis]
    if state_combo == "attn_ff":     return np.stack([attn, ff],  axis=-1)  # [L, d, 2]
    if state_combo == "attn_act":    return np.stack([attn, act], axis=-1)
    if state_combo == "ff_act":      return np.stack([ff,  act],  axis=-1)
    if state_combo == "all":         return np.stack([act, attn, ff], axis=-1)  # [L, d, 3]
    raise ValueError(f"Unknown state_combination: {state_combo}")


def load_samples(samples_dir: Path,
                 state_combo: str,
                 ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load every .npz present in samples_dir that has a valid is_correct label.

    Label convention is fixed: 1 = correct, 0 = incorrect. Class imbalance is
    handled by BCE pos_weight = n_neg / n_pos at training time.
    """
    npz_files = sorted(samples_dir.glob("*.npz"))
    logger.info(f"  Found {len(npz_files)} .npz files in {samples_dir}")

    inputs_list, labels_list, ids_list = [], [], []
    skipped_nc = skipped_err = 0

    for f in tqdm(npz_files, desc=f"  Loading {samples_dir.parent.name}", leave=False):
        try:
            data = np.load(f, allow_pickle=True)
            ic   = data["is_correct"]
            if ic is None or (isinstance(ic, np.ndarray) and ic.item() is None):
                skipped_nc += 1
                continue
            inputs_list.append(build_input_tensor(data, state_combo))
            labels_list.append(bool(ic))
            hid = data["hash_id"]
            ids_list.append(
                str(hid.item() if isinstance(hid, np.ndarray) and hid.shape == () else hid))
        except Exception as e:
            skipped_err += 1
            if skipped_err <= 3:
                logger.warning(f"  Error loading {f.name}: {e}")

    if skipped_nc:
        logger.info(f"  Skipped {skipped_nc} samples (no correctness label).")
    if skipped_err:
        logger.warning(f"  Skipped {skipped_err} samples (load error).")
    if not inputs_list:
        raise ValueError(f"No valid samples found in {samples_dir}")

    all_inputs = np.stack(inputs_list, axis=0).astype(np.float32)
    all_labels = np.array(labels_list, dtype=bool)

    nan_mask = np.isnan(all_inputs)
    inf_mask = np.isinf(all_inputs)
    if nan_mask.any() or inf_mask.any():
        logger.warning(f"  Zeroing {int(nan_mask.sum())} NaN and {int(inf_mask.sum())} Inf values in inputs.")
        all_inputs[nan_mask] = 0.0
        all_inputs[inf_mask] = 0.0

    n_pos = int(np.sum(all_labels))
    n_neg = int(len(all_labels) - n_pos)
    logger.info(f"  Loaded {len(all_inputs)} samples | shape={all_inputs.shape} | "
                f"correct={n_pos}  incorrect={n_neg}")
    return all_inputs, all_labels, ids_list


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  GPU-resident dataset                                                     ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class IIDataset:
    """Pre-loads all data to GPU as float32. Inputs stored [N, C, L, d] channels-first."""

    def __init__(self, inputs: np.ndarray, labels: np.ndarray, device: Optional[str] = None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        inputs_t    = np.transpose(inputs, (0, 3, 1, 2))   # [N, L, d, C] → [N, C, L, d]
        self.inputs = torch.from_numpy(inputs_t).float().to(self.device)
        self.labels = torch.from_numpy(labels.astype(np.float32)).to(self.device)

    def __len__(self):
        return len(self.inputs)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Model  –  exact paper architecture                                       ║
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
    """ResNet18-style encoder.  Input [B, C, L, d]  →  output [B, embed_dim]."""

    def __init__(self, in_channels: int, embed_dim: int = EMBED_DIM):
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
    """3-layer MLP with ReLU + dropout, as specified in the paper."""

    def __init__(self, input_dim: int,
                 hidden: Tuple[int, ...] = CLASSIFIER_HIDDEN,
                 dropout: float = DROPOUT):
        super().__init__()
        layers: List[nn.Module] = []
        curr = input_dim
        for h in hidden:
            layers += [nn.Linear(curr, h), nn.ReLU(inplace=True), nn.Dropout(dropout)]
            curr = h
        layers.append(nn.Linear(curr, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class InternalInspectorModel(nn.Module):

    def __init__(self, in_channels: int):
        super().__init__()
        self.encoder    = CNNEncoder(in_channels=in_channels, embed_dim=EMBED_DIM)
        self.classifier = MLPClassifier(input_dim=EMBED_DIM)

    def forward(self, x):
        z     = self.encoder(x)
        z_n   = F.normalize(z, p=2, dim=-1)
        logit = self.classifier(z)
        return z_n, logit

    def predict_confidence(self, x):
        _, logit = self.forward(x)
        return torch.sigmoid(logit)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Loss                                                                     ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def supervised_contrastive_loss(embeddings: torch.Tensor,
                                 labels: torch.Tensor,
                                 temperature: float = CONTRASTIVE_TEMP
                                 ) -> torch.Tensor:
    """
    Paper eq. 3: for each anchor pick one random positive,
    use all negatives as denominator.
    embeddings : [B, D]  L2-normalised
    labels     : [B]     float  {0., 1.}
    """
    B      = embeddings.shape[0]
    device = embeddings.device
    sim    = torch.matmul(embeddings, embeddings.T) / temperature  # [B, B]
    eye    = torch.eye(B, dtype=torch.bool, device=device)
    same   = (labels.unsqueeze(1) == labels.unsqueeze(0)) & ~eye
    diff   = ~same & ~eye

    total, count = torch.tensor(0.0, device=device), 0
    for j in range(B):
        pos = same[j].nonzero(as_tuple=False).squeeze(1)
        neg = diff[j]
        if len(pos) == 0 or neg.sum() == 0:
            continue
        chosen = pos[torch.randint(len(pos), (1,), device=device)].item()
        total += -(sim[j, chosen] - torch.logsumexp(sim[j][neg], dim=0))
        count += 1
    return total / max(count, 1)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Validation                                                               ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def composite_score(auroc: float, ece: float, w: float = 0.6) -> float:
    return w * auroc + (1.0 - w) * (1.0 - ece)


def evaluate(model, dataset: IIDataset, batch_size: int, device: torch.device
             ) -> Tuple[Dict, np.ndarray, np.ndarray]:
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for i in range(0, len(dataset), batch_size):
            sl = slice(i, i + batch_size)
            x = dataset.inputs[sl].to(device) if dataset.inputs.device.type == "cpu" else dataset.inputs[sl]
            y = dataset.labels[sl].to(device) if dataset.labels.device.type == "cpu" else dataset.labels[sl]

            if torch.isnan(x).any() or torch.isinf(x).any():
                x = torch.where(torch.isnan(x) | torch.isinf(x), torch.zeros_like(x), x)

            probs_batch = model.predict_confidence(x).cpu()

            if torch.isnan(probs_batch).any():
                nan_count_pred = torch.isnan(probs_batch).sum().item()
                logger.warning(f"  Batch {i//batch_size}: {nan_count_pred} NaN predictions, zeroing.")
                probs_batch = torch.where(torch.isnan(probs_batch),
                                          torch.zeros_like(probs_batch), probs_batch)

            all_probs.append(probs_batch)
            all_labels.append(y.cpu())

    probs  = torch.cat(all_probs).numpy()
    labels = torch.cat(all_labels).numpy()

    nan_mask_probs = np.isnan(probs)
    if np.sum(nan_mask_probs) > 0:
        logger.warning(f"  {np.sum(nan_mask_probs)} NaN values in final predictions — zeroing.")
        probs[nan_mask_probs] = 0.0

    return calculate_all_metrics(labels, probs), labels, probs


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Training loop                                                            ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def train(model: InternalInspectorModel,
          train_ds: IIDataset,
          val_ds: IIDataset,
          batch_size: int,
          max_epochs: int,
          patience: int,
          auroc_w: float,
          device: torch.device,
          pos_weight: torch.Tensor) -> Tuple[InternalInspectorModel, Dict]:

    model     = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # SPARROW alignment: BCE with pos_weight = n_neg / n_pos.
    # Departs from the paper's unweighted Eq. 4 to be comparable with PIK / SAPLMA / BICR / CCPS.

    n_train = len(train_ds)
    best_score   = float("-inf")
    best_state   = None
    best_epoch   = 0
    patience_ctr = 0
    history      = []

    for epoch in range(max_epochs):
        # ── train ──────────────────────────────────────────────────────────
        model.train()
        perm       = torch.randperm(n_train, device=device)
        epoch_loss = 0.0

        for i in range(0, n_train, batch_size):
            idx = perm[i:i + batch_size]
            x, y = train_ds.inputs[idx], train_ds.labels[idx]

            if torch.isnan(x).any() or torch.isinf(x).any():
                nan_count = torch.isnan(x).sum().item()
                inf_count = torch.isinf(x).sum().item()
                logger.warning(f"  Epoch {epoch}, batch {i//batch_size}: "
                               f"{nan_count} NaN + {inf_count} Inf in input — zeroing.")
                x = torch.where(torch.isnan(x) | torch.isinf(x), torch.zeros_like(x), x)

            optimizer.zero_grad()
            z, logits = model(x)

            if torch.isnan(logits).any() or torch.isnan(z).any():
                logger.warning(f"  Epoch {epoch}, batch {i//batch_size}: NaN in model output.")
                logits = torch.where(torch.isnan(logits), torch.zeros_like(logits), logits)
                z      = torch.where(torch.isnan(z),      torch.zeros_like(z),      z)

            # SPARROW-aligned: pos_weight-balanced BCE + supervised contrastive (Eq. 3)
            l_cls   = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight)
            l_contr = supervised_contrastive_loss(z, y)
            loss    = l_cls + l_contr

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item() * len(idx)

        avg_train = epoch_loss / n_train

        # ── validate ───────────────────────────────────────────────────────
        metrics, _, _ = evaluate(model, val_ds, batch_size, device)
        score         = composite_score(metrics["auroc"], metrics["ece"], auroc_w)
        history.append({"epoch": epoch, "train_loss": avg_train,
                        "auroc": metrics["auroc"], "ece": metrics["ece"],
                        "score": score})

        logger.info(f"Epoch {epoch:3d} | train_loss={avg_train:.4f}  "
                    f"AUROC={metrics['auroc']:.4f}  ECE={metrics['ece']:.4f}  "
                    f"score={score:.4f}")

        if score > best_score:
            best_score   = score
            best_state   = copy.deepcopy(model.state_dict())
            best_epoch   = epoch
            patience_ctr = 0
            logger.info(f"  *** new best: {score:.4f} ***")
        else:
            patience_ctr += 1

        if patience_ctr >= patience:
            logger.info(f"Early stopping at epoch {epoch}.")
            break

    model.load_state_dict(best_state)
    logger.info(f"Best epoch: {best_epoch}  best score: {best_score:.4f}")
    return model, {"history": history, "best_epoch": best_epoch,
                   "best_score": best_score}


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Save                                                                     ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def save_model(model: InternalInspectorModel,
               output_dir: Path,
               val_metrics: Dict,
               state_combo: str,
               num_layers: int,
               hidden_dim: int,
               in_channels: int,
               train_history: Dict):
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "model.pth")

    info = {
        "state_combination": state_combo,
        "encoder_type":      "cnn",
        "num_layers":        num_layers,
        "hidden_dim":        hidden_dim,
        "in_channels":       in_channels,
        "embed_dim":         EMBED_DIM,
        "classifier_layers": ",".join(str(h) for h in CLASSIFIER_HIDDEN),
        "hidden_layers":     list(CLASSIFIER_HIDDEN),
        "dropout":           DROPOUT,
        "contrastive_temp":  CONTRASTIVE_TEMP,
        "lr":                LR,
        "weight_decay":      WEIGHT_DECAY,
        "loss":              "BCEWithLogitsLoss(pos_weight) + supervised_contrastive",
        "metrics":           val_metrics,
        "n_params":          count_params(model),
        "train_history":     train_history,
    }
    with open(output_dir / "model_info.json", "w") as f:
        json.dump(info, f, indent=2)
    logger.info(f"Saved to {output_dir}  ({info['n_params']:,} params)")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Main                                                                     ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def main():
    logger.info("InternalInspector (I²) Training  –  paper best config")
    logger.info(f"  state_combination : {args.state_combination}  "
                f"(channels={STATE_COMBINATION_CHANNELS[args.state_combination]})")
    logger.info(f"  lr={LR}  wd={WEIGHT_DECAY}  dropout={DROPOUT}  "
                f"temp={CONTRASTIVE_TEMP}  embed_dim={EMBED_DIM}")
    logger.info(f"  classifier hidden : {CLASSIFIER_HIDDEN}")
    logger.info(f"  BCE loss          : pos_weight=n_neg/n_pos")
    logger.info(f"  Labels            : 1 = correct, 0 = incorrect")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"  device : {device}")

    model_name_part = args.model_name.split("/")[-1]
    data_dir        = Path(args.data_dir)
    train_samples   = data_dir / model_name_part / args.train_dataset_name / "samples"
    val_samples     = data_dir / model_name_part / args.val_dataset_name   / "samples"
    if args.seed is not None:
        output_dir = Path(args.output_dir) / "best"
    else:
        output_dir = (Path(args.output_dir) / model_name_part
                      / args.train_dataset_name / args.state_combination / "best")

    for d in (train_samples, val_samples):
        if not d.exists():
            raise FileNotFoundError(f"Samples directory not found: {d}")

    # ── load data ──────────────────────────────────────────────────────────
    logger.info("\n── Training data ──")
    train_inputs, train_labels, _ = load_samples(train_samples, args.state_combination)

    logger.info("── Validation data ──")
    val_inputs, val_labels, _ = load_samples(val_samples, args.state_combination)

    _, num_layers, hidden_dim, in_channels = train_inputs.shape
    logger.info(f"\n  num_layers={num_layers}  hidden_dim={hidden_dim}  "
                f"in_channels={in_channels}")

    train_ds = IIDataset(train_inputs, train_labels, device="cuda")
    del train_inputs, train_labels
    torch.cuda.empty_cache()

    val_ds = IIDataset(val_inputs, val_labels, device="cpu")
    del val_inputs, val_labels

    # ── build model ────────────────────────────────────────────────────────
    model = InternalInspectorModel(in_channels=in_channels)
    logger.info(f"  Parameters: {count_params(model):,}")

    # ── pos_weight (SPARROW alignment) ─────────────────────────────────────
    n_pos = float((train_ds.labels == 1).sum().item())
    n_neg = float((train_ds.labels == 0).sum().item())
    pw = n_neg / n_pos if n_pos > 0 else 1.0
    pos_weight_tensor = torch.tensor([pw], device=device, dtype=torch.float)
    logger.info(f"  pos_weight = n_neg/n_pos = {n_neg:.0f}/{n_pos:.0f} = {pw:.3f}")

    # ── train ──────────────────────────────────────────────────────────────
    model, history = train(
        model=model, train_ds=train_ds, val_ds=val_ds,
        batch_size=args.batch_size, max_epochs=args.max_epochs,
        patience=args.patience, auroc_w=args.composite_auroc_weight,
        device=device, pos_weight=pos_weight_tensor,
    )

    # ── final val metrics ──────────────────────────────────────────────────
    val_metrics, _, _ = evaluate(model, val_ds, args.batch_size, device)

    # ── save ───────────────────────────────────────────────────────────────
    save_model(model, output_dir, val_metrics,
               args.state_combination, num_layers, hidden_dim, in_channels,
               history)

    logger.info(f"\n{'='*60}")
    logger.info(f"Done.")
    logger.info(f"  AUROC     : {val_metrics['auroc']:.4f}")
    logger.info(f"  ECE       : {val_metrics['ece']:.4f}")
    logger.info(f"  Best epoch: {history['best_epoch']}")
    logger.info(f"  Output    : {output_dir}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()


# Qwen/Qwen3-VL-8B-Instruct
# llava-hf/llava-v1.6-vicuna-13b-hf
# OpenGVLab/InternVL3_5-14B-HF
# google/gemma-3-27b-it
# deepseek-ai/deepseek-vl2

# ── Example – full method (paper best) ───────────────────────────────────────
# python II_train.py \
#   --data_dir           ../data/II_extraction_v2/ \
#   --model_name         Qwen/Qwen3-VL-8B-Instruct \
#   --train_dataset_name train \
#   --val_dataset_name   validation \
#   --output_dir         ../trained_models/II/ \
#   --state_combination  all \
#   --batch_size         32 \
#   --max_epochs         200 \
#   --patience           20 \
#   --cuda_devices       7

# ── Ablation examples ────────────────────────────────────────────────────────
# python II_train.py  ...  --state_combination ff        # FFN only
# python II_train.py  ...  --state_combination attention # attention only
# python II_train.py  ...  --state_combination attn_ff   # attention + FFN
# python II_train.py  ...  --state_combination ff_act    # FFN + activation
#
# Output layout: {output_dir}/{model_name}/{train_dataset}/{state_combination}/best/