"""Reliability diagrams + bar charts that match the paper's calibration plots.

Reads per-(method, vlm, seed) `test_labels.json` files to recover the actual
confidence-vs-correctness pairs needed for binning.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from .aggregate import METHODS, RESULTS_ROOT, SEEDS, VLMS, TEST_SPLIT_DIR

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         10,
    "axes.titlesize":    11,
    "axes.labelsize":    10,
    "legend.fontsize":   8,
    "savefig.bbox":      "tight",
    "savefig.dpi":       180,
})


def _load_pairs(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Return (confidence, label_correct) arrays from a test_labels.json file.

    Transparently handles both `test_labels.json` and `test_labels.json.gz`
    so the bundled artefacts can ship gzipped to keep the repository light.
    """
    import gzip
    if path.suffix == ".gz":
        opener = lambda p: gzip.open(p, "rt")
    elif (gz := path.with_suffix(path.suffix + ".gz")).exists():
        path = gz
        opener = lambda p: gzip.open(p, "rt")
    else:
        opener = lambda p: open(p)
    with opener(path) as f:
        records = json.load(f)
    if isinstance(records, dict) and "samples" in records:
        records = records["samples"]
    confs, labels = [], []
    for r in records:
        c = r.get("confidence_score", r.get("confidence"))
        y = r.get("ground_truth_correctness", r.get("label"))
        if c is None or y is None:
            continue
        confs.append(float(c))
        labels.append(int(y))
    return np.asarray(confs, dtype=float), np.asarray(labels, dtype=int)


def _pooled_pairs(method_key: str) -> Tuple[np.ndarray, np.ndarray]:
    """All confidences/labels for a method, pooled across VLMs and seeds."""
    confs_all, labels_all = [], []
    for vlm in VLMS:
        seeded_dir = RESULTS_ROOT / method_key / vlm
        # seeded layout
        seeded_paths = [seeded_dir / f"seed_{s}" / TEST_SPLIT_DIR / "test_labels.json"
                        for s in SEEDS]
        used_any = False
        for p in seeded_paths:
            if p.exists():
                c, y = _load_pairs(p)
                confs_all.append(c)
                labels_all.append(y)
                used_any = True
        if not used_any:
            # inference-only: final/
            p = seeded_dir / "final" / "test_labels.json"
            if p.exists():
                c, y = _load_pairs(p)
                confs_all.append(c)
                labels_all.append(y)
    if not confs_all:
        return np.array([]), np.array([])
    return np.concatenate(confs_all), np.concatenate(labels_all)


def reliability_curve(conf: np.ndarray, labels: np.ndarray, n_bins: int = 10
                      ) -> Tuple[np.ndarray, np.ndarray]:
    """Return (bin_centers, empirical_accuracy_per_bin). NaN for empty bins."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    acc = np.full(n_bins, np.nan)
    for i in range(n_bins):
        in_bin = (conf >= edges[i]) & (conf < edges[i + 1] if i < n_bins - 1 else conf <= edges[i + 1])
        if in_bin.sum() > 0:
            acc[i] = float(labels[in_bin].mean())
    return centers, acc


def plot_cross_vlm_calibration(out: Path, n_bins: int = 10) -> Path:
    """Eight methods overlaid on a single reliability plot."""
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=0.8, label="Perfect")

    cmap = plt.get_cmap("tab10")
    for i, (key, display) in enumerate(METHODS):
        conf, labels = _pooled_pairs(key)
        if conf.size == 0:
            continue
        centers, acc = reliability_curve(conf, labels, n_bins=n_bins)
        ax.plot(centers, acc, marker="o", color=cmap(i % 10),
                label=display, linewidth=1.4, markersize=3)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted confidence")
    ax.set_ylabel("Empirical accuracy")
    ax.set_title("Reliability diagram (cross-VLM, 5 seeds)")
    ax.legend(loc="upper left", frameon=False)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out
