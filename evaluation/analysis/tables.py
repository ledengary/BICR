"""Emit LaTeX-friendly tables that mirror the paper's main + appendix tables.

All numbers come from the JSONs under `results/SPARROW/` via `aggregate.py`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from .aggregate import (
    METHODS,
    VLMS,
    cross_vlm_mean,
    load_ablation_results,
    load_main_results,
    load_null_results,
    per_vlm_table,
)


def _fmt(v: float, pct: bool = True) -> str:
    """Two-decimal % (matching paper) or four-decimal raw."""
    if pd.isna(v):
        return "—"
    return f"{v * 100:.2f}" if pct else f"{v:.4f}"


def build_main_table(out: Path) -> pd.DataFrame:
    df = load_main_results()
    table = cross_vlm_mean(df, metrics=["ece", "brier", "aucpr", "auroc"])
    rows = []
    for _, r in table.iterrows():
        rows.append([
            r["method"],
            _fmt(r["ece"]),
            _fmt(r["brier"]),
            _fmt(r["aucpr"]),
            _fmt(r["auroc"]),
        ])

    header = ["Method", "ECE ↓", "BS ↓", "AUCPR ↑", "AUROC ↑"]
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        f.write("\\begin{tabular}{l" + "r" * (len(header) - 1) + "}\n")
        f.write("\\toprule\n")
        f.write(" & ".join(header) + " \\\\\n")
        f.write("\\midrule\n")
        for row in rows:
            f.write(" & ".join(row) + " \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
    return table


def build_ablation_table(out: Path) -> pd.DataFrame:
    df = load_ablation_results()
    if df.empty:
        return df
    # Average across seeds within VLM, then across VLMs
    per_vlm = df.groupby(["variant", "vlm", "metric"])["value"].mean().reset_index()
    cross = per_vlm.groupby(["variant", "metric"])["value"].mean().reset_index()
    pivot = cross.pivot(index="variant", columns="metric", values="value")

    variant_order = ["full", "no_brier", "no_rank", "bce_only"]
    display_names = {
        "full":     "Full BICR",
        "no_brier": r"$-\mathcal{L}_{\mathrm{brier}}$",
        "no_rank":  r"$-\mathcal{L}_{\mathrm{rank}}$",
        "bce_only": r"$\mathcal{L}_{\mathrm{bce}}$ only",
    }

    out.parent.mkdir(parents=True, exist_ok=True)
    metric_order = ["ece", "brier", "aucpr", "auroc"]
    with open(out, "w") as f:
        f.write("\\begin{tabular}{l" + "r" * len(metric_order) + "}\n")
        f.write("\\toprule\n")
        f.write("Variant & " + " & ".join([
            "ECE ↓", "BS ↓", "AUCPR ↑", "AUROC ↑"]) + " \\\\\n")
        f.write("\\midrule\n")
        for v in variant_order:
            if v not in pivot.index:
                continue
            cells = [display_names[v]] + [_fmt(pivot.loc[v, m]) for m in metric_order]
            f.write(" & ".join(cells) + " \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
    return pivot


def build_null_table(out: Path, vlm: str = "Qwen3-VL-8B-Instruct") -> pd.DataFrame:
    df = load_null_results()
    if df.empty:
        return df
    sub = df[df["vlm"] == vlm]
    per_seed = sub.groupby(["null_type", "metric"])["value"].mean().reset_index()
    pivot = per_seed.pivot(index="null_type", columns="metric", values="value")

    null_order = ["black", "white", "gaussian_noise", "blurred", "pixel_shuffled"]
    display_names = {
        "black":          "Black (baseline)",
        "white":          "White",
        "gaussian_noise": "Gaussian noise",
        "blurred":        "Blurred",
        "pixel_shuffled": "Pixel-shuffled",
    }
    metric_order = ["auroc", "aucpr", "ece", "brier", "accuracy", "f1"]
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        f.write("\\begin{tabular}{l" + "r" * len(metric_order) + "}\n")
        f.write("\\toprule\n")
        f.write("Null type & " + " & ".join(
            ["AUROC ↑", "AUCPR ↑", "ECE ↓", "BS ↓", "Acc ↑", "F1 ↑"]) + " \\\\\n")
        f.write("\\midrule\n")
        for v in null_order:
            if v not in pivot.index:
                continue
            cells = [display_names[v]] + [_fmt(pivot.loc[v, m]) for m in metric_order]
            f.write(" & ".join(cells) + " \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
    return pivot


def build_per_vlm_table(out: Path, metric: str = "auroc") -> pd.DataFrame:
    df = load_main_results()
    table = per_vlm_table(df, metric=metric)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        f.write("\\begin{tabular}{l" + "r" * len(VLMS) + "}\n")
        f.write("\\toprule\n")
        f.write("Method & " + " & ".join(VLMS) + " \\\\\n")
        f.write("\\midrule\n")
        for _, r in table.iterrows():
            cells = [r["method"]] + [_fmt(r[v]) for v in VLMS]
            f.write(" & ".join(cells) + " \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
    return table
