"""Load `results/SPARROW/{method}/{vlm}/seed_*/VLCB_test/test_results.json` and
return tidy dataframes for table builders and plotters.

Aggregation reproduces the paper's published cross-VLM averages:
  - For each (method, VLM, seed) we read metrics from test_results.json.
  - Per VLM we average those across seeds → "per-VLM mean".
  - Across the 5 VLMs we average those means → "cross-VLM average" (mean) and
    std (±std) per the paper's Table 2 caption.
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

RESULTS_ROOT = Path(__file__).resolve().parents[2] / "results" / "SPARROW"

METHODS = [
    ("PTRUE",        "P(True)"),
    ("SELF_PROBING", "Self-Probing"),
    ("PE",           "PE"),
    ("PIK",          "P(IK)"),
    ("SAPLMA",       "SAPLMA"),
    ("II",           "InternalInspector"),
    ("CCPS",         "CCPS"),
    ("BICR",         "BICR"),
]

VLMS = [
    "Qwen3-VL-8B-Instruct",
    "llava-v1.6-vicuna-13b-hf",
    "InternVL3_5-14B-HF",
    "gemma-3-27b-it",
    "deepseek-vl2",
]

SEEDS = [23, 42, 137, 2024, 3407]
METRICS = ["ece", "brier", "aucpr", "auroc", "accuracy", "f1",
           "precision", "recall", "sensitivity", "specificity"]
TEST_SPLIT_DIR = "VLCB_test"


def _open_maybe_gz(path: Path, mode: str):
    if path.suffix == ".gz":
        return gzip.open(path, mode)
    return open(path, mode)


def _read_metrics(p: Path) -> Optional[Dict[str, float]]:
    if not p.exists():
        return None
    try:
        with open(p) as f:
            obj = json.load(f)
    except json.JSONDecodeError:
        return None
    if "overall" in obj and isinstance(obj["overall"], dict):
        block = obj["overall"]
    elif "metrics" in obj and isinstance(obj["metrics"], dict):
        block = obj["metrics"]
    else:
        return None
    return {k: float(v) for k, v in block.items()
            if k in METRICS and isinstance(v, (int, float))}


def _seeded_results_path(method: str, vlm: str, seed: int) -> Path:
    return RESULTS_ROOT / method / vlm / f"seed_{seed}" / TEST_SPLIT_DIR / "test_results.json"


def _inference_results_path(method: str, vlm: str) -> Path:
    return RESULTS_ROOT / method / vlm / "final" / "test_metrics_and_results.json"


def load_main_results(results_root: Path = RESULTS_ROOT) -> pd.DataFrame:
    """Long-format frame: (method, vlm, seed, metric, value).

    Inference-only methods (PTRUE / SELF_PROBING / PE) have seed=NaN.
    """
    rows: List[Dict] = []
    for method, display in METHODS:
        for vlm in VLMS:
            seeded_files = [_seeded_results_path(method, vlm, s) for s in SEEDS]
            any_seeded = any(p.exists() for p in seeded_files)
            if any_seeded:
                for s in SEEDS:
                    m = _read_metrics(_seeded_results_path(method, vlm, s))
                    if m is None:
                        continue
                    for metric, value in m.items():
                        rows.append({
                            "method": display, "method_key": method,
                            "vlm": vlm, "seed": s,
                            "metric": metric, "value": value,
                        })
            else:
                m = _read_metrics(_inference_results_path(method, vlm))
                if m is None:
                    continue
                for metric, value in m.items():
                    rows.append({
                        "method": display, "method_key": method,
                        "vlm": vlm, "seed": None,
                        "metric": metric, "value": value,
                    })
    return pd.DataFrame(rows)


def cross_vlm_mean(df: pd.DataFrame, metrics: Optional[List[str]] = None) -> pd.DataFrame:
    """For each (method, metric): mean across seeds within VLM, then mean across VLMs.
    This is the paper's "cross-VLM average" column convention."""
    metrics = metrics or ["ece", "brier", "aucpr", "auroc"]
    df = df[df["metric"].isin(metrics)]
    per_vlm = df.groupby(["method", "method_key", "vlm", "metric"])["value"].mean().reset_index()
    cross = per_vlm.groupby(["method", "method_key", "metric"])["value"].mean().reset_index()
    table = cross.pivot(index=["method", "method_key"], columns="metric", values="value").reset_index()
    method_order = [m[1] for m in METHODS]
    table["__order"] = table["method"].apply(method_order.index)
    return table.sort_values("__order").drop(columns="__order").reset_index(drop=True)[
        ["method"] + metrics
    ]


def cross_vlm_std(df: pd.DataFrame, metrics: Optional[List[str]] = None) -> pd.DataFrame:
    """For each (method, metric): population std across the 5 per-VLM means."""
    metrics = metrics or ["ece", "brier", "aucpr", "auroc"]
    df = df[df["metric"].isin(metrics)]
    per_vlm = df.groupby(["method", "method_key", "vlm", "metric"])["value"].mean().reset_index()
    cross = per_vlm.groupby(["method", "method_key", "metric"])["value"].std(ddof=0).reset_index()
    table = cross.pivot(index=["method", "method_key"], columns="metric", values="value").reset_index()
    method_order = [m[1] for m in METHODS]
    table["__order"] = table["method"].apply(method_order.index)
    return table.sort_values("__order").drop(columns="__order").reset_index(drop=True)[
        ["method"] + metrics
    ]


def per_vlm_table(df: pd.DataFrame, metric: str = "auroc") -> pd.DataFrame:
    df = df[df["metric"] == metric]
    per_vlm = df.groupby(["method", "method_key", "vlm"])["value"].mean().reset_index()
    out = per_vlm.pivot(index=["method", "method_key"], columns="vlm", values="value").reset_index()
    method_order = [m[1] for m in METHODS]
    out["__order"] = out["method"].apply(method_order.index)
    return out.sort_values("__order").drop(columns="__order").reset_index(drop=True)


def load_ablation_results(results_root: Path = RESULTS_ROOT) -> pd.DataFrame:
    """BICR loss-component ablation under BICR_abl/{variant}/{vlm}/seed_*/VLCB_test/."""
    rows: List[Dict] = []
    base = results_root / "BICR_abl"
    if not base.exists():
        return pd.DataFrame()
    for variant_dir in base.iterdir():
        if not variant_dir.is_dir():
            continue
        variant = variant_dir.name
        for vlm in VLMS:
            for s in SEEDS:
                p = variant_dir / vlm / f"seed_{s}" / TEST_SPLIT_DIR / "test_results.json"
                m = _read_metrics(p)
                if m is None:
                    continue
                for metric, value in m.items():
                    rows.append({
                        "variant": variant, "vlm": vlm, "seed": s,
                        "metric": metric, "value": value,
                    })
    # Full variant from BICR/ proper
    for vlm in VLMS:
        for s in SEEDS:
            p = results_root / "BICR" / vlm / f"seed_{s}" / TEST_SPLIT_DIR / "test_results.json"
            m = _read_metrics(p)
            if m is None:
                continue
            for metric, value in m.items():
                rows.append({
                    "variant": "full", "vlm": vlm, "seed": s,
                    "metric": metric, "value": value,
                })
    return pd.DataFrame(rows)


def load_null_results(results_root: Path = RESULTS_ROOT) -> pd.DataFrame:
    """BICR_null_ablation: {null_type}/{vlm}/seed_*/VLCB_test/."""
    rows: List[Dict] = []
    base = results_root / "BICR_null_ablation"
    if not base.exists():
        return pd.DataFrame()
    for null_dir in base.iterdir():
        if not null_dir.is_dir():
            continue
        null = null_dir.name
        for vlm_dir in null_dir.iterdir():
            if not vlm_dir.is_dir():
                continue
            vlm = vlm_dir.name
            for seed_dir in vlm_dir.glob("seed_*"):
                p = seed_dir / TEST_SPLIT_DIR / "test_results.json"
                m = _read_metrics(p)
                if m is None:
                    continue
                seed = int(seed_dir.name.replace("seed_", ""))
                for metric, value in m.items():
                    rows.append({
                        "null_type": null, "vlm": vlm, "seed": seed,
                        "metric": metric, "value": value,
                    })
    for vlm in VLMS:
        for s in SEEDS:
            p = results_root / "BICR" / vlm / f"seed_{s}" / TEST_SPLIT_DIR / "test_results.json"
            m = _read_metrics(p)
            if m is None:
                continue
            for metric, value in m.items():
                rows.append({
                    "null_type": "black", "vlm": vlm, "seed": s,
                    "metric": metric, "value": value,
                })
    return pd.DataFrame(rows)
