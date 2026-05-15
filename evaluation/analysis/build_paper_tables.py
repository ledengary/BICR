#!/usr/bin/env python3
"""Regenerate every paper-reported table and figure from results/SPARROW/.

Outputs are written under repo/manuscript/tables/ and repo/manuscript/figures/.
"""
from __future__ import annotations

from pathlib import Path

from .plots import plot_cross_vlm_calibration
from .tables import (
    build_ablation_table,
    build_main_table,
    build_null_table,
    build_per_vlm_table,
)

ROOT = Path(__file__).resolve().parents[2]
TABLES = ROOT / "docs" / "tables"
FIGURES = ROOT / "docs" / "figures_generated"
TABLES.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)


def main() -> None:
    print("Building main results table (Table 1) …")
    build_main_table(TABLES / "main_results.tex")

    print("Building loss-component ablation table (Table 2) …")
    build_ablation_table(TABLES / "ablation.tex")

    print("Building null-image ablation table (Table 4) …")
    build_null_table(TABLES / "null_ablation.tex")

    print("Building per-VLM AUROC table (appendix) …")
    build_per_vlm_table(TABLES / "per_vlm_auroc.tex", metric="auroc")
    build_per_vlm_table(TABLES / "per_vlm_ece.tex", metric="ece")

    print("Plotting cross-VLM reliability diagram (Figure 1) …")
    plot_cross_vlm_calibration(FIGURES / "calibration_cross_vlm.pdf")

    print(f"\nAll outputs written under:")
    print(f"  Tables : {TABLES}")
    print(f"  Figures: {FIGURES}")


if __name__ == "__main__":
    main()
