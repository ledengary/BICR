#!/usr/bin/env python3
"""
Aggregate null-ablation BICR results and compare against the existing
black baseline. Emits a markdown summary and a LaTeX table.

Inputs:
  - existing black baseline:
      results/SPARROW/BICR/Qwen3-VL-8B-Instruct/seed_*/test/test_results.json
  - new null variants:
      results/SPARROW/BICR_null_ablation/{null}/Qwen3-VL-8B-Instruct/seed_*/test/test_results.json

Outputs:
  - models/BICR/null_ablation/null_ablation_results.md
  - manuscript/tables/null_ablation.tex
"""
import json
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[3]  # repo root (results/ alongside)
VLM_SHORT = "Qwen3-VL-8B-Instruct"
SEEDS = [23, 42, 137, 2024, 3407]
NULL_TYPES_NEW = ["white", "gaussian_noise", "blurred", "pixel_shuffled"]
TEST_SPLIT = "test"
METRICS = ["auroc", "aucpr", "ece", "brier", "accuracy", "f1"]
METRIC_LABELS = {
    "auroc": "AUROC", "aucpr": "AUCPR", "ece": "ECE", "brier": "Brier",
    "accuracy": "Acc", "f1": "F1",
}
LOWER_IS_BETTER = {"ece", "brier"}

BASELINE_DIR = ROOT / "results" / "SPARROW" / "BICR" / VLM_SHORT
NULL_DIR     = ROOT / "results" / "SPARROW" / "BICR_null_ablation"
MD_OUT  = ROOT / "models" / "BICR" / "null_ablation" / "null_ablation_results.md"
TEX_OUT = ROOT / "manuscript" / "tables" / "null_ablation.tex"


def load_results(path: Path):
    if not path.exists():
        return None
    with open(path) as f:
        d = json.load(f)
    if "overall" in d:
        return d["overall"]
    return d


def collect(null_type: str | None):
    """Return dict {metric: list-of-5-seed-values} for the given null_type
    (None == existing black baseline)."""
    out = {m: [] for m in METRICS}
    for seed in SEEDS:
        if null_type is None:
            p = BASELINE_DIR / f"seed_{seed}" / TEST_SPLIT / "test_results.json"
        else:
            p = NULL_DIR / null_type / VLM_SHORT / f"seed_{seed}" / TEST_SPLIT / "test_results.json"
        d = load_results(p)
        if d is None:
            print(f"  MISSING: {p}")
            continue
        for m in METRICS:
            v = d.get(m)
            if v is not None:
                out[m].append(float(v))
    return out


def fmt(x):
    return f"{x:.4f}" if not np.isnan(x) else "—"


def fmt_pct(p):
    if p < 0.001: return r"$<$0.001$^{***}$"
    if p < 0.01:  return f"{p:.3f}$^{{**}}$"
    if p < 0.05:  return f"{p:.3f}$^{{*}}$"
    return f"{p:.3f}"


def main():
    # Collect all variants
    print("Loading existing black baseline (results/SPARROW/BICR/...)")
    black = collect(None)
    new = {}
    for nt in NULL_TYPES_NEW:
        print(f"Loading {nt}")
        new[nt] = collect(nt)

    # Build markdown
    md = []
    md.append("# BICR Null-Image Ablation — Qwen3-VL-8B-Instruct\n")
    md.append("Test split: `test`. Each cell = mean ± std across 5 seeds "
              "(23, 42, 137, 2024, 3407). p-values from paired Wilcoxon vs the black "
              "baseline (n = 5 paired by seed).\n")
    md.append("## Mean ± std per metric\n")
    md.append("| Null type | " + " | ".join(METRIC_LABELS[m] for m in METRICS) + " |")
    md.append("|---|" + "|".join(["---"] * len(METRICS)) + "|")

    def row(name, data):
        cells = []
        for m in METRICS:
            vals = np.array(data[m])
            if len(vals) == 0:
                cells.append("—")
            else:
                cells.append(f"{vals.mean():.4f} ± {vals.std(ddof=1):.4f}")
        return f"| **{name}** | " + " | ".join(cells) + " |"

    md.append(row("black (baseline)", black))
    for nt in NULL_TYPES_NEW:
        md.append(row(nt, new[nt]))

    md.append("\n## Paired Wilcoxon p-values (each null vs black)\n")
    md.append("| Null type vs black | " + " | ".join(METRIC_LABELS[m] for m in METRICS) + " |")
    md.append("|---|" + "|".join(["---"] * len(METRICS)) + "|")
    for nt in NULL_TYPES_NEW:
        cells = []
        for m in METRICS:
            a = np.array(black[m])
            b = np.array(new[nt][m])
            n = min(len(a), len(b))
            if n < 5:
                cells.append("—")
                continue
            try:
                _, p = wilcoxon(a[:n], b[:n])
                cells.append(fmt_pct(p))
            except Exception as e:
                cells.append(f"err({e})")
        md.append(f"| {nt} | " + " | ".join(cells) + " |")

    # Delta vs black
    md.append("\n## Δ vs black (mean of new − mean of black, lower-is-better metrics negated)\n")
    md.append("| Null type | " + " | ".join(METRIC_LABELS[m] for m in METRICS) + " |")
    md.append("|---|" + "|".join(["---"] * len(METRICS)) + "|")
    for nt in NULL_TYPES_NEW:
        cells = []
        for m in METRICS:
            a = np.array(black[m]); b = np.array(new[nt][m])
            if len(a) == 0 or len(b) == 0:
                cells.append("—"); continue
            d = b.mean() - a.mean()
            if m in LOWER_IS_BETTER:
                d = -d  # so positive = better than black
            sign = "+" if d > 0 else ""
            cells.append(f"{sign}{d:.4f}")
        md.append(f"| {nt} | " + " | ".join(cells) + " |")

    md_text = "\n".join(md) + "\n"
    MD_OUT.write_text(md_text)
    print(f"Wrote {MD_OUT}")
    print()
    print(md_text)

    # LaTeX table — drop-in for the appendix
    tex = []
    tex.append(r"% Auto-generated by models/BICR/null_ablation/compare_nulls.py")
    tex.append(r"\begin{table*}[t]")
    tex.append(r"\centering")
    tex.append(r"\small")
    tex.append(r"\setlength{\tabcolsep}{4pt}")
    tex.append(r"\caption{BICR null-image ablation on Qwen3-VL-8B-Instruct test split. Each cell is mean$\pm$std across 5 seeds. The bottom panel reports paired Wilcoxon p-values vs the black-image baseline (n=5).}")
    tex.append(r"\label{tab:null_ablation}")
    tex.append(r"\begin{tabular}{l " + "r " * len(METRICS) + "}")
    tex.append(r"\toprule")
    tex.append(r"\textbf{Null type} & " + " & ".join(rf"\textbf{{{METRIC_LABELS[m]}}}" for m in METRICS) + r" \\")
    tex.append(r"\midrule")

    def tex_row(name, data):
        cells = []
        for m in METRICS:
            vals = np.array(data[m])
            if len(vals) == 0:
                cells.append("---")
            else:
                cells.append(f"{vals.mean():.4f}\\,\\tiny{{$\\pm$\\,{vals.std(ddof=1):.4f}}}")
        return name + " & " + " & ".join(cells) + r" \\"

    tex.append(tex_row(r"\textit{black (baseline)}", black))
    for nt in NULL_TYPES_NEW:
        tex.append(tex_row(nt.replace("_", r"\_"), new[nt]))

    tex.append(r"\midrule")
    tex.append(r"\multicolumn{" + str(len(METRICS) + 1) + r"}{l}{\textit{Paired Wilcoxon p-value (vs black)}} \\")
    for nt in NULL_TYPES_NEW:
        cells = []
        for m in METRICS:
            a = np.array(black[m]); b = np.array(new[nt][m])
            n = min(len(a), len(b))
            if n < 5:
                cells.append("---"); continue
            try:
                _, p = wilcoxon(a[:n], b[:n])
                cells.append(fmt_pct(p))
            except Exception:
                cells.append("---")
        tex.append("\\quad vs " + nt.replace("_", r"\_") + " & " + " & ".join(cells) + r" \\")

    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}")
    tex.append(r"\end{table*}")

    TEX_OUT.parent.mkdir(parents=True, exist_ok=True)
    TEX_OUT.write_text("\n".join(tex) + "\n")
    print(f"Wrote {TEX_OUT}")


if __name__ == "__main__":
    main()
