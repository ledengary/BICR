#!/usr/bin/env python3
"""Reconstruct the VLCB item table locally from the original source distributors.

VLCB cannot be redistributed as a single archive because three of its seven
constituent datasets carry restrictive licenses. This script invokes each
per-source curator, then combines them into the canonical train / validation /
test splits and writes them as HuggingFace arrow shards under {data_root}/.

Output:
    {data_root}/train/        20,000 GQA train rows
    {data_root}/validation/    5,000 GQA val rows
    {data_root}/test/         30,514 rows across 7 sources
    {data_root}/expected_counts.json (copied from repo for offline verification)

After this finishes, run `python data/join_model_outputs.py` to download the
model-outputs table from `Ledengary/VLCB` and join it on hash_id.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from datasets import Dataset, concatenate_datasets

import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from preprocessing.datasets import (
    gqa,
    pope,
    gmai_mmbench,
    mmmu_pro,
    mme_finance,
    llava_in_the_wild,
)


def _save_split(ds: Dataset, out_dir: Path, split: str) -> None:
    split_dir = out_dir / split
    if split_dir.exists():
        shutil.rmtree(split_dir)
    ds.save_to_disk(str(split_dir))
    print(f"  ✓ {split}: {len(ds)} rows  →  {split_dir}")


def build_test(mme_finance_source: Path) -> Dataset:
    """Concatenate seven per-source test sets in deterministic order."""
    sources = [
        ("GQA",          gqa.build_test()),
        ("POPE",         pope.build_test()),
        ("GMAI-MMBench", gmai_mmbench.build_test()),
        ("MMMU_Pro_4",   mmmu_pro.build_test(n_options=4)),
        ("MMMU_Pro_10",  mmmu_pro.build_test(n_options=10)),
        ("MME-Finance",  mme_finance.build_test(mme_finance_source)),
        ("LLaVA-Wild",   llava_in_the_wild.build_test()),
    ]
    print("\nPer-source test counts:")
    for name, ds in sources:
        print(f"  {name:>14s}: {len(ds):>6,} rows")
    return concatenate_datasets([ds for _, ds in sources])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_root", type=Path, default=Path("./data/vlcb"),
                    help="Output directory (default: ./data/vlcb)")
    ap.add_argument("--mme_finance_source", type=Path, required=True,
                    help="Local directory containing MME-Finance.tsv + images/ "
                         "(download from github.com/MME-Benchmarks/MME-Finance).")
    ap.add_argument("--splits", nargs="+",
                    default=["train", "validation", "test"],
                    choices=["train", "validation", "test"])
    args = ap.parse_args()

    args.data_root.mkdir(parents=True, exist_ok=True)

    if "train" in args.splits:
        print("\nBuilding TRAIN (GQA, 20,000) …")
        _save_split(gqa.build_train(), args.data_root, "train")
    if "validation" in args.splits:
        print("\nBuilding VALIDATION (GQA, 5,000) …")
        _save_split(gqa.build_validation(), args.data_root, "validation")
    if "test" in args.splits:
        print("\nBuilding TEST (7-source union, 30,514) …")
        _save_split(build_test(args.mme_finance_source), args.data_root, "test")

    # Copy the expected_counts contract alongside the data
    src = ROOT / "data" / "expected_counts.json"
    if src.exists():
        shutil.copy2(src, args.data_root / "expected_counts.json")

    print("\nReconstruction complete. Next:")
    print("  python data/join_model_outputs.py "
          f"--data_root {args.data_root}")


if __name__ == "__main__":
    main()
