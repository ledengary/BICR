"""POPE curator — test only (9,000 rows).

Source: HuggingFace `lmms-lab/POPE` test split.

hash_id inputs: ("POPE", category, question, answer, image_source).
"""
from __future__ import annotations

import argparse
from pathlib import Path

from datasets import Dataset, load_dataset
from tqdm import tqdm

from ._hash import md5_hash_id


def build_test(revision: str = "main") -> Dataset:
    raw = load_dataset("lmms-lab/POPE", split="test", revision=revision)
    rows = []
    for r in tqdm(raw, desc="POPE test"):
        rows.append({
            "question": r["question"],
            "answer":   r["answer"],
            "image":    r["image"],
            "category": r["category"],
            "dataset":  "POPE",
            "hash_id":  md5_hash_id("POPE", r["category"], r["question"],
                                    r["answer"], r["image_source"]),
        })
    return Dataset.from_list(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    ds = build_test()
    args.out.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(args.out))
    print(f"Wrote {len(ds)} rows to {args.out}")
