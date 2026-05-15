"""MME-Finance curator — test only (892 rows after category filtering).

Source: Official release at https://github.com/MME-Benchmarks/MME-Finance.
There is no HuggingFace mirror, so users must download the archive themselves.
Pass the local extraction path via --source.

Expected source layout (matches the directory the original curation script used):
    {source}/MMfin.tsv             # one row per sample, with columns
                                   #   task_category, question, answer, image_path
    {source}/images/MMfin/<file>   # referenced images, named per `image_path`

Six task categories are kept (the remaining ones were excluded in the paper's
curation, leaving exactly 892 samples):
    Accurate Numerical Calculation
    Numerical Calculation
    Spatial Awareness
    Entity Recognition
    OCR
    Financial Knowledge

hash_id inputs: ("MME-Finance", task_category, question, answer, image_path),
where image_path is the raw value from the TSV (not the resolved absolute path).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from datasets import Dataset
from PIL import Image
from tqdm import tqdm

from ._hash import md5_hash_id

SELECTED_CATEGORIES = [
    "Accurate Numerical Calculation",
    "Numerical Calculation",
    "Spatial Awareness",
    "Entity Recognition",
    "OCR",
    "Financial Knowledge",
]


def build_test(source: Path) -> Dataset:
    tsv = source / "MMfin.tsv"
    if not tsv.exists():
        raise FileNotFoundError(
            f"MMfin.tsv not found at {tsv}. Download the dataset from "
            "https://github.com/MME-Benchmarks/MME-Finance and pass the "
            "extraction directory via --source.")
    img_root = source / "images" / "MMfin"
    df = pd.read_csv(tsv, sep="\t")
    df = df[df["task_category"].isin(SELECTED_CATEGORIES)].copy()
    rows = []
    for _, r in tqdm(df.iterrows(), total=len(df), desc="MME-Finance"):
        image_path = r["image_path"]
        full_img = img_root / image_path
        if not full_img.exists():
            raise FileNotFoundError(f"Image not found: {full_img}")
        with Image.open(full_img) as im:
            img = im.convert("RGB").copy()
        q = str(r["question"])
        a = str(r["answer"])
        cat = r["task_category"]
        if pd.isna(cat) or str(cat).strip() == "":
            cat = "N/A"
        rows.append({
            "question": q,
            "answer":   a,
            "image":    img,
            "category": cat,
            "dataset":  "MME-Finance",
            "hash_id":  md5_hash_id("MME-Finance", cat, q, a, image_path),
        })
    return Dataset.from_list(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, required=True,
                    help="Local directory containing MMfin.tsv and images/MMfin/")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    ds = build_test(args.source)
    args.out.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(args.out))
    print(f"Wrote {len(ds)} rows to {args.out}")
