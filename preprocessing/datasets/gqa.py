"""GQA curator — train (20,000) / validation (5,001) / test (~12,578 testdev).

Source: HuggingFace `lmms-lab/GQA`. The dataset is split into two configs:
  - `{split}_balanced_instructions` — question, answer, imageId, types (a
    stringified dict whose `detailed` key is our stratification category)
  - `{split}_balanced_images`       — imageId, image (PIL)

Stratified subsetting matches the curation notebook used to build VLMCE_v3:
  - Drop rows with NaN `detailed`.
  - Drop classes with < 2 samples.
  - sklearn.model_selection.train_test_split with stratify=detailed,
    shuffle=True, random_state=42 (FIXED — different from SEED=23).
  - train_size = 20_000 (train), 5_001 (val).
  - Test uses all rows after filter (TEST_SUBSET_SIZE = 20_000 in notebook
    exceeds available testdev ~12,578).

hash_id inputs (matching the curation notebook exactly):
  dataset   = "GQA"
  category  = types["detailed"]  (with "N/A" fallback)
  question  = row.question
  answer    = row.answer
  image_key = str(row.imageId)
"""
from __future__ import annotations

import argparse
import ast
from pathlib import Path
from typing import Dict

import pandas as pd
from datasets import Dataset, load_dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from ._hash import md5_hash_id

RANDOM_STATE = 42
TRAIN_TARGET = 20_000
VAL_TARGET = 5_001
TEST_TARGET = 20_000  # exceeds testdev; effectively "all"


def _extract_detailed(types_field) -> str | None:
    if isinstance(types_field, dict):
        return types_field.get("detailed")
    try:
        d = ast.literal_eval(str(types_field))
    except (ValueError, SyntaxError):
        return None
    return d.get("detailed") if isinstance(d, dict) else None


def _stratified_subset(df: pd.DataFrame, size: int) -> pd.DataFrame:
    df2 = df.dropna(subset=["detailed"]).copy()
    counts = df2["detailed"].value_counts()
    valid = counts[counts >= 2].index
    df2 = df2[df2["detailed"].isin(valid)]
    if size >= len(df2):
        return df2.reset_index(drop=True)
    subset, _ = train_test_split(
        df2, train_size=size, stratify=df2["detailed"],
        shuffle=True, random_state=RANDOM_STATE,
    )
    return subset.reset_index(drop=True)


def _build_split(split: str, target: int, desc: str) -> Dataset:
    instr_config = f"{split}_balanced_instructions"
    imgs_config = f"{split}_balanced_images"
    raw = load_dataset("lmms-lab/GQA", instr_config, split=split)
    img_ds = load_dataset("lmms-lab/GQA", imgs_config, split=split)
    # Build imageId → row index for image lookup
    img_index: Dict[str, int] = {
        str(iid): i for i, iid in enumerate(img_ds["id"] if "id" in img_ds.column_names else img_ds["imageId"])
    }

    df = pd.DataFrame({
        "detailed": [_extract_detailed(t) for t in raw["types"]],
        "question": raw["question"],
        "answer":   raw["answer"],
        "imageId":  raw["imageId"],
    })
    df["_orig_idx"] = range(len(df))
    sub = _stratified_subset(df, target)
    rows = []
    for _, r in tqdm(sub.iterrows(), total=len(sub), desc=desc):
        idx = int(r["_orig_idx"])
        hf_row = raw[idx]
        category = _extract_detailed(hf_row.get("types"))
        if category is None or pd.isna(category) or str(category).strip() == "":
            category = "N/A"
        img_id = str(hf_row["imageId"])
        if img_id in img_index:
            img = img_ds[img_index[img_id]]["image"]
        else:
            continue
        rows.append({
            "question": str(hf_row["question"]),
            "answer":   str(hf_row["answer"]),
            "image":    img,
            "category": category,
            "dataset":  "GQA",
            "hash_id":  md5_hash_id("GQA", category, hf_row["question"],
                                    hf_row["answer"], img_id),
        })
    return Dataset.from_list(rows)


def build_train() -> Dataset:
    return _build_split("train", TRAIN_TARGET, "GQA train")


def build_validation() -> Dataset:
    return _build_split("val", VAL_TARGET, "GQA val")


def build_test() -> Dataset:
    return _build_split("testdev", TEST_TARGET, "GQA testdev")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "validation", "test"], required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    ds = {"train": build_train, "validation": build_validation, "test": build_test}[args.split]()
    args.out.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(args.out))
    print(f"Wrote {len(ds)} rows to {args.out}")
