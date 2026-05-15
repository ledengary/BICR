"""LLaVA-in-the-Wild curator — test only (60 rows).

Source: HuggingFace `liuhaotian/llava-bench-in-the-wild` raw files:
  - questions.jsonl     (one record per question with image filename, category, question_id, text)
  - answers_gpt4.jsonl  (GPT-4 reference answer keyed by question_id)
We download both via the HF resolve URL, merge on question_id, and emit one row
per question with the image filename as the join key.

hash_id inputs (matching the curation notebook used for VLMCE_v3):
  dataset   = "LLaVA-Wild"
  category  = row["category"]  (with N/A fallback)
  question  = row["question"]
  answer    = row["answer"]    (GPT-4 reference text)
  image_key = str(row["image"])  (the image FILENAME from the source jsonl)
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd
import requests
from datasets import Dataset
from PIL import Image
from tqdm import tqdm

from ._hash import md5_hash_id

BASE_URL = "https://huggingface.co/datasets/liuhaotian/llava-bench-in-the-wild/resolve/main"


def _download(url: str, save: Path) -> Path:
    save.parent.mkdir(parents=True, exist_ok=True)
    if save.exists():
        return save
    r = requests.get(url)
    r.raise_for_status()
    save.write_bytes(r.content)
    return save


def _load_jsonl(path: Path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def build_test(cache_dir: Path = Path("./.cache/llava_wild")) -> Dataset:
    cache_dir.mkdir(parents=True, exist_ok=True)
    q_path = _download(f"{BASE_URL}/questions.jsonl", cache_dir / "questions.jsonl")
    a_path = _download(f"{BASE_URL}/answers_gpt4.jsonl", cache_dir / "answers_gpt4.jsonl")
    df_q = pd.DataFrame(_load_jsonl(q_path))
    df_a = pd.DataFrame(_load_jsonl(a_path))
    df = pd.merge(df_q, df_a[["question_id", "text"]], on="question_id", how="inner")
    df = df.rename(columns={"text_x": "question", "text_y": "answer"})

    # Download images
    img_dir = cache_dir / "images"
    img_dir.mkdir(exist_ok=True)
    for img_name in df["image"].unique():
        _download(f"{BASE_URL}/images/{img_name}", img_dir / img_name)

    rows = []
    for _, r in tqdm(df.iterrows(), total=len(df), desc="LLaVA-Wild"):
        img = Image.open(img_dir / r["image"]).convert("RGB")
        category = r["category"]
        if pd.isna(category) or str(category).strip() == "":
            category = "N/A"
        rows.append({
            "question": str(r["question"]),
            "answer":   str(r["answer"]),
            "image":    img,
            "category": category,
            "dataset":  "LLaVA-Wild",
            "hash_id":  md5_hash_id("LLaVA-Wild", category,
                                    r["question"], r["answer"], str(r["image"])),
        })
    return Dataset.from_list(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cache", type=Path, default=Path("./.cache/llava_wild"))
    args = ap.parse_args()
    ds = build_test(args.cache)
    args.out.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(args.out))
    print(f"Wrote {len(ds)} rows to {args.out}")
