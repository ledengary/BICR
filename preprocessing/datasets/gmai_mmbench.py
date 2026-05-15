"""GMAI-MMBench curator — test only (4,549 rows after answer-letter filtering).

Source: HuggingFace `OpenGVLab/GMAI-MMBench`, file `GMAI_mm_bench_VAL.tsv`.
The original notebook loads this specific TSV (streaming), in which the `image`
column is a base64 STRING (not a decoded PIL Image). The hash uses that raw
base64 string verbatim — so we must load the TSV directly to reproduce it.

License: CC BY-NC-SA — non-commercial research use only.

hash_id inputs (matching the curation notebook exactly):
  dataset   = "GMAI-MMBench"
  category  = row["clinical VQA task"]    (with N/A fallback)
  question  = formatted question (raw question + "\\nA) opt1 ... [\\nE) optE])
  answer    = answer letter (A-E), matched by exact/contains to row["category"]
  image_key = row["image"]   (raw base64 STRING from the TSV)
"""
from __future__ import annotations

import argparse
import base64
import io
from pathlib import Path

import pandas as pd
from datasets import Dataset, load_dataset
from PIL import Image
from tqdm import tqdm

from ._hash import md5_hash_id


def _format_question(question: str, a, b, c, d, e) -> str:
    formatted = str(question).strip()
    formatted += f"\nA) {a}"
    formatted += f"\nB) {b}"
    formatted += f"\nC) {c}"
    formatted += f"\nD) {d}"
    if e is not None and pd.notna(e) and str(e).strip().lower() != "none":
        formatted += f"\nE) {e}"
    return formatted


def _find_answer_letter(row) -> str | None:
    cat_text = str(row.get("category", "")).strip().lower()
    options = {
        "A": str(row["A"]).strip().lower() if pd.notna(row.get("A")) else "",
        "B": str(row["B"]).strip().lower() if pd.notna(row.get("B")) else "",
        "C": str(row["C"]).strip().lower() if pd.notna(row.get("C")) else "",
        "D": str(row["D"]).strip().lower() if pd.notna(row.get("D")) else "",
        "E": str(row["E"]).strip().lower() if (pd.notna(row.get("E")) and
                                                str(row.get("E", "")).strip().lower() != "none") else "",
    }
    for letter, text in options.items():
        if text and cat_text == text:
            return letter
    for letter, text in options.items():
        if text and (cat_text in text or text in cat_text):
            return letter
    return None


def _decode_b64_image(b64_str: str):
    return Image.open(io.BytesIO(base64.b64decode(b64_str))).convert("RGB")


def build_test() -> Dataset:
    # Stream the official VAL TSV (the only split with answers)
    ds_stream = load_dataset(
        "OpenGVLab/GMAI-MMBench",
        data_files="GMAI_mm_bench_VAL.tsv",
        streaming=True,
    )
    samples = list(ds_stream["train"])  # streaming + data_files puts it under "train"
    df = pd.DataFrame(samples)
    df["task_category"] = df["clinical VQA task"]

    rows = []
    for _, r in tqdm(df.iterrows(), total=len(df), desc="GMAI-MMBench"):
        answer_letter = _find_answer_letter(r)
        if answer_letter is None or pd.isna(answer_letter):
            continue
        formatted_q = _format_question(r["question"], r["A"], r["B"], r["C"], r["D"], r.get("E"))
        task_category = r.get("task_category", "N/A")
        if pd.isna(task_category) or str(task_category).strip() == "":
            task_category = "N/A"
        b64_str = r["image"]
        img = _decode_b64_image(b64_str)
        rows.append({
            "question": formatted_q,
            "answer":   answer_letter,
            "image":    img,
            "category": task_category,
            "dataset":  "GMAI-MMBench",
            "hash_id":  md5_hash_id("GMAI-MMBench", task_category,
                                    formatted_q, answer_letter, b64_str),
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
