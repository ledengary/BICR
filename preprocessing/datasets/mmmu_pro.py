"""MMMU-Pro curator — test only, two variants:
    MMMU_Pro_4   — 4-option configuration
    MMMU_Pro_10  — 10-option configuration

Source: HuggingFace `MMMU/MMMU_Pro` configs `standard (4 options)` and
`standard (10 options)`, test split.

VLMCE v2 had a bug where MMMU options weren't embedded in the question text,
so LVLMs were effectively answering blind. v3 (and this curator) appends the
options as `\\nA) opt1\\nB) opt2 ...` so the stored question is what should
actually be sent to the LVLM. The hash, however, is computed against the
RAW question (before options were appended) so it remains identical to the
v2/v3 published hash_ids.

Note: the upstream `MMMU/MMMU_Pro` dataset updated the option text for ~14
questions (≈ 0.4% of the 3,445 test rows) between when VLMCE_v3 was built
and the latest HF revision. The hash and the correct answer letter are
unchanged for those samples, but their option strings will read slightly
differently than in v3. Pinning to any of the listed HF revisions does
not recover the v3 text (the divergence predates HF's earliest revision),
so users should treat option-text identity as best-effort. The hashes and
labels — the part this benchmark joins on — are stable.

hash_id inputs (matching the curation notebook exactly):
  dataset   = "MMMU_Pro_{4|10}"
  category  = "{subject}[SEP]{topic_difficulty}[SEP]{img_type_str}" where
              img_type_str = "[LSEP]".join(img_type_list) or "N/A" if empty.
  question  = sample["question"]   (RAW — no options appended)
  answer    = sample["answer"]
  image_key = "image_1,image_2,..." (comma-joined non-None image_i fields,
              or "no_images" if no images present)

Stored question column:
  question_text + "\\nA) opt1\\nB) opt2 ... \\n{letter}) opt_n"
"""
from __future__ import annotations

import argparse
import ast
from pathlib import Path

import pandas as pd
from datasets import Dataset, load_dataset
from PIL import Image
from tqdm import tqdm

from ._hash import md5_hash_id

CONFIGS = {
    4:  "standard (4 options)",
    10: "standard (10 options)",
}


def _build_category(sample) -> str:
    subject = sample.get("subject", "N/A")
    if pd.isna(subject) or str(subject).strip() == "":
        subject = "N/A"
    topic_difficulty = sample.get("topic_difficulty", "N/A")
    if pd.isna(topic_difficulty) or str(topic_difficulty).strip() == "":
        topic_difficulty = "N/A"
    img_type = sample.get("img_type", [])
    if isinstance(img_type, str):
        try:
            img_type = ast.literal_eval(img_type)
        except (ValueError, SyntaxError):
            img_type = []
    if not img_type or (isinstance(img_type, list) and len(img_type) == 0):
        img_type_str = "N/A"
    elif isinstance(img_type, list):
        img_type_str = "[LSEP]".join(str(x) for x in img_type)
    else:
        img_type_str = str(img_type)
    return f"{subject}[SEP]{topic_difficulty}[SEP]{img_type_str}"


def _format_question_with_options(question_text: str, options) -> str:
    """Append the multiple-choice options as the LVLM prompt should see them.

    Layout: "{question}\\nA) opt1\\nB) opt2\\nC) opt3..." matching the format
    stored in VLMCE_v3.
    """
    if isinstance(options, str):
        try:
            options = ast.literal_eval(options)
        except (ValueError, SyntaxError):
            options = []
    if not isinstance(options, (list, tuple)) or len(options) == 0:
        return str(question_text)
    formatted = str(question_text)
    for idx, opt in enumerate(options):
        letter = chr(ord("A") + idx)
        formatted += f"\n{letter}) {opt}"
    return formatted


def build_test(n_options: int = 4, revision: str = "main") -> Dataset:
    config = CONFIGS[n_options]
    raw = load_dataset("MMMU/MMMU_Pro", config, split="test", revision=revision)
    dataset_name = f"MMMU_Pro_{n_options}"
    rows = []
    for sample in tqdm(raw, desc=dataset_name):
        images = []
        image_info_parts = []
        for i in range(1, 8):
            img_key = f"image_{i}"
            if img_key in sample and sample[img_key] is not None:
                img = sample[img_key]
                if not isinstance(img, Image.Image):
                    img = Image.fromarray(img)
                images.append(img.convert("RGB"))
                image_info_parts.append(img_key)
        primary_image = images[0] if images else None
        image_info = ",".join(image_info_parts) if image_info_parts else "no_images"

        category = _build_category(sample)
        raw_question = sample["question"]
        answer       = sample["answer"]
        formatted_question = _format_question_with_options(raw_question, sample.get("options"))

        rows.append({
            "question": formatted_question,
            "answer":   str(answer),
            "image":    primary_image,
            "category": category,
            "dataset":  dataset_name,
            "hash_id":  md5_hash_id(dataset_name, category, raw_question, answer, image_info),
        })
    return Dataset.from_list(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_options", type=int, choices=[4, 10], required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    ds = build_test(args.n_options)
    args.out.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(args.out))
    print(f"Wrote {len(ds)} rows to {args.out}")
