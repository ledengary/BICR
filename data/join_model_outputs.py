#!/usr/bin/env python3
"""Download Ledengary/VLCB (model outputs + correctness labels) and INNER-JOIN
it onto the user's locally-reconstructed item table on hash_id.

The result is written as HF arrow shards at {data_root}/{split}_with_outputs/,
ready to be consumed by the extraction and training scripts.

Refuses to proceed if any split's join row count differs from the expected
contract in expected_counts.json — a single hash mismatch indicates that the
local reconstruction drifted from the canonical curation and must be re-run.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from datasets import Dataset, load_dataset, load_from_disk
from tqdm import tqdm


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data_root", type=Path, default=Path("./data/vlcb"),
                    help="Directory containing train/, validation/, test/ from "
                         "reconstruct_vlcb.py.")
    ap.add_argument("--hf_repo", type=str, default="Ledengary/VLCB",
                    help="HF dataset repo id for model outputs.")
    args = ap.parse_args()

    expected_path = Path(__file__).resolve().parent / "expected_counts.json"
    expected = json.loads(expected_path.read_text())

    print(f"Loading model outputs from {args.hf_repo} …")
    outputs = load_dataset(args.hf_repo)

    for split in ("train", "validation", "test"):
        local_dir = args.data_root / split
        if not local_dir.exists():
            print(f"  SKIP {split}: {local_dir} missing")
            continue

        print(f"\nJoining {split} …")
        items = load_from_disk(str(local_dir))
        item_hashes = set(items["hash_id"])
        out_hashes = set(outputs[split]["hash_id"])

        common = item_hashes & out_hashes
        missing_in_outputs = item_hashes - out_hashes
        missing_in_items   = out_hashes - item_hashes
        if missing_in_items:
            print(f"  ! {len(missing_in_items):,} hash_ids in outputs not found in local items "
                  "(extra outputs — drop).")
        if missing_in_outputs:
            print(f"  ! {len(missing_in_outputs):,} hash_ids in local items not in outputs.")
            print(f"    First 5: {list(missing_in_outputs)[:5]}")

        item_lut = {r["hash_id"]: r for r in items}
        merged_rows = []
        for r in tqdm(outputs[split], desc=f"  merging {split}"):
            it = item_lut.get(r["hash_id"])
            if it is None:
                continue
            merged_rows.append({
                **it,
                "model_name":     r["model_name"],
                "model_response": r["model_response"],
                "is_correct":     int(r["is_correct"]),
            })

        n_rows_expected = expected["splits"][split]["n_rows"]
        if len(merged_rows) != n_rows_expected:
            print(f"  ✗ {split} merged rows {len(merged_rows):,} != expected {n_rows_expected:,}.")
            print("    Local item reconstruction does not match the canonical curation. "
                  "Re-run preprocessing/datasets/* and reconstruct_vlcb.py.")
            sys.exit(1)

        merged = Dataset.from_list(merged_rows)
        out_dir = args.data_root / f"{split}_with_outputs"
        merged.save_to_disk(str(out_dir))
        print(f"  ✓ {split}: {len(merged):,} rows  →  {out_dir}")

    print("\nDone. Next: python data/verify_reconstruction.py")


if __name__ == "__main__":
    main()
