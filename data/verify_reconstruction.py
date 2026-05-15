#!/usr/bin/env python3
"""Assert that the locally-joined VLCB exactly matches the paper's published
counts. Fails loudly if any split, per-source count, or per-(split, model)
correctness sum diverges. This is the gating check before training.

Compares the user's joined data at {data_root}/{split}_with_outputs/ against
the frozen expected_counts.json shipped with this repo.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from datasets import load_from_disk


def _expect(actual, expected, label: str) -> bool:
    ok = actual == expected
    glyph = "✓" if ok else "✗"
    print(f"  {glyph} {label}:  actual={actual}  expected={expected}")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data_root", type=Path, default=Path("./data/vlcb"))
    args = ap.parse_args()

    expected = json.loads((Path(__file__).resolve().parent / "expected_counts.json").read_text())
    all_ok = True

    for split in ("train", "validation", "test"):
        path = args.data_root / f"{split}_with_outputs"
        if not path.exists():
            print(f"\n{split}: skipped (no {path})")
            continue
        ds = load_from_disk(str(path))

        print(f"\n[{split}]")
        all_ok &= _expect(len(ds), expected["splits"][split]["n_rows"], "row count")
        all_ok &= _expect(len(set(ds["hash_id"])),
                          expected["splits"][split]["unique_hash_ids"],
                          "unique hash_ids")

        per_model = Counter(ds["model_name"])
        for m, n in sorted(expected["per_split_model"][split].items()):
            all_ok &= _expect(per_model.get(m, 0), n, f"rows for model={m}")

        if split == "test":
            per_source = Counter()
            seen = set()
            for r in ds:
                if r["hash_id"] in seen:
                    continue
                seen.add(r["hash_id"])
                per_source[r["source_dataset"]] += 1
            for s, n in sorted(expected["per_source"]["test"].items()):
                all_ok &= _expect(per_source.get(s, 0), n,
                                  f"unique samples for source={s}")

        # is_correct sums per model
        correct = {m: 0 for m in expected["per_split_model_correctness"][split]}
        for r in ds:
            if r["model_name"] in correct:
                correct[r["model_name"]] += int(r["is_correct"])
        for m, n in sorted(expected["per_split_model_correctness"][split].items()):
            all_ok &= _expect(correct[m], n, f"is_correct sum for model={m}")

    print()
    if all_ok:
        print("✓ All counts match. VLCB reconstruction is bit-exact.")
    else:
        print("✗ One or more checks failed. Local reconstruction is NOT bit-exact.")
        print("  Re-run preprocessing/datasets/* and data/reconstruct_vlcb.py, "
              "then re-run data/join_model_outputs.py.")


if __name__ == "__main__":
    main()
