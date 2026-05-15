#!/usr/bin/env python3
"""
Inspect dry-run outputs for each null type:
- mask_blank rate (should be ≥ 99%)
- cos_base_blank distribution (should differ across null types)
- shape of h_blank (should match h_base)

Run after `orchestrate_extraction.py --dry-run`.
"""
import json
import numpy as np
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[3]  # repo root (data/ + results/ alongside)
EXTRACT_ROOT = ROOT / "data" / "null_ablation_extraction"
V3_REF_DIR = ROOT / "data" / "extraction/BICR" / "Qwen3-VL-8B-Instruct" / "test" / "samples"

VLM = "Qwen3-VL-8B-Instruct"
NULL_TYPES = ["white", "gaussian_noise", "blurred", "pixel_shuffled"]
SPLITS = ["train", "validation", "test"]


def stats_for_dir(samples_dir: Path):
    """Return dict of summary stats over npz files in samples_dir."""
    files = sorted(samples_dir.glob("*.npz"))
    if not files:
        return None
    n = len(files)
    n_blank_ok = 0
    cos_blank = []
    h_blank_shapes = set()
    h_base_shapes = set()
    for f in files:
        try:
            d = dict(np.load(f, allow_pickle=True))
        except Exception:
            continue
        mb = d.get("mask_blank", 0)
        if hasattr(mb, "item"): mb = mb.item()
        if int(mb) == 1:
            n_blank_ok += 1
        cb = d.get("cos_base_blank", float("nan"))
        if hasattr(cb, "item"): cb = cb.item()
        if not np.isnan(cb):
            cos_blank.append(float(cb))
        h_blank_shapes.add(tuple(d["h_blank"].shape))
        h_base_shapes.add(tuple(d["h_base"].shape))
    return {
        "n_files": n,
        "mask_blank_rate": n_blank_ok / max(n, 1),
        "cos_blank_mean": float(np.mean(cos_blank)) if cos_blank else float("nan"),
        "cos_blank_std":  float(np.std(cos_blank))  if cos_blank else float("nan"),
        "cos_blank_min":  float(np.min(cos_blank))  if cos_blank else float("nan"),
        "cos_blank_max":  float(np.max(cos_blank))  if cos_blank else float("nan"),
        "h_blank_shapes": h_blank_shapes,
        "h_base_shapes":  h_base_shapes,
    }


def main():
    print("Sanity-check report for null-ablation dry-run\n")

    # v3 black baseline (read-only) — for comparison
    print("Reading v3 baseline (black) cos_base_blank from a sample of test_raw npz...")
    if V3_REF_DIR.exists():
        files = sorted(V3_REF_DIR.glob("*.npz"))[:200]
        cos_v3 = []
        for f in files:
            try:
                d = dict(np.load(f, allow_pickle=True))
                cb = d.get("cos_base_blank", float("nan"))
                if hasattr(cb, "item"): cb = cb.item()
                if not np.isnan(cb):
                    cos_v3.append(float(cb))
            except Exception:
                continue
        if cos_v3:
            print(f"  v3/black cos_base_blank (n={len(cos_v3)}): "
                  f"mean={np.mean(cos_v3):.4f} std={np.std(cos_v3):.4f} "
                  f"min={np.min(cos_v3):.4f} max={np.max(cos_v3):.4f}")
        else:
            print("  v3/black: no cos_base_blank found")
    else:
        print(f"  v3 baseline dir not found at {V3_REF_DIR} — skipping reference")

    print("\nNew null variants:")
    bad = 0
    for null_type in NULL_TYPES:
        print(f"\n=== {null_type} ===")
        for split in SPLITS:
            d = EXTRACT_ROOT / null_type / VLM / split / "samples"
            stats = stats_for_dir(d)
            if stats is None:
                print(f"  {split:<22s}: NO npz files at {d}")
                bad += 1
                continue
            cb_str = (f"cos_base_blank: mean={stats['cos_blank_mean']:.4f} "
                      f"std={stats['cos_blank_std']:.4f} "
                      f"range=[{stats['cos_blank_min']:.4f}, {stats['cos_blank_max']:.4f}]")
            mb_rate = stats["mask_blank_rate"]
            mb_flag = "OK" if mb_rate >= 0.99 else "LOW"
            print(f"  {split:<22s}: n={stats['n_files']:<3d} "
                  f"mask_blank={mb_rate:.2%} ({mb_flag})  {cb_str}")
            if stats["h_blank_shapes"] != stats["h_base_shapes"]:
                print(f"    WARN: h_blank shapes {stats['h_blank_shapes']} != "
                      f"h_base shapes {stats['h_base_shapes']}")
                bad += 1
            if mb_rate < 0.99:
                bad += 1

    print(f"\n{'='*60}")
    if bad == 0:
        print("ALL CHECKS PASSED — safe to launch full extraction.")
    else:
        print(f"FAILED {bad} check(s). Inspect above before launching full run.")


if __name__ == "__main__":
    main()
