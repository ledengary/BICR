#!/usr/bin/env python3
"""
Orchestrate null-ablation extraction across 8 GPUs for Qwen3-VL-8B-Instruct.

Job space: (null_type, shard_idx) for null in 4 NULLS, shard in {0..7}.
That's 32 jobs; 8 run concurrently (one per GPU). When a GPU's job exits,
the next pending job is launched on it.

Each job extracts shard `shard_idx` of train + val + test for one null_type.
The extractor is idempotent (--skip-if-processed), so re-runs are safe.

Usage
-----
  # Sanity: 100-sample dry run for every null
  python models/BICR/null_ablation/orchestrate_extraction.py --dry-run

  # Full run, all 8 GPUs
  python models/BICR/null_ablation/orchestrate_extraction.py

  # Subset (one null only)
  python models/BICR/null_ablation/orchestrate_extraction.py --null_types blurred

  # Specific GPUs
  python models/BICR/null_ablation/orchestrate_extraction.py --gpus 0 1 2 3
"""
import os
import sys
import time
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent.parent.parent  # repo root (data/ + results/ alongside)
EXTRACTOR = ROOT / "models" / "BICR" / "null_ablation" / "null_extraction.py"
LOG_DIR = ROOT / "logs" / "null_ablation" / "extraction"
LOG_DIR.mkdir(parents=True, exist_ok=True)

VLM = "Qwen/Qwen3-VL-8B-Instruct"
VLM_SHORT = "Qwen3-VL-8B-Instruct"
NULL_TYPES_ALL = ["white", "gaussian_noise", "blurred", "pixel_shuffled"]
SPLITS = ["train", "validation", "test"]

# Approximate dataset sizes (for shard math). Real datasets may be smaller —
# end_at_idx is min'd against len(dataset) inside the extractor, so over-spec is fine.
SPLIT_SIZES = {
    "train": 20_000,
    "validation":    5_000,
    "test":  35_000,  # actual is ~30.5K; over-spec is harmless
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 6, 7])
    p.add_argument("--null_types", type=str, nargs="+", default=NULL_TYPES_ALL,
                   choices=NULL_TYPES_ALL)
    p.add_argument("--n_shards", type=int, default=8,
                   help="Number of index-shards per (null, split). Default = number of GPUs.")
    p.add_argument("--dry-run", action="store_true",
                   help="100-sample dry run per null type instead of full extraction.")
    p.add_argument("--data_dir", type=str, default=str(ROOT / "data" / "VLCB" / "raw"))
    p.add_argument("--gen_extraction_dir", type=str,
                   default=str(ROOT / "data" / "extraction/raw"))
    p.add_argument("--pe_dir", type=str, default=str(ROOT / "data" / "PE"))
    p.add_argument("--output_dir", type=str,
                   default=str(ROOT / "data" / "null_ablation_extraction"),
                   help="Parent dir; null_type subdir is appended automatically by the extractor.")
    p.add_argument("--poll_seconds", type=int, default=30)
    p.add_argument("--no-skip", action="store_true",
                   help="Disable --skip-if-processed (re-extract all samples).")
    return p.parse_args()


def shard_range(total: int, shard_idx: int, n_shards: int):
    """Inclusive-start, exclusive-end index range for shard `shard_idx`."""
    start = (total * shard_idx) // n_shards
    end   = (total * (shard_idx + 1)) // n_shards
    return start, end


def build_jobs(args):
    """Return list of dicts describing each job."""
    jobs = []
    for null_type in args.null_types:
        for shard_idx in range(args.n_shards):
            jobs.append({
                "null_type":  null_type,
                "shard_idx":  shard_idx,
            })
    return jobs


def is_done_for_shard(args, null_type: str, shard_idx: int) -> bool:
    """
    Lightweight finished-check: a shard is considered done if every split's
    samples/ dir contains at least the expected number of npz files for the
    shard's index range. This avoids re-launching jobs that already finished.
    """
    n = args.n_shards
    for split in SPLITS:
        out = (Path(args.output_dir) / null_type / VLM_SHORT / split / "samples")
        if not out.exists():
            return False
        s, e = shard_range(SPLIT_SIZES[split], shard_idx, n)
        # We can't cheaply tell which hash_ids belong to this shard without
        # loading the dataset; instead, we conservatively check that the total
        # count matches (or exceeds, since other shards write to the same dir).
        # In practice: only treat the WHOLE null as done when total count for
        # the split matches the expected total. This per-shard check returns
        # False; we'll let --skip-if-processed handle redundancy at the
        # sample level inside the extractor.
        return False
    return True  # unreachable


def launch_job(job, gpu: int, args):
    null_type = job["null_type"]
    shard_idx = job["shard_idx"]
    n = args.n_shards

    cmd = [
        sys.executable, str(EXTRACTOR),
        "--null_type", null_type,
        "--model_id", VLM,
        "--gpu_ids", str(gpu),
        "--dtype", "float32",
        "--dataset_path", args.data_dir,
        "--target_datasets", *SPLITS,
        "--train_dataset", "train",
        "--generation_extraction_dir", args.gen_extraction_dir,
        "--pe_dir", args.pe_dir,
        "--output_dir", args.output_dir,
        "--layer_offsets", "0",
    ]
    if not args.no_skip:
        cmd.append("--skip-if-processed")

    if args.dry_run:
        cmd += ["--max_samples", "100"]
    else:
        # Per-split start/end is set via single --start_at_idx/--end_at_idx
        # which the extractor applies to ALL --target_datasets. Different
        # splits have different sizes, so we pass the SHARD INDEX directly
        # via two flags computed for each split? — the extractor currently
        # uses a single (start, end). To keep it simple, we shard by INDEX
        # FRACTION: for each shard launch we pass start/end appropriate to
        # the largest split (test) and let smaller splits get sliced
        # naturally (out-of-range -> empty).
        #
        # BUT the extractor applies start/end to EACH split. So if we pass
        # start=0, end=30000 for shard 0, train will get [0, min(30000,20000)]
        # = [0, 20000] which is ALL of train, not 1/8 of it. That's wrong.
        #
        # Instead: launch one extractor invocation per (split, shard) so
        # shard math is correct per-split. We trade the model-load-once
        # benefit for correctness.
        pass

    log_file = LOG_DIR / f"{null_type}_shard{shard_idx}_gpu{gpu}.log"
    return cmd, log_file


def launch_split_job(null_type, split, shard_idx, gpu, args):
    """Launch the extractor for a single (null, split, shard, gpu) combo."""
    n = args.n_shards
    total = SPLIT_SIZES[split]
    start, end = shard_range(total, shard_idx, n)

    cmd = [
        sys.executable, str(EXTRACTOR),
        "--null_type", null_type,
        "--model_id", VLM,
        "--gpu_ids", str(gpu),
        "--dtype", "float32",
        "--dataset_path", args.data_dir,
        "--target_datasets", split,
        "--train_dataset", "train",
        "--generation_extraction_dir", args.gen_extraction_dir,
        "--pe_dir", args.pe_dir,
        "--output_dir", args.output_dir,
        "--layer_offsets", "0",
        "--start_at_idx", str(start),
        "--end_at_idx", str(end),
    ]
    if not args.no_skip:
        cmd.append("--skip-if-processed")
    if args.dry_run:
        cmd += ["--max_samples", "100"]

    log_file = LOG_DIR / f"{null_type}_{split}_shard{shard_idx}_gpu{gpu}.log"
    return cmd, log_file


def main():
    args = parse_args()
    null_types = args.null_types
    gpus = args.gpus
    n_shards = args.n_shards

    # Job queue: one entry per (null_type, split, shard_idx).
    # Each one is a separate extractor invocation so the index range is
    # correct per split (train/val/test have different sizes).
    queue = []
    for null_type in null_types:
        for shard_idx in range(n_shards):
            for split in SPLITS:
                queue.append({
                    "null_type": null_type,
                    "split":     split,
                    "shard_idx": shard_idx,
                })

    print(f"\n{'='*70}")
    print(f"Null-ablation extraction orchestrator")
    print(f"{'='*70}")
    print(f"GPUs:         {gpus}")
    print(f"Null types:   {null_types}")
    print(f"Shards:       {n_shards}")
    print(f"Splits:       {SPLITS}")
    print(f"Total jobs:   {len(queue)}  (= {len(null_types)} nulls × {n_shards} shards × {len(SPLITS)} splits)")
    print(f"Dry run:      {args.dry_run}")
    print(f"Output:       {args.output_dir}")
    print(f"{'='*70}\n")

    active = {}      # gpu -> (proc, job_dict, log_path, start_time)
    job_idx = 0
    done = []
    failed = []

    while job_idx < len(queue) or active:
        # Reap finished jobs
        for gpu in list(active.keys()):
            proc, job, log_path, t_start = active[gpu]
            rc = proc.poll()
            if rc is not None:
                elapsed = (datetime.now() - t_start).total_seconds()
                key = f"{job['null_type']}/{job['split']}/sh{job['shard_idx']}"
                if rc == 0:
                    print(f"  [GPU {gpu}] OK   {key} (t={elapsed:.0f}s)  log={log_path.name}")
                    done.append(key)
                else:
                    print(f"  [GPU {gpu}] FAIL {key} rc={rc} (t={elapsed:.0f}s)  log={log_path.name}")
                    failed.append((key, rc))
                del active[gpu]

        # Launch on free GPUs
        for gpu in gpus:
            if gpu in active:
                continue
            if job_idx >= len(queue):
                continue
            job = queue[job_idx]
            cmd, log_path = launch_split_job(
                job["null_type"], job["split"], job["shard_idx"], gpu, args)
            print(f"  [GPU {gpu}] LAUNCH {job['null_type']}/{job['split']}/sh{job['shard_idx']}  "
                  f"({job_idx+1}/{len(queue)})")
            f = open(log_path, "w")
            proc = subprocess.Popen(
                cmd, stdout=f, stderr=subprocess.STDOUT,
                env={**os.environ, "PYTHONUNBUFFERED": "1"})
            active[gpu] = (proc, job, log_path, datetime.now())
            job_idx += 1

        if active:
            time.sleep(args.poll_seconds)

    print(f"\n{'='*70}")
    print(f"Done. {len(done)} succeeded, {len(failed)} failed.")
    if failed:
        print(f"Failures:")
        for key, rc in failed:
            print(f"  {key}: rc={rc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
