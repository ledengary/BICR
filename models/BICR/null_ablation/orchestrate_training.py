#!/usr/bin/env python3
"""
Train BICR for each (null_type, seed) on Qwen3-VL-8B-Instruct using the
null-ablation extractions in data/null_ablation_extraction/{null_type}/...

Job space: 4 nulls × 5 seeds = 20 jobs over 8 GPUs (round-robin pool).
"""
import os
import sys
import time
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent.parent.parent
TRAINER = ROOT / "models" / "BICR" / "BICR_train.py"
LOG_DIR = ROOT / "logs" / "null_ablation" / "training"
LOG_DIR.mkdir(parents=True, exist_ok=True)

VLM = "Qwen/Qwen3-VL-8B-Instruct"
VLM_SHORT = "Qwen3-VL-8B-Instruct"
NULL_TYPES_ALL = ["white", "gaussian_noise", "blurred", "pixel_shuffled"]
SEEDS = [23, 42, 137, 2024, 3407]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 6, 7])
    p.add_argument("--null_types", type=str, nargs="+", default=NULL_TYPES_ALL,
                   choices=NULL_TYPES_ALL)
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    p.add_argument("--n_trials", type=int, default=50)
    p.add_argument("--data_root", type=str,
                   default=str(ROOT / "data" / "null_ablation_extraction"))
    p.add_argument("--checkpoint_root", type=str,
                   default=str(ROOT / "trained_models" / "SPARROW" / "BICR_null_ablation"))
    p.add_argument("--poll_seconds", type=int, default=30)
    p.add_argument("--dry-run", action="store_true",
                   help="Print job list without launching.")
    return p.parse_args()


def is_done(checkpoint_root: str, null_type: str, seed: int) -> bool:
    p = Path(checkpoint_root) / null_type / VLM_SHORT / f"seed_{seed}" / "best" / "model.pth"
    return p.exists()


def launch_job(null_type: str, seed: int, gpu: int, args):
    data_dir = Path(args.data_root) / null_type
    out_dir  = Path(args.checkpoint_root) / null_type / VLM_SHORT / f"seed_{seed}"
    cmd = [
        sys.executable, str(TRAINER),
        "--gpu", str(gpu),
        "--model-name", VLM,
        "--seed", str(seed),
        "--data-dir", str(data_dir),
        "--output-dir", str(out_dir),
        "--n-trials", str(args.n_trials),
    ]
    log_file = LOG_DIR / f"{null_type}_seed{seed}_gpu{gpu}.log"
    return cmd, log_file


def main():
    args = parse_args()
    gpus = args.gpus

    # Build job list (skip already-done)
    jobs = []
    skipped = 0
    for null_type in args.null_types:
        for seed in args.seeds:
            if is_done(args.checkpoint_root, null_type, seed):
                skipped += 1
                continue
            jobs.append({"null_type": null_type, "seed": seed})

    print(f"\n{'='*70}")
    print(f"Null-ablation BICR training orchestrator")
    print(f"{'='*70}")
    print(f"GPUs:        {gpus}")
    print(f"Null types:  {args.null_types}")
    print(f"Seeds:       {args.seeds}")
    print(f"n_trials:    {args.n_trials}")
    print(f"Total jobs:  {len(jobs)} (skipped {skipped} already-trained)")
    print(f"Data root:   {args.data_root}")
    print(f"Output root: {args.checkpoint_root}")
    print(f"{'='*70}\n")

    if not jobs:
        print("Nothing to do.")
        return

    if args.dry_run:
        for j in jobs:
            print(f"  WOULD RUN: null={j['null_type']:<16s} seed={j['seed']}")
        return

    active = {}
    job_idx = 0
    done, failed = [], []

    while job_idx < len(jobs) or active:
        for gpu in list(active.keys()):
            proc, job, log_path, t_start = active[gpu]
            rc = proc.poll()
            if rc is not None:
                elapsed = (datetime.now() - t_start).total_seconds()
                key = f"{job['null_type']}/seed{job['seed']}"
                if rc == 0:
                    print(f"  [GPU {gpu}] OK   {key} (t={elapsed:.0f}s)")
                    done.append(key)
                else:
                    print(f"  [GPU {gpu}] FAIL {key} rc={rc} (t={elapsed:.0f}s)  log={log_path.name}")
                    failed.append((key, rc))
                del active[gpu]

        for gpu in gpus:
            if gpu in active or job_idx >= len(jobs):
                continue
            job = jobs[job_idx]
            cmd, log_path = launch_job(job["null_type"], job["seed"], gpu, args)
            print(f"  [GPU {gpu}] LAUNCH {job['null_type']}/seed{job['seed']}  ({job_idx+1}/{len(jobs)})")
            f = open(log_path, "w")
            proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT,
                                    env={**os.environ, "PYTHONUNBUFFERED": "1"})
            active[gpu] = (proc, job, log_path, datetime.now())
            job_idx += 1

        if active:
            time.sleep(args.poll_seconds)

    print(f"\n{'='*70}")
    print(f"Done. {len(done)} succeeded, {len(failed)} failed.")
    if failed:
        for key, rc in failed:
            print(f"  {key}: rc={rc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
