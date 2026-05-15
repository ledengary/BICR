#!/bin/bash
# Full pipeline: wait for Phase A (extraction) to finish, then run Phase B
# (training), then Phase C (eval), then Phase D (compare).
#
# Idempotent — uses --skip-if-processed everywhere; safe to re-run.

set -e
# Run from the repo root (the directory containing models/, data/, results/).
REPO_ROOT="${VLCB_ROOT:-$(cd "$(dirname "$0")/../../.." && pwd)}"
cd "$REPO_ROOT"
# Activate your vLLM-capable conda env (override with VLCB_ENV env var if needed).
ENV_NAME="${VLCB_ENV:-vlcb_vllm}"
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

LOG=/tmp/null_pipeline.log
echo "[pipeline] starting at $(date)" | tee "$LOG"

# Phase A — extraction (assumed already running; we just block until it exits)
echo "[pipeline] waiting for orchestrate_extraction.py to finish..." | tee -a "$LOG"
while pgrep -f "orchestrate_extraction.py" > /dev/null; do
    sleep 60
done
echo "[pipeline] Phase A finished at $(date)" | tee -a "$LOG"

# Phase B — training
echo "[pipeline] launching Phase B (training)" | tee -a "$LOG"
python models/BICR/null_ablation/orchestrate_training.py \
    --gpus 0 1 2 3 4 5 6 7 2>&1 | tee -a "$LOG"
echo "[pipeline] Phase B finished at $(date)" | tee -a "$LOG"

# Phase C — eval
echo "[pipeline] launching Phase C (eval)" | tee -a "$LOG"
python models/BICR/null_ablation/orchestrate_eval.py \
    --gpus 0 1 2 3 4 5 6 7 2>&1 | tee -a "$LOG"
echo "[pipeline] Phase C finished at $(date)" | tee -a "$LOG"

# Phase D — comparison
echo "[pipeline] launching Phase D (compare)" | tee -a "$LOG"
python models/BICR/null_ablation/compare_nulls.py 2>&1 | tee -a "$LOG"
echo "[pipeline] Phase D finished at $(date)" | tee -a "$LOG"

echo "[pipeline] ALL PHASES DONE at $(date)" | tee -a "$LOG"
