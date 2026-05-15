# BICR Null-Image Ablation

This experiment swaps BICR's solid-black blank null view for four alternatives
and re-trains the method end-to-end on Qwen3-VL-8B-Instruct, isolating what the
null is actually doing.

## Null types

| Null | What it isolates |
|---|---|
| `white` | Luminance vs uniformity (vs black: same flatness, opposite luminance) |
| `gaussian_noise` | Information content vs image-likeness (high entropy, no structure) |
| `blurred` | High-frequency detail vs total absence (low-freq preserved) |
| `pixel_shuffled` | Spatial structure vs color statistics (pixels permuted) |

The existing **black** baseline at `data/extraction/BICR/...` and trained models
at `trained_models/SPARROW/BICR/...` are the comparison reference and are NEVER
modified by this experiment.

## Files

| File | Purpose |
|---|---|
| `null_extraction.py` | Clone of `models/extraction/BICR.py`; only the blank view's image generator is swapped. `--null_type` selects which of the 4 to use. |
| `orchestrate_extraction.py` | 8-GPU pool over `(null_type × split × shard) → 96 jobs`. |
| `orchestrate_training.py` | 8-GPU pool over `(null_type × seed) → 20 jobs`. Calls `BICR_train.py`. |
| `orchestrate_eval.py` | 8-GPU pool over `(null_type × seed) → 20 jobs`. Calls `BICR_eval.py`. |
| `compare_nulls.py` | Aggregates per-(null,seed) test metrics, paired Wilcoxon vs black, emits markdown + LaTeX. |

## Run order

```bash
cd <repo-root> && conda activate <your-vllm-env>

# Phase A — sanity check (100 samples per null)
python models/BICR/null_ablation/orchestrate_extraction.py --dry-run --gpus 0 1 2 3

# Phase A — full extraction (~24-36h)
python models/BICR/null_ablation/orchestrate_extraction.py \
  --gpus 0 1 2 3 4 5 6 7

# Phase B — training (~2h)
python models/BICR/null_ablation/orchestrate_training.py --gpus 0 1 2 3 4 5 6 7

# Phase C — eval (~15 min)
python models/BICR/null_ablation/orchestrate_eval.py --gpus 0 1 2 3 4 5 6 7

# Phase D — comparison (seconds)
python models/BICR/null_ablation/compare_nulls.py
```

## Outputs

```
data/null_ablation_extraction/{null_type}/Qwen3-VL-8B-Instruct/{train,val,test}/samples/*.npz
trained_models/SPARROW/BICR_null_ablation/{null_type}/Qwen3-VL-8B-Instruct/seed_{N}/best/...
results/SPARROW/BICR_null_ablation/{null_type}/Qwen3-VL-8B-Instruct/seed_{N}/test/test_{labels,results}.json
models/BICR/null_ablation/null_ablation_results.md
manuscript/tables/null_ablation.tex
```

## Determinism

- Pixel-shuffle and gaussian-noise nulls use
  `get_deterministic_seed(hash_id, args.null_seed, salt='null_<type>')` so the
  same hash_id gets the same null pixels across reruns.
- All other views (base, paraphrase, image-noise, swap) are extracted with
  identical config to v3 (`dtype=float32`, `attn_implementation='eager'`,
  `max_image_dim=2048`, `layer_offsets=0`), so h_base/h_para/h_noise/h_swap
  match v3's bit-for-bit.
- BICR_train uses Optuna with seeded TPESampler so the architecture search is
  also reproducible per seed.
