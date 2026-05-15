# BICR Null-Image Ablation — Qwen3-VL-8B-Instruct

Test split: `test`. Each cell = mean ± std across 5 seeds (23, 42, 137, 2024, 3407). p-values from paired Wilcoxon vs the black baseline (n = 5 paired by seed).

## Mean ± std per metric

| Null type | AUROC | AUCPR | ECE | Brier | Acc | F1 |
|---|---|---|---|---|---|---|
| **black (baseline)** | 0.8008 ± 0.0019 | 0.9014 ± 0.0009 | 0.0886 ± 0.0167 | 0.1747 ± 0.0034 | 0.7281 ± 0.0044 | 0.7866 ± 0.0068 |
| **white** | 0.6780 ± 0.0383 | 0.5125 ± 0.0224 | 0.1332 ± 0.0276 | 0.2223 ± 0.0144 | 0.6908 ± 0.0026 | 0.3665 ± 0.0339 |
| **gaussian_noise** | 0.7035 ± 0.0670 | 0.5338 ± 0.0366 | 0.1231 ± 0.0346 | 0.2125 ± 0.0222 | 0.6961 ± 0.0069 | 0.3872 ± 0.0636 |
| **blurred** | 0.7007 ± 0.0277 | 0.5231 ± 0.0160 | 0.1255 ± 0.0311 | 0.2153 ± 0.0146 | 0.6919 ± 0.0052 | 0.3273 ± 0.0997 |
| **pixel_shuffled** | 0.7215 ± 0.0292 | 0.5354 ± 0.0270 | 0.1206 ± 0.0216 | 0.2101 ± 0.0136 | 0.6957 ± 0.0069 | 0.3657 ± 0.0493 |

## Paired Wilcoxon p-values (each null vs black)

| Null type vs black | AUROC | AUCPR | ECE | Brier | Acc | F1 |
|---|---|---|---|---|---|---|
| white | 0.062 | 0.062 | 0.062 | 0.062 | 0.062 | 0.062 |
| gaussian_noise | 0.062 | 0.062 | 0.125 | 0.062 | 0.062 | 0.062 |
| blurred | 0.062 | 0.062 | 0.188 | 0.062 | 0.062 | 0.062 |
| pixel_shuffled | 0.062 | 0.062 | 0.062 | 0.062 | 0.062 | 0.062 |

## Δ vs black (mean of new − mean of black, lower-is-better metrics negated)

| Null type | AUROC | AUCPR | ECE | Brier | Acc | F1 |
|---|---|---|---|---|---|---|
| white | -0.1228 | -0.3889 | -0.0446 | -0.0476 | -0.0373 | -0.4201 |
| gaussian_noise | -0.0973 | -0.3676 | -0.0346 | -0.0379 | -0.0320 | -0.3994 |
| blurred | -0.1001 | -0.3783 | -0.0370 | -0.0406 | -0.0362 | -0.4593 |
| pixel_shuffled | -0.0793 | -0.3660 | -0.0320 | -0.0354 | -0.0324 | -0.4209 |
