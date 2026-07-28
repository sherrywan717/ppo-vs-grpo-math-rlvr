# Four-model candidate-0 paired error analysis

All comparisons use the same frozen 400 problems, candidate index 0, prompt, seed schedule, parser and verifier.

| Comparison | Improved | Regressed | Both correct | Both wrong | Delta (pp) | McNemar p |
|---|---:|---:|---:|---:|---:|---:|
| base → old_grpo_v1 | 11 | 0 | 6 | 383 | +2.75 | 0.000976562 |
| base → warmstart_only | 4 | 0 | 6 | 390 | +1.00 | 0.125 |
| base → selected_grpo_v2 | 38 | 1 | 5 | 356 | +9.25 | 1.45519e-10 |
| old_grpo_v1 → selected_grpo_v2 | 31 | 5 | 12 | 352 | +6.50 | 1.29135e-05 |
| warmstart_only → selected_grpo_v2 | 37 | 4 | 6 | 353 | +8.25 | 1.02584e-07 |

This is a single-seed paired result. Test outcomes are final and cannot trigger tuning or retraining.
