# GRPO-v2 warm-start matched dev-v2 evaluation

Both frozen runs completed 128/128 candidate-0 generations with exact problem/hash/seed/prompt alignment. Training, backward, optimizer steps and checkpoint writes were zero.

## Main results

| Metric | Base | Warm-start | Delta |
|---|---:|---:|---:|
| candidate-0 pass@1 | 6/128 (4.69%) | 8/128 (6.25%) | +1.56 pp |
| format rate | 17/128 (13.28%) | 23/128 (17.97%) | +4.69 pp |
| parseable / valid-answer | 16/128 (12.50%) | 20/128 (15.62%) | +3.12 pp |
| truncation | 4/128 (3.12%) | 3/128 (2.34%) | -0.78 pp |

Paired candidate-0 transitions were 5 improvements, 3 regressions, 3 unchanged-correct and 117 unchanged-wrong. The paired bootstrap 95% interval for the pass@1 delta is [-2.34, +6.25] pp. McNemar exact p=0.7265625.

Warm-start improved protocol adherence and produced a small positive dev-v2 pass@1 change. The interval includes zero, so this is not evidence of a reliable hidden-test ability improvement. It is a dev-only result and cannot be used to alter the hidden test. Proceeding to GRPO-v2 requires explicit user review and authorization.

Pass@4 and pass@10 are unavailable (`dev_protocol_one_candidate_per_problem`); no n=10 estimator was run.
