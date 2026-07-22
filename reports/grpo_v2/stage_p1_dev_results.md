# Stage P.1 matched Base versus warm-start dev-v2 results

Both pre-registered runs completed exactly once on the same ordered 128-problem
`dev_v2` manifest. Problem IDs, content hashes, prompt hashes, candidate index 0,
sampling identity, and per-problem generation seeds match row by row. Neither run
called training, backward, an optimizer, or checkpoint writing.

| Metric | Base | Warm-start | Delta |
|---|---:|---:|---:|
| Candidate-0 pass@1 | 6/128 (4.6875%) | 8/128 (6.2500%) | +1.5625 pp |
| Format valid | 17/128 (13.2813%) | 23/128 (17.9688%) | +4.6875 pp |
| Parseable / valid answer | 16/128 (12.5000%) | 20/128 (15.6250%) | +3.1250 pp |
| Accuracy given parseable | 6/16 (37.5000%) | 8/20 (40.0000%) | +2.5000 pp |
| EOS | 124/128 (96.8750%) | 125/128 (97.6563%) | +0.7813 pp |
| Truncation | 4/128 (3.1250%) | 3/128 (2.3438%) | -0.7813 pp |

The paired candidate-0 transitions are five improvements, three regressions,
three unchanged-correct, and 117 unchanged-wrong. The paired bootstrap 95% interval
for the pass@1 delta is [-2.34375, +6.25] percentage points (10,000 resamples,
seed 42); exact McNemar p=0.7265625. The small positive dev change is therefore
uncertain. The defensible conclusion is that warm-start improved protocol adherence
and produced a modest observed dev pass@1 increase, not that it has established a
hidden-test mathematical ability gain.

## Domain and level results

- GSM8K: Base 1/64; warm-start 2/64.
- MATH: Base 5/64; warm-start 6/64.
- MATH Level 1: 3/16 for both.
- MATH Level 2: 2/24 for both.
- MATH Level 3: Base 0/24; warm-start 1/24.

Pass@4 and pass@10 are unavailable with reason
`dev_protocol_one_candidate_per_problem`; the hidden n=10 estimator was not run.
Mechanical case selection and all paired rows are saved under
`dev_evaluation/paired/`. No hidden-test data or result was used.

## Resource accounting

| Run | Tokens | Wall time | Peak VRAM | GPU-hours | Cost |
|---|---:|---:|---:|---:|---:|
| Base | 12,949 | 270.530 s | 3,831 MiB | 0.075147 | ¥0.6673 |
| Warm-start | 13,176 | 392.726 s | 3,841 MiB | 0.109090 | ¥0.9687 |
| Total | 26,125 | 663.256 s | — | 0.184238 | ¥1.6360 |

Both workers exited, left no compute process, and restored GPU memory to the 0 MiB
pre-run baseline. Full run archives and their SHA256 sidecars remain outside Git.
