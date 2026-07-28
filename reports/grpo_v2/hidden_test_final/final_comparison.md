# GRPO-v2 frozen four-model hidden-test comparison

## Headline candidate-0 accuracy (400 problems)

| Model | Correct | Accuracy | Format | Parseable | Truncated |
|---|---:|---:|---:|---:|---:|
| Base | 6/400 | 1.50% | 7.75% | 7.00% | 10.75% |
| Old GRPO-v1 | 17/400 | 4.25% | 21.25% | 17.00% | 11.00% |
| Warm-start-only | 10/400 | 2.50% | 12.00% | 10.25% | 11.50% |
| Selected GRPO-v2 | 43/400 | 10.75% | 58.75% | 48.50% | 8.50% |

## Shared 100-problem unbiased pass@k

| Model | pass@1 | pass@4 | pass@10 |
|---|---:|---:|---:|
| Base | 1.70% | 6.03% | 12.00% |
| Old GRPO-v1 | 5.60% | 15.87% | 25.00% |
| Warm-start-only | 3.30% | 11.03% | 19.00% |
| Selected GRPO-v2 | 14.40% | 31.14% | 42.00% |

Candidate-0 accuracy on all 400 problems is the primary metric. The three pass@k estimates use the same n=10 candidate pool on the frozen 100-problem subset.

## Paired candidate-0 comparisons

| Comparison | Delta | Paired bootstrap 95% CI | Improved / regressed | McNemar exact p |
|---|---:|---:|---:|---:|
| Base → old GRPO-v1 | +2.75 pp | [+1.25, +4.50] pp | 11 / 0 | 0.0009766 |
| Base → warm-start-only | +1.00 pp | [+0.25, +2.00] pp | 4 / 0 | 0.125 |
| Base → selected GRPO-v2 | +9.25 pp | [+6.50, +12.25] pp | 38 / 1 | 1.46e-10 |
| Old GRPO-v1 → selected GRPO-v2 | +6.50 pp | [+3.75, +9.50] pp | 31 / 5 | 1.29e-5 |
| Warm-start-only → selected GRPO-v2 | +8.25 pp | [+5.25, +11.25] pp | 37 / 4 | 1.03e-7 |

The warm-start-only comparison has only four discordant pairs; its exact McNemar
result does not establish a reliable mathematical-accuracy gain. Selected GRPO-v2
improves both strict protocol adherence and canonical accuracy relative to the
warm-start-only checkpoint, supporting an incremental RLVR contribution under this
single frozen seed and protocol.

## Dataset and resource summary

- Selected GRPO-v2 candidate-0: GSM8K 27/200 (13.5%); MATH500 16/200 (8.0%).
- Old GRPO-v1 candidate-0: GSM8K 11/200 (5.5%); MATH500 6/200 (3.0%).
- Warm-start-only candidate-0: GSM8K 6/200 (3.0%); MATH500 4/200 (2.0%).
- Base candidate-0: GSM8K 3/200 (1.5%); MATH500 3/200 (1.5%).
- The three Stage S.4 runs used 453,920 generated tokens, 1.622362 GPU-hours,
  and ¥14.4066. Including recovered Base, the frozen four-model evaluation used
  606,487 generated tokens, 1.980286 GPU-hours, and ¥17.5849.
- Peak observed nvidia-smi VRAM across all four models was 5,321 MiB.

Base generation remains historically
`engineering_failure_after_generation_during_metric_finalization`; its immutable
evidence was recovered as
`scientifically_complete_with_recovered_metric_finalization` without regeneration.

This is one frozen seed. MATH Level 1 has only three problems and is diagnostic only. Hidden-test results cannot trigger tuning, checkpoint reselection or another run.
