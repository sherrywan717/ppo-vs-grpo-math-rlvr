# Matched 0.5B PPO/GRPO pilot

**Matched 0.5B pilot - not the final benchmark**

All six frozen single-update runs completed with exact 16-completion and one-optimizer/global-step contracts. Historical engineering failures are excluded from every statistic. Three seeds do not support significance claims or a claim that either algorithm is superior.

## Per-seed results

| Algorithm | Seed | Run ID | Reward | Pass@1 | Pass@4 | Format | Valid expr. | Number usage | Tokens | Loss | Value loss | Grad norm | VRAM MiB | Seconds | GPU-hours | CNY |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| GRPO | 42 | `grpo_matched_0p5b_seed42_20260716T115219Z` | 0.078125 | 0 | 0 | 0.1875 | 0.375 | 0.0625 | 545 | 0.303718 | null | 1.93786 | 3301 | 11.3425 | 0.00315071 | 0.0279783 |
| PPO | 42 | `ppo_matched_0p5b_seed42_20260716T114710Z` | 0.078125 | 0 | 0 | 0.125 | 0.375 | 0.125 | 574 | 0.0406283 | 6.80511 | null | 11189 | 17.9055 | 0.00497375 | 0.0441669 |
| GRPO | 123 | `grpo_matched_0p5b_seed123_20260716T115714Z` | 0.084375 | 0 | 0 | 0.25 | 0.375 | 0.125 | 724 | 0.21627 | null | 1.62095 | 3301 | 11.0505 | 0.00306958 | 0.0272579 |
| PPO | 123 | `ppo_matched_0p5b_seed123_20260716T120000Z` | 0.09375 | 0 | 0 | 0.1875 | 0.5 | 0.1875 | 565 | 0.0125609 | 3.94414 | null | 10015 | 15.2371 | 0.00423253 | 0.0375849 |
| GRPO | 2026 | `grpo_matched_0p5b_seed2026_20260716T123716Z` | 0.078125 | 0 | 0 | 0.0625 | 0.5625 | 0.125 | 703 | 0.463458 | null | 2.62471 | 3301 | 11.4586 | 0.00318294 | 0.0282645 |
| PPO | 2026 | `ppo_matched_0p5b_seed2026_20260716T122924Z` | 0.09375 | 0 | 0 | 0 | 0.6875 | 0.25 | 512 | -0.0195912 | 11.3005 | null | 11189 | 16.2806 | 0.00452238 | 0.0401587 |

## Three-seed summary

| Algorithm | Metric | Mean | Sample SD | Seed 42 | Seed 123 | Seed 2026 |
|---|---|---:|---:|---:|---:|---:|
| PPO | format_accuracy | 0.104167 | 0.0954703 | 0.125 | 0.1875 | 0 |
| PPO | valid_expression_rate | 0.520833 | 0.157288 | 0.375 | 0.5 | 0.6875 |
| PPO | number_usage_accuracy | 0.1875 | 0.0625 | 0.125 | 0.1875 | 0.25 |
| PPO | reward_mean | 0.0885417 | 0.0090211 | 0.078125 | 0.09375 | 0.09375 |
| PPO | mean_group_reward_variance | 0.00134115 | 5.9669e-05 | 0.00128906 | 0.00132812 | 0.00140625 |
| PPO | generated_tokens | 550.333 | 33.5012 | 574 | 565 | 512 |
| PPO | wall_time_seconds | 16.4744 | 1.34472 | 17.9055 | 15.2371 | 16.2806 |
| PPO | peak_vram_mib | 10797.7 | 677.809 | 11189 | 10015 | 11189 |
| PPO | gpu_hours | 0.00457622 | 0.000373532 | 0.00497375 | 0.00423253 | 0.00452238 |
| PPO | cost_cny | 0.0406368 | 0.00331697 | 0.0441669 | 0.0375849 | 0.0401587 |
| GRPO | format_accuracy | 0.166667 | 0.0954703 | 0.1875 | 0.25 | 0.0625 |
| GRPO | valid_expression_rate | 0.4375 | 0.108253 | 0.375 | 0.375 | 0.5625 |
| GRPO | number_usage_accuracy | 0.104167 | 0.0360844 | 0.0625 | 0.125 | 0.125 |
| GRPO | reward_mean | 0.0802083 | 0.00360844 | 0.078125 | 0.084375 | 0.078125 |
| GRPO | mean_group_reward_variance | 0.00160156 | 0.00048789 | 0.00144531 | 0.00121094 | 0.00214844 |
| GRPO | generated_tokens | 657.333 | 97.8485 | 545 | 724 | 703 |
| GRPO | wall_time_seconds | 11.2839 | 0.210271 | 11.3425 | 11.0505 | 11.4586 |
| GRPO | peak_vram_mib | 3301 | 0 | 3301 | 3301 | 3301 |
| GRPO | gpu_hours | 0.00313441 | 5.84085e-05 | 0.00315071 | 0.00306958 | 0.00318294 |
| GRPO | cost_cny | 0.0278335 | 0.000518668 | 0.0279783 | 0.0272579 | 0.0282645 |

## Interpretation and limitations

- Total resource window: 83.274789 seconds; 0.023131886 GPU-hours; estimated cost CNY 0.205411.
- Pass@1 and pass@4 were zero for every run. This pilot validates comparable execution and artifacts, not mathematical capability or learning.
- Shaped reward varied despite zero canonical success. This is optimization signal, not task correctness.
- PPO and GRPO losses have different definitions and are not compared as quality. PPO grad norm is null because TRL PPO did not emit it; GRPO KL is null because beta=0.
- Recovered backup-sidecar warnings remain documented; actual archive hashes agree with committed assessments.

## Historical engineering failures

- `ppo_matched_0p5b_seed42_20260714T073357Z`
- `ppo_matched_0p5b_seed42_20260714T082003Z`
- `ppo_matched_0p5b_seed42_20260714T085240Z`
- `ppo_matched_0p5b_seed42_20260716T111934Z`

These runs are excluded from all aggregates.

## Next-stage recommendation

Proceed to CPU-only configuration freezing for the 1.5B GSM8K+MATH stage, retaining the same evidence-precedence, checkpoint-safety, and no-auto-retry discipline. Do not treat this pilot as the final benchmark.
