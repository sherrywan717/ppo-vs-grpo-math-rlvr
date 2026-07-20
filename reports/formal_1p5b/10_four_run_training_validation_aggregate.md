# Four-run PPO versus GRPO training and validation aggregate

All four active formal training runs are now complete under the frozen Qwen 1.5B contract: PPO and GRPO at seeds 42 and 123. This is a pre-final-test, descriptive two-seed aggregate. It does not claim statistical significance and does not use validation or the independent test baseline to retune or select checkpoints.

## Matched contract

Every run used the same model/revision, train and validation manifests, prompt, reward, parser/verifier, policy LoRA, sampling, 256-token completion cap, 512 training completions, 131,072-token hard cap, 32 optimizer/global updates, and checkpoints/validation at 8/16/24/32. PPO alone has a value adapter/scalar head and value loss; GRPO alone uses four-completion relative-advantage groups. PPO and GRPO losses are not numerically comparable.

## Training outcomes by seed

| Algorithm | Seed | Tokens | Mean reward | Canonical pass | Format | Parseable |
|---|---:|---:|---:|---:|---:|---:|
| PPO | 42 | 51,369 | 0.236523 | 16.7969% | 48.6328% | 42.5781% |
| PPO | 123 | 51,969 | 0.216113 | 15.2344% | 44.7266% | 40.2344% |
| GRPO | 42 | 50,773 | 0.267188 | 19.1406% | 54.6875% | 46.8750% |
| GRPO | 123 | 52,284 | 0.244238 | 16.9922% | 51.5625% | 44.1406% |

Across the two seeds, training canonical-pass means ± sample SD were PPO 16.0156% ± 1.1049 pp and GRPO 18.0664% ± 1.5192 pp. These are on-policy rollout outcomes, not held-out test estimates. Both algorithms retained reward variance; PPO seed123 had 92/128 nonzero-variance diagnostic groups, while GRPO seeds 42/123 had 101/128 and 100/128.

Native entropy magnitudes are not compared across algorithms because PPO uses unmasked response-axis TRL entropy while GRPO uses its completion mask through EOS. Unified `response_token_entropy_mean` is unavailable for all runs under the no-extra-forward/non-intrusive contract. Within each algorithm, neither two-seed trace demonstrates a sustained collapse; update-level entropy remains variable.

## Fixed step-32 validation

| Algorithm | Seed | Correct | Pass@1 | Format | Parseable | Truncation |
|---|---:|---:|---:|---:|---:|---:|
| PPO | 42 | 2/64 | 3.1250% | 10.9375% | 10.9375% | 10.9375% |
| PPO | 123 | 3/64 | 4.6875% | 15.6250% | 15.6250% | 7.8125% |
| GRPO | 42 | 5/64 | 7.8125% | 21.8750% | 18.7500% | 3.1250% |
| GRPO | 123 | 6/64 | 9.3750% | 23.4375% | 21.8750% | 7.8125% |

Two-seed step-32 pass@1 means ± sample SD were PPO 3.90625% ± 1.10485 pp and GRPO 8.59375% ± 1.10485 pp. GRPO is higher by 3–4 correct problems per seed on this 64-problem validation, but two seeds and 64 problems do not establish statistical significance or general superiority. Each problem has one candidate, so pass@4 is null/unavailable for all four runs.

The validation curves are PPO42 4.6875→3.125%, PPO123 3.125→4.6875%, GRPO42 3.125→7.8125%, and GRPO123 4.6875→9.375%. The fixed step-32 policy was chosen before execution; no checkpoint was selected from these trajectories.

## Resources

| Algorithm | Seed | Wall time | Peak VRAM | GPU-hours | Cost |
|---|---:|---:|---:|---:|---:|
| PPO | 42 composite | 1,728.72 s | 53,151 MiB | 0.480200 | CNY 4.2642 |
| PPO | 123 | 1,534.44 s | 53,821 MiB | 0.426233 | CNY 3.7849 |
| GRPO | 42 | 1,189.82 s | 11,209 MiB | 0.330505 | CNY 2.9349 |
| GRPO | 123 | 1,113.96 s | 8,741 MiB | 0.309433 | CNY 2.7478 |

PPO's separate value backbone materially increases peak memory. Seed-42 PPO resource scope includes recovered validation processes after its post-training cadence failure; seed123 validation was in-process. Costs use CNY 8.88/GPU-hour.

## What this stage supports

The four-run aggregate supports an auditable statement that all matched training and validation contracts executed and that GRPO showed a higher descriptive step-32 validation pass@1 in both seeds while using less peak memory. It cannot establish statistical significance, final test generalization, or causality. The formal test baseline is a separate protocol and is not subtracted from validation. Final evaluation remains unexecuted and separately authorized.

## Figures

- [Four-run training comparison](figures/four_run_training_comparison.png)
- [Four-run checkpoint validation](figures/four_run_validation_comparison.png)
- [Four-run resource comparison](figures/four_run_resource_comparison.png)
