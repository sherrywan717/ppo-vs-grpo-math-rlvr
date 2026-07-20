# Formal PPO seed 42 versus seed 123 stability review

This is a descriptive two-seed review under the unchanged frozen PPO contract. Seed 42 is the transparent composite `ppo_formal_1p5b_seed42_composite_20260720T020928Z`; seed 123 is `ppo_formal_1p5b_seed123_20260720T043732Z`. Both contain exactly one 32-update optimization trajectory, 512 training completions, trusted checkpoints 8/16/24/32, and four identical 64-problem validations. No tuning, final test, or checkpoint selection occurred between seeds.

## Training outcomes

| Metric | Seed 42 | Seed 123 | Two-seed mean ± sample SD |
|---|---:|---:|---:|
| Training rollout tokens | 51,369 | 51,969 | 51,669 ± 424 |
| Mean shaped reward | 0.236523 | 0.216113 | 0.226318 ± 0.014432 |
| Canonical pass | 16.7969% | 15.2344% | 16.0156% ± 1.1049 pp |
| Format accuracy | 48.6328% | 44.7266% | 46.6797% ± 2.7621 pp |
| Canonical parseable | 42.5781% | 40.2344% | 41.4062% ± 1.6573 pp |
| Native entropy mean | 1.280508 | 1.128555 | not cross-algorithm comparable |
| Truncation | 2.7344% | 3.7109% | 3.2227% ± 0.6905 pp |

Both runs had finite losses and nonzero reward variance. Seed 123 was modestly lower on on-policy training reward and canonical pass, illustrating seed variability. Native entropy is comparable only within this PPO implementation and is not the unified response-token entropy metric.

## Checkpoint validation

| Step | Seed 42 pass@1 | Seed 123 pass@1 | Difference (123−42) |
|---:|---:|---:|---:|
| 8 | 4.6875% (3/64) | 3.1250% (2/64) | −1.5625 pp |
| 16 | 3.1250% (2/64) | 4.6875% (3/64) | +1.5625 pp |
| 24 | 3.1250% (2/64) | 4.6875% (3/64) | +1.5625 pp |
| 32 | 3.1250% (2/64) | 4.6875% (3/64) | +1.5625 pp |

At fixed step 32, the two PPO seeds differ by one correct problem. The two-seed mean is 3.90625% with sample SD 1.10485 percentage points. This is raw small-sample variability, not statistical significance. Pass@4 is unavailable for both because validation has one candidate per problem.

## Resources

| Metric | Seed 42 composite | Seed 123 |
|---|---:|---:|
| Wall time | 1,728.72 s | 1,534.44 s |
| GPU-hours | 0.480200 | 0.426233 |
| Cost | CNY 4.2642 | CNY 3.7849 |
| Peak nvidia-smi VRAM | 53,151 MiB | 53,821 MiB |

Seed 42 uses one training process plus four recovered validation-only processes, while seed 123 performed validation in-process after training. Resource scope is therefore disclosed rather than treated as identical implementation overhead.

## Interpretation limits

Two seeds allow raw values, means, sample SDs, and paired descriptive comparisons, but not a significance or general superiority claim. Formal test has not run for either adapter. The existing test baseline is not a validation delta, and the frozen step-32 checkpoint was not selected from these validation results.

## Figures

- [Training stability](figures/ppo_seed_stability_training.png)
- [Checkpoint validation stability](figures/ppo_seed_stability_validation.png)
- [Native PPO entropy](figures/ppo_seed_stability_entropy.png)
- [Resource comparison](figures/ppo_seed_stability_resources.png)
