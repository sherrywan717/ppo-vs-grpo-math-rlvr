# Formal GRPO seed 42 versus seed 123 stability review

This is a descriptive two-seed review under the unchanged frozen GRPO contract. Seed 42 is `grpo_formal_1p5b_seed42_20260720T031006Z` and seed 123 is `grpo_formal_1p5b_seed123_20260720T035927Z`. Both completed 32 updates, 128 microsteps, 512 training completions, four checkpoints, and four identical 64-problem validations. No tuning, test evaluation, or checkpoint selection occurred between seeds.

## Training outcomes

| Metric | Seed 42 | Seed 123 |
|---|---:|---:|
| Training rollout tokens | 50,773 | 52,284 |
| Mean shaped reward | 0.267188 | 0.244238 |
| Canonical pass | 19.1406% | 16.9922% |
| Format accuracy | 54.6875% | 51.5625% |
| Canonical parseable | 46.8750% | 44.1406% |
| Nonzero-variance groups | 101/128 | 100/128 |
| Zero-advantage groups | 27/128 | 28/128 |
| Native entropy mean | 0.281434 | 0.264968 |
| Truncation | 3.1250% | 2.9297% |

Training-level reward and verifier outcomes varied across seeds, while both retained substantial group variance. Seed 123 had lower on-policy training reward/pass than seed 42 but a stronger held-out validation curve. This illustrates why on-policy training completions cannot substitute for checkpoint validation.

## Checkpoint validation

| Step | Seed 42 pass@1 | Seed 123 pass@1 | Difference (123−42) |
|---:|---:|---:|---:|
| 8 | 3.1250% | 4.6875% | +1.5625% |
| 16 | 4.6875% | 7.8125% | +3.1250% |
| 24 | 6.2500% | 9.3750% | +3.1250% |
| 32 | 7.8125% | 9.3750% | +1.5625% |


Seed 42 rose 3.125%→7.8125%; seed 123 rose 4.6875%→9.375%. At fixed step 32, the two results differ by one correct problem (5/64 versus 6/64). This is small-sample seed variability, not statistical significance or a general performance claim. Pass@4 is unavailable for both because validation has one candidate per problem.

## Resources

| Metric | Seed 42 | Seed 123 |
|---|---:|---:|
| Wall time | 1189.82 s | 1113.96 s |
| GPU-hours | 0.330505 | 0.309433 |
| Cost | CNY 2.9349 | CNY 2.7478 |
| Peak nvidia-smi VRAM | 11209 MiB | 8741 MiB |

## Interpretation limits

Two seeds permit raw values, means, sample SDs, paired deltas, and later problem-level bootstrap intervals under the frozen plan. This stage alone does not establish statistical significance. Formal test has not run for either GRPO adapter, and test baseline is not a validation delta. PPO seed 123 and the fixed final evaluations remain necessary before the final PPO-versus-GRPO comparison.

## Figures

- [Training stability](figures/grpo_seed_stability_training.png)
- [Validation stability](figures/grpo_seed_stability_validation.png)
- [Native entropy trends](figures/grpo_seed_stability_entropy.png)
- [Resource comparison](figures/grpo_seed_stability_resources.png)
