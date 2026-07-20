# Seed-42 matched PPO versus GRPO comparison

This is a descriptive same-seed comparison under the frozen Qwen 1.5B contract. PPO is the transparent `scientifically_complete_with_recovered_validation` composite; GRPO is successful run `grpo_formal_1p5b_seed42_20260720T031006Z`. Both share model/revision, manifests, prompt, reward, parser/verifier, policy LoRA, sampling, 256 completion cap, 512 training completions, 131,072-token ceiling, 32 updates, and checkpoint validation at 8/16/24/32.

Algorithm-required differences remain explicit: PPO has a separate value model/value adapter/head and value loss; GRPO uses four-completion group-relative advantages and has no value role. PPO and GRPO loss magnitudes are not directly comparable.

## Training description

| Metric | PPO seed 42 | GRPO seed 42 |
|---|---:|---:|
| Training completions | 512 | 512 |
| Training rollout tokens | 51,369 | 50,773 |
| Mean shaped reward | 0.236523 | 0.267188 |
| Canonical pass | 16.7969% | 19.1406% |
| Format accuracy | 48.6328% | 54.6875% |
| Canonical parseable | 42.5781% | 46.8750% |
| Nonzero-variance groups | 96/128 | 101/128 |
| Zero-advantage groups | 32/128 | 27/128 |
| EOS | 97.2656% | 96.8750% |
| Truncation | 2.7344% | 3.1250% |
| Duplicate rate | 0.5859% | 0.7813% |

GRPO is descriptively higher on aggregate training reward and verifier outcomes in this one seed, and both runs show substantial nonzero group variance. These are on-policy sampled training outputs, not held-out proof of ability. Neither training curve is monotonic.

Native entropy cannot be compared by magnitude. PPO's `policy/entropy_avg` is an unmasked response-axis reduction that includes PAD/EOS positions; GRPO's `entropy` uses the completion mask. PPO entropy increased overall (1.1067→1.7759), while GRPO fluctuated and ended slightly below its start (0.2823→0.2612) without sustained collapse. GRPO's zero-advantage fraction also did not remain high: 12.5% over updates 25–32. There is no combined low-entropy/zero-advantage collapse signal.

PPO approximate KL, ratio, and clip fraction were available and small. GRPO beta is zero, so KL is unavailable by design; ratio metrics were not exposed and clip fraction was zero. This prevents symmetric optimizer-diagnostic comparisons beyond what each trainer reliably saved.

## Checkpoint validation

| Step | PPO pass@1 | GRPO pass@1 | PPO format | GRPO format | PPO trunc. | GRPO trunc. |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 4.6875% | 3.1250% | 14.0625% | 14.0625% | 9.375% | 9.375% |
| 16 | 3.1250% | 4.6875% | 10.9375% | 15.6250% | 7.8125% | 7.8125% |
| 24 | 3.1250% | 6.2500% | 10.9375% | 17.1875% | 10.9375% | 3.125% |
| 32 | 3.1250% | 7.8125% | 10.9375% | 21.8750% | 10.9375% | 3.125% |

On the same frozen 64 problems, GRPO rose from 2 to 5 canonical passes while PPO moved from 3 to 2 and then stayed flat. At step 32 GRPO led descriptively by 3 problems. The manifest is small and this is one seed, so this does not establish statistical significance or GRPO superiority. Validation was not used to alter training, select a checkpoint, or change the fixed step-32 final-evaluation policy.

Both validation protocols have one candidate per problem. Pass@4 is null/unavailable, never inferred. The existing 4%/10% seed-42 baseline is a separate formal-test protocol and is not a validation delta. No final test ran in this stage.

## Resources

| Scope: training plus checkpoint validation | PPO | GRPO |
|---|---:|---:|
| Wall time | 1,728.72 s | 1,189.82 s |
| GPU-hours | 0.480200 | 0.330505 |
| Cost at CNY 8.88/h | CNY 4.2642 | CNY 2.9349 |
| Peak nvidia-smi VRAM | 53,151 MiB | 11,209 MiB |

PPO's higher memory is consistent with its separate policy/value roles; GRPO has no value model. Wall/cost scopes both include four checkpoint validations, although PPO validations were recovered in separate processes and GRPO validations ran in-process.

## What this seed supports

The evidence supports that both frozen algorithms completed real optimization with nonzero reward variance, safe checkpoints, full per-completion evidence, and the same validation protocol. GRPO showed the stronger validation trajectory and lower measured resource use in seed 42. It does not yet support a general algorithm ranking, reward-hacking conclusion, or test-set generalization claim. Seed 123 and fixed final tests remain necessary for the planned two-seed portfolio analysis.

## Figures

- [Matched training curves](figures/seed42_ppo_vs_grpo_training.png)
- [Checkpoint validation](figures/seed42_ppo_vs_grpo_validation.png)
- [Native entropy trends](figures/seed42_entropy_vs_tokens.png)
- [Measured resources](figures/seed42_ppo_vs_grpo_resources.png)
