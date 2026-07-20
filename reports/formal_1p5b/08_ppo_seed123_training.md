# Formal PPO seed 123 training and checkpoint validation

Status: `scientific_success`. Run `ppo_formal_1p5b_seed123_20260720T043732Z` executed the frozen command once from commit `f6a62eeb8ce59c438f355b675db493552044de18`, with zero retries. No GRPO, seed 2026, baseline, or final-test command ran.

## Contract and independent ledgers

| Ledger | Problems/pairs | Completions | Generated tokens | Training budget? |
|---|---:|---:|---:|---|
| Training rollout | 128 unique prompts | 512 | 51,969 | yes; cap 131,072 |
| Checkpoint validation | 256 checkpoint/problem pairs | 256 | 26,859 | no |
| Final test | not run | 0 | unavailable | separate and unauthorized |

Training completed 32/32 update, optimizer, and global steps. Each update persisted 16 completion rows and one metric row before checkpoint callbacks. Checkpoints 8/16/24/32 bind exact prefixes of 128/256/384/512 completions and 8/16/24/32 metrics.

## Training metrics

Across 512 completions, mean shaped reward was 0.216113 (population SD 0.342682), canonical pass 15.2344% (78/512), format accuracy 44.7266% (229/512), canonical parseable rate 40.2344% (206/512), and valid-answer component rate 46.0938% (236/512). The last metric is `formal_domain_valid_answer_component_v1`, based on flat `valid_answer_component`; it is not canonical parseability.

Mean policy/value/total loss was 0.004609 / 3.319867 / 0.336595. Approximate KL averaged 0.000615, ratio 0.994947, and clip fraction 0.005078; all required training values remained finite. These diagnostics indicate small PPO updates rather than an obvious instability. PPO loss values are only interpreted within PPO and are not compared numerically to GRPO loss.

Native TRL entropy (`policy/entropy_avg`) averaged 1.128555 nats and moved 1.604956→1.261153, with substantial update-level fluctuation rather than a monotonic collapse. It is an unmasked mean over response-axis logits, excludes prompt but not PAD/EOS, and is then averaged across PPO cells. Unified `response_token_entropy_mean`, entropy std, policy/value grad norms, advantage, and return are null/unavailable with explicit reasons because TRL 0.24.0 did not expose them under the non-intrusive contract.

Training statuses were 283 format errors, 23 parse errors, 128 parseable wrong answers, and 78 canonical passes. EOS rate was 96.2891%, truncation 3.7109%, and exact within-group duplicate rate averaged zero. Truncated rows remain in all headline metrics; truncation is a formatting diagnostic, not automatically mathematical inability.

## Reward-group learning signal

Of 128 four-completion groups, 92 (71.8750%) had nonzero reward variance. Thirty-six were all-equal/zero-advantage and 26 were all-zero. The run therefore contained repeated learning signal and is not a `no_learning_signal` execution. Group statistics are diagnostic; they were computed from training rollout rewards, never test data.

## Frozen checkpoint validation

| Step | Correct | Pass@1 | Format | Parseable | Tokens | Truncation |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 2/64 | 3.1250% | 10.9375% | 10.9375% | 6,593 | 6.2500% |
| 16 | 3/64 | 4.6875% | 14.0625% | 14.0625% | 6,450 | 6.2500% |
| 24 | 3/64 | 4.6875% | 14.0625% | 12.5000% | 7,040 | 7.8125% |
| 32 | 3/64 | 4.6875% | 15.6250% | 15.6250% | 6,776 | 7.8125% |

Each checkpoint used the same frozen 64 problems and one candidate per problem. Pass@4 is `null`, `available=false`, reason `validation_protocol_has_one_candidate_per_problem`. No matching base-model 64-problem validation exists, so base validation delta is null/unavailable. The formal test baseline is a different problem/candidate protocol and is not subtracted or used for checkpoint selection.

At step 32, GSM8K pass@1 was 0/32 and MATH was 3/32 (9.375%). MATH Level 1–5 results were 0/3, 1/8, 1/13, 1/5, and 0/3. These tiny difficulty cells are descriptive only. Validation EOS is unavailable because the rows do not persist an EOS flag.

## Resources, checkpoints, and warnings

Measured wall time was 1,534.438 seconds, 0.426233 GPU-hours, and CNY 3.784948 at CNY 8.88/GPU-hour. Peak nvidia-smi VRAM was 53,821 MiB; mean utilization was 36.32%. PyTorch peak allocated/reserved was 49,560.6/53,142.0 MiB.

Checkpoints 8/16/24/32 contain policy LoRA, value LoRA, scalar head, optimizer, scheduler, Python/NumPy/PyTorch CPU/CUDA RNG, counters, prefixes, identities, and SHA256 inventory. `base_weights_included=false` at every step. All run checksums passed. The verified full backup is `/root/autodl-fs/math-rlvr-backups/ppo_formal_1p5b_seed123_20260720T043732Z.tar.gz`, SHA256 `689924eaa4392a4806f9d1adaa2bbf890b76d6813a6edfeafc2ca50213bc63c0`.

Warnings are limited to TRL/PyTorch deprecation telemetry, unavailable optional metrics, unavailable validation EOS/per-checkpoint resource boundaries, and non-authoritative pre-exit allocator residue. Parent post-exit verification showed 0 MiB and no compute process.

## Rebuildable figures

- [Training reward and verifier metrics](figures/ppo_seed123_training_reward_accuracy.png)
- [PPO losses](figures/ppo_seed123_training_losses.png)
- [Policy diagnostics](figures/ppo_seed123_policy_diagnostics.png)
- [Reward-group learning signal](figures/ppo_seed123_reward_group_learning_signal.png)
- [Checkpoint validation curve](figures/ppo_seed123_checkpoint_validation_curve.png)
- [Validation status distribution](figures/ppo_seed123_validation_status_distribution.png)
- [Completion analysis](figures/ppo_seed123_completion_analysis.png)
- [GPU resources](figures/ppo_seed123_gpu_resources.png)

All figures are generated from committed CSV/JSON. Full completion/token evidence and checkpoint payloads remain outside Git in the verified run and backup.
