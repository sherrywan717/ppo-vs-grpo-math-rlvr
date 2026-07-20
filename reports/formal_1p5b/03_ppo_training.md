# Formal PPO seed 42 training and recovered validation

Composite status: `scientifically_complete_with_recovered_validation`. Optimization occurred once in immutable run `ppo_formal_1p5b_seed42_20260719T131800Z`; four validation-only runs later evaluated its trusted checkpoints after Stage H.2 fixed the post-training cadence cursor. Training was not rerun or resumed, the original failure summary/checksums were not changed, and formal test was not run.

## Independent ledgers

| Ledger | Problems/pairs | Completions | Generated tokens | Training budget? |
|---|---:|---:|---:|---|
| Training rollout | 128 unique prompts | 512 | 51,369 | yes; cap 131,072 |
| Checkpoint validation | 256 checkpoint/problem pairs (64 frozen problems × 4) | 256 | 30,541 | no |
| Final test | not run | 0 | unavailable | separate, not authorized |

## Training evidence

The 32 update/optimizer/global-step contract completed. Across updates, mean reward was 0.236523, verifier-derived canonical pass 16.7969%, format validity 48.6328%, and parseable rate 42.5781%. Mean policy/value/total loss was 0.009956/4.474166/0.457373; approximate KL 0.000607, ratio 0.994854, clip fraction 0.004608.

TRL native policy entropy (`policy/entropy_avg`) moved from 1.1067 to 1.7759 nats and did not show sustained collapse. Its mean was 1.2805. It is an unmasked response-axis logits reduction: prompt excluded, PAD/EOS not excluded, BF16 logits, TRL 0.24.0 / Transformers 4.57.6. Unified `response_token_entropy_mean` and entropy std are null/unavailable because obtaining them would require intrusive or extra logits work. Policy/value grad norms, advantage and return are likewise null/unavailable with recorded reasons.

Reward variance was nonzero in 96/128 prompt groups; 32/128 were all-equal/zero-advantage. Mean unique-completion rate was 99.4141%, duplicate rate 0.5859%, EOS 97.2656%, and truncation 2.7344%. Reward, format, parseable, and canonical pass were volatile rather than monotonically improving. Approximate KL and clip fraction stayed small and ratio remained near one, consistent with restrained PPO updates; value loss stayed finite but materially larger than policy loss. Entropy rose overall instead of collapsing, so reward changes were not accompanied by rapid entropy loss. The run had learning signal, but one seed cannot establish capability gain or PPO superiority.

The runtime per-update `valid_answer_rate=0` is explicitly not used: it came from a stale nested `components` lookup after RewardResult evidence became flat. Primary canonical statuses are complete, so this aggregate derives parseable as `wrong_answer` or `verified_pass`. This is a reporting correction, not a model/config change or GPU rerun.

## Checkpoint validation

| Step | Pass@1 | Format | Parseable | Acc. given parseable | Tokens | EOS | Trunc. | Wall (s) | GPU-h | CNY |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 4.6875% | 14.0625% | 14.0625% | 33.33% | 7,533 | 90.62% | 9.38% | 238.03 | 0.06612 | 0.587 |
| 16 | 3.1250% | 10.9375% | 10.9375% | 28.57% | 7,848 | 92.19% | 7.81% | 251.36 | 0.06982 | 0.620 |
| 24 | 3.1250% | 10.9375% | 10.9375% | 28.57% | 7,663 | 89.06% | 10.94% | 247.54 | 0.06876 | 0.611 |
| 32 | 3.1250% | 10.9375% | 10.9375% | 28.57% | 7,497 | 89.06% | 10.94% | 240.52 | 0.06681 | 0.593 |

Validation pass@1 was 4.6875% at step 8 and 3.125% at steps 16/24/32: no monotonic improvement, with a one-problem decrease after step 8. Format/parseable fell from 14.0625% to 10.9375%. Status distributions were 55/6/3 at step 8 and 57/5/2 thereafter for format_error/wrong_answer/verified_pass. These are small fixed-manifest estimates; results were not used to alter training or choose a checkpoint.

Every validation problem has one sampled candidate. Therefore sampled pass@1 is the mean of `canonical_correct`; pass@4 is `null`, `available=false`, reason `validation_protocol_has_one_candidate_per_problem`. Native aggregate pass@1 is null because validation rows use `sample_kind=validation`; this report derives it from complete primary evidence.

Across checkpoints, GSM8K pass@1 was 3.125%, 0%, 0%, 0%; MATH was 6.25% throughout. MATH Level 1–5 were 0%, 0%, 7.69%, 20%, 0% at every checkpoint. Tiny level denominators (3/8/13/5/3) preclude broad difficulty claims. Truncation ranged 7.81–10.94%; it is separated by status/domain/level and is not automatically interpreted as mathematical inability.

Four validations used 30,541 tokens, 977.45s summed wall time, 0.271514 GPU-hours and CNY 2.411047; peak nvidia-smi VRAM was 3847 MiB. All backups and artifact checksums verified.

## Baseline relation and limits

The existing seed-42 baseline is the formal test protocol (sampled pass@1 4%, pass@4 10%), not this 64-problem validation set. No matched base-model 64-problem validation exists, so `base_validation_delta=null`, `available=false`, reason `matching_base_validation_not_available`. The test baseline was not subtracted, used for tuning, or used to select a checkpoint.

This composite establishes an evidence-complete PPO seed-42 training plus recovered checkpoint-validation result. It does not include final test, a second PPO seed, any GRPO result, or evidence of algorithm superiority.

## Rebuildable figures

- [Training reward and accuracy](figures/ppo_training_reward_accuracy.png)
- [Training losses](figures/ppo_training_losses.png)
- [Policy diagnostics](figures/ppo_policy_diagnostics.png)
- [Entropy vs tokens](figures/entropy_vs_tokens.png)
- [Entropy vs reward](figures/entropy_vs_reward.png)
- [Completion diversity](figures/completion_diversity.png)
- [Reward-group learning signal](figures/ppo_reward_group_learning_signal.png)
- [Checkpoint validation curve](figures/ppo_checkpoint_validation_curve.png)
- [Validation status distribution](figures/ppo_validation_status_distribution.png)
- [Validation completion analysis](figures/ppo_validation_completion_analysis.png)
- [Validation resource cost](figures/ppo_validation_resource_cost.png)

All figures are rebuilt from committed CSV/JSON sources. Raw full-text/token evidence and checkpoint payloads remain outside Git in run directories and verified backups.
