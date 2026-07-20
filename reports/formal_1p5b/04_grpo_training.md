# Formal GRPO seed 42 training and checkpoint validation

Status: `scientific_success`. Run `grpo_formal_1p5b_seed42_20260720T031006Z` executed the frozen command once from commit `548e2d371cbc09d5527aed3ed9dbf0ac1ad94a1d`, with zero retries. No PPO, seed 123, baseline, or final-test command ran.

## Independent ledgers

| Ledger | Problems/pairs | Completions | Generated tokens | Training budget? |
|---|---:|---:|---:|---|
| Training rollout | 128 unique prompts | 512 | 50,773 | yes; cap 131,072 |
| Checkpoint validation | 256 checkpoint/problem pairs | 256 | 29,113 | no |
| Final test | not run | 0 | unavailable | separate and unauthorized |

Training completed 32/32 update, optimizer, and global steps. The frozen contract expects 128 microsteps (four per update); trusted checkpoint resume counters record 32/64/96/128 microsteps at steps 8/16/24/32, exactly matching that contract. Checkpoints and validation completed at the same four steps.

## Training metrics

Across 512 completions, mean reward was 0.267188 (population SD 0.365675), canonical pass 19.1406%, format accuracy 54.6875%, canonical parseable rate 46.875%, and extracted-answer valid-component rate 54.8828%. The latter uses `formal_domain_valid_answer_component_v1`; it is intentionally not conflated with canonical parseability.

GRPO loss averaged -0.016647 and remained finite in [-0.1938, 0.2438]. Native TRL entropy (`entropy`) averaged 0.281434 nats, moved from 0.282285 to 0.261231, and ranged 0.195481–0.364780: it fluctuated without sustained collapse. It uses TRL's completion mask over generated response tokens through EOS, excludes prompt/PAD but not EOS, and is microbatch-then-log averaged. Unified `response_token_entropy_mean` and entropy std are null/unavailable because they cannot be reconstructed without extra or intrusive logits work.

Grad norm averaged 0.568892 and stayed finite in [0.213991, 1.556754]. Clip fraction was zero throughout. GRPO beta is 0, so KL is null/unavailable with that reason; ratio and ratio variance are also null/unavailable because TRL 0.24.0 did not expose them. Policy/value loss comparison is inapplicable: GRPO has no PPO value model, adapter, head, or value loss.

Mean EOS rate was 96.875%, truncation 3.125%, and exact-text duplicate rate 0.78125%. The official metrics include truncated completions. A non-truncated diagnostic is available in the truncation CSV but does not replace the official result.

## Reward-group learning signal

Of 128 four-completion groups, 101 (78.90625%) had nonzero reward variance. Twenty-seven groups were all-equal/zero-advantage and 15 were all-zero. Nonzero-variance fractions were 75% at update 1, 78.57% for updates 2–8, 81.25% for 9–16, 68.75% for 17–24, and 87.5% for 25–32. The run therefore had learning signal; it is not a `no_learning_signal` infrastructure-only success.

The late-window reward mean (updates 25–32) was 0.373047 versus 0.200446 for updates 2–8, but update-level reward and canonical pass remained volatile. This single seed does not prove generalized mathematical ability or rule out output-pattern effects.

## Frozen checkpoint validation

| Step | Pass@1 | Format | Parseable | Acc. given parseable | Tokens | Truncation |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 3.1250% | 14.0625% | 14.0625% | 22.22% | 7,159 | 9.375% |
| 16 | 4.6875% | 15.6250% | 14.0625% | 33.33% | 7,321 | 7.8125% |
| 24 | 6.2500% | 17.1875% | 14.0625% | 44.44% | 7,388 | 3.125% |
| 32 | 7.8125% | 21.8750% | 18.7500% | 41.67% | 7,245 | 3.125% |

Each checkpoint used the same 64 problems and one candidate per problem. Thus pass@4 is `null`, `available=false`, reason `validation_protocol_has_one_candidate_per_problem`. A matching base-model 64-problem validation does not exist, so base delta is also null/unavailable; the independent formal-test baseline was not subtracted or used for selection.

At step 32, GSM8K pass@1 was 9.375% (3/32); MATH was 6.25% (2/32). MATH Level 1–5 pass@1 was 0%, 0%, 7.69%, 20%, 0% on denominators 3/8/13/5/3. These small level cells support error inspection, not broad difficulty claims. Validation EOS is null/unavailable because the saved validation rows lack an EOS flag; it was not inferred from non-truncation.

## Resources, checkpoints, and warnings

Measured end-to-end wall time was 1,189.819 seconds, 0.330505 GPU-hours, and CNY 2.934886 at CNY 8.88/GPU-hour. Peak nvidia-smi VRAM was 11,209 MiB; mean/peak utilization was 37.17%/99%. PyTorch allocator summary is null/unavailable because the runtime file is empty.

All run checksums and the persistent backup verified. Checkpoints 8/16/24/32 each contain policy LoRA, optimizer, scheduler, Python/NumPy/PyTorch CPU/CUDA RNG, counters, prefixes, identities, and inventory SHA256. They contain no PPO value files or full base weights. Backup: `/root/autodl-fs/math-rlvr-backups/grpo_formal_1p5b_seed42_20260720T031006Z.tar.gz`, SHA256 `b584363595f99c1d3b61a7b6cc088cdda7ac38a29169058df7b30cd38bea5023`.

Two evidence warnings do not change the completed scientific result:

- run-root incremental JSONL was populated after `train()` returned, while trusted checkpoint prefixes preserved exact 8/16/24/32 evidence; recovery between checkpoint boundaries would restart from the last trusted checkpoint;
- `model_roles.json` captured `optimizer_exact_role_match=false` before lazy optimizer creation. The trusted checkpoint-32 optimizer contains 224 unique parameter IDs and 224 state entries, matching the 224 recorded LoRA trainable tensors; base/reward trainables are zero.

## Rebuildable figures

- [Training reward and verifier metrics](figures/grpo_training_reward_accuracy.png)
- [Policy diagnostics](figures/grpo_policy_diagnostics.png)
- [Reward-group learning signal](figures/grpo_reward_group_learning_signal.png)
- [Checkpoint validation curve](figures/grpo_checkpoint_validation_curve.png)
- [Validation status distribution](figures/grpo_validation_status_distribution.png)
- [Completion analysis](figures/grpo_completion_analysis.png)
- [GPU resources](figures/grpo_gpu_resources.png)

All figures are generated from committed CSV/JSON. Full completion text/token evidence and checkpoint payloads remain outside Git in the verified run and backup.
