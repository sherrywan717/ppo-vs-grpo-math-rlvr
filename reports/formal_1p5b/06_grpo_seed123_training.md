# Formal GRPO seed 123 training and checkpoint validation

Status: `scientific_success`. Run `grpo_formal_1p5b_seed123_20260720T035927Z` executed the frozen command once from commit `e54d84d9795ad74da855e6fdf6e8a15700d36d1d`, with zero retries. No PPO, seed 2026, baseline, or final-test command ran.

## Contract and independent ledgers

| Ledger | Problems/pairs | Completions | Generated tokens | Training budget? |
|---|---:|---:|---:|---|
| Training rollout | 128 unique prompts | 512 | 52,284 | yes; cap 131,072 |
| Checkpoint validation | 256 checkpoint/problem pairs | 256 | 27,513 | no |
| Final test | not run | 0 | unavailable | separate and unauthorized |

Training completed 32/32 update, optimizer, and global steps. Trusted checkpoint resume counters record 32/64/96/128 microsteps at steps 8/16/24/32, matching the frozen 128-microstep contract.

## Training metrics

Across 512 completions, mean reward was 0.244238 (population SD 0.351672), canonical pass 16.9922%, format accuracy 51.5625%, canonical parseable rate 44.1406%, and valid-answer component rate 51.9531%. The last metric is `formal_domain_valid_answer_component_v1`, not canonical parseability.

GRPO loss averaged -0.019922 and stayed finite in [-0.1356, 0.1211]. Native TRL entropy (`entropy`) averaged 0.264968 nats and moved 0.286388→0.244270, with range 0.186447–0.336777. It uses TRL's completion mask through EOS, excludes prompt/PAD but not EOS, and is microbatch-then-log averaged. Unified `response_token_entropy_mean` and entropy std are null/unavailable because obtaining identical PPO/GRPO semantics would require additional or intrusive logits work.

Grad norm averaged 0.476887 and remained finite. Clip fraction was zero throughout. GRPO beta is zero, so KL is null/unavailable; ratio metrics were not exposed by TRL 0.24.0. GRPO has no PPO value adapter/head or value loss. EOS was 97.0703%, truncation 2.9297%, and mean exact-text duplicate rate 0.1953%. Training statuses were 248 format errors, 38 parse errors, 139 parseable wrong answers, and 87 canonical passes. All 15 truncated training completions were format failures, but 233 additional format failures were not truncated; truncation therefore explains only a small subset of formatting failure and is not labeled as mathematical inability.

## Reward-group learning signal

Of 128 four-completion groups, 100 (78.1250%) had nonzero reward variance. 28 were all-equal/zero-advantage and 16 were all-zero. Nonzero-variance fractions were 50% at update 1, 82.14% for updates 2–8, 75% for 9–16, 71.875% for 17–24, and 87.5% for 25–32. Corresponding reward means were 0.196875, 0.216518, 0.158203, 0.235547, and 0.369141. The run therefore had learning signal and is not a `no_learning_signal` execution; the late increase remains a single-seed on-policy observation.

## Frozen checkpoint validation

| Step | Pass@1 | Format | Parseable | Acc. given parseable | Tokens | Truncation |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 4.6875% | 14.0625% | 14.0625% | 33.33% | 6,590 | 7.8125% |
| 16 | 7.8125% | 20.3125% | 18.7500% | 41.67% | 7,089 | 4.6875% |
| 24 | 9.3750% | 23.4375% | 21.8750% | 42.86% | 6,938 | 3.1250% |
| 32 | 9.3750% | 23.4375% | 21.8750% | 42.86% | 6,896 | 7.8125% |


Each checkpoint used the same frozen 64 problems and one candidate per problem. Thus pass@4 is `null`, `available=false`, reason `validation_protocol_has_one_candidate_per_problem`. No matching base-model 64-problem validation exists, so base delta is also null/unavailable. Formal test was not run or used for selection.

At step 32, GSM8K pass@1 was 12.5% (4/32) and MATH was 6.25% (2/32). MATH Level 1–5 pass@1 was 0.00%, 0.00%, 7.69%, 20.00%, 0.00% on denominators 3/8/13/5/3. These small cells support diagnostics, not broad difficulty claims. Validation EOS is null/unavailable because rows do not persist an EOS flag.

## Resources, checkpoints, and warnings

Measured wall time was 1113.959 seconds, 0.309433 GPU-hours, and CNY 2.747765 at CNY 8.88/GPU-hour. Peak nvidia-smi VRAM was 8741 MiB; mean utilization was 37.58%. PyTorch peak allocated/reserved was 6054.4/8060.0 MiB.

Checkpoints 8/16/24/32 contain policy LoRA, optimizer, scheduler, Python/NumPy/PyTorch CPU/CUDA RNG, counters, prefixes, identities and inventory SHA256. They contain no PPO value files or full base weights. The verified backup is `/root/autodl-fs/math-rlvr-backups/grpo_formal_1p5b_seed123_20260720T035927Z.tar.gz`, SHA256 `e78eb0719bc93c1076bd06e50037cc453cbaa5103cf1e1fbfc9e8151212e521a`.

Warnings: run-root JSONL is finalized after `train()` returns while checkpoint prefixes preserve 8/16/24/32; the model-role snapshot precedes lazy optimizer creation; pre-exit allocator residue is non-authoritative. After inventory/hash validation, checkpoint-32's project-created optimizer was read with `weights_only=True`: 224 unique parameter IDs and 224 state entries matched all 224 policy LoRA trainable tensors, with zero base/reward trainables. Parent post-exit verification showed 0 MiB and no compute process.

## Rebuildable figures

- [Training reward and verifier metrics](figures/grpo_seed123_training_reward_accuracy.png)
- [Reward-group learning signal](figures/grpo_seed123_reward_group_learning_signal.png)
- [Checkpoint validation curve](figures/grpo_seed123_checkpoint_validation_curve.png)
- [Validation status distribution](figures/grpo_seed123_validation_status_distribution.png)
- [Completion analysis](figures/grpo_seed123_completion_analysis.png)
- [GPU resources](figures/grpo_seed123_gpu_resources.png)

All figures are generated from committed CSV/JSON. Full completion/token evidence and checkpoint payloads remain outside Git in the verified run and backup.
