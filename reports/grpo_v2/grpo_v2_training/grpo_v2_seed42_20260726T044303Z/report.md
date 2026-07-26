# GRPO-v2 seed-42 training result

Run `grpo_v2_seed42_20260726T044303Z` completed the frozen scientific training and matched-dev contract.
The launcher later remained blocked returning the already-finalized oversized result
through its multiprocessing queue. The completed worker and stuck parent were
terminated without retry. This is disclosed as
`launcher_ipc_manual_termination_after_scientific_finalization`, not as a normal
launcher exit and not as a training failure.

## Contract

- Training: 512/512 unique curriculum prompts, 4 completions each, 2,048/2,048 completions.
- Updates / optimizer / global / microsteps: 128 / 128 / 128 / 512.
- Exact rollout tokens: 230,675 / 524,288. Dev used a separate 53,609 tokens.
- Checkpoints and dev: 32, 64, 96 and 128; every dev run has 128 single-candidate problems.
- Hidden-test accesses: 0.

The warm-start policy adapter SHA is
`44066dd13d8cfa4f5c40f10cad705eea617c37ce2e2f85ff5407751fb5a972b9`. GRPO created a fresh
optimizer and scheduler: no SFT optimizer state was loaded, the pre-update optimizer
state had zero entries, and 224 AdamW state entries materialized after the first update.
The optimizer parameter union exactly matched the 4,358,144 policy-LoRA trainables;
base and reward-model trainables were zero.

## Training evidence

- Mean scalar reward: 0.390063; population SD 0.384210.
- Canonical pass / format / valid-answer / parseable:
  27.8809% / 84.3750% /
  79.9805% / 75.4395%.
- EOS / truncation: 95.7520% / 4.2480%.
- GRPO loss mean 0.012509, range
  [-0.1859, 0.2233].
- Entropy mean 0.262211 nats from raw key
  `entropy`; grad norm mean 0.418964.
- Reward groups: 512 total; 367 nonzero-variance,
  145 zero-variance/zero-advantage,
  6 all-zero and 145 all-equal.
- KL is null/unavailable because beta=0.0. Ratio and ratio variance are
  null/unavailable because the reviewed TRL 0.24.0 update rows did not expose them.
  Clip fraction is available from `clip_ratio/region_mean` and its run mean is 0.

## Frozen dev selection

| checkpoint | correct | pass@1 | parseable | format | truncation | tokens |
|---:|---:|---:|---:|---:|---:|---:|
| 32 | 23/128 | 17.9688% | 44.5312% | 50.0000% | 5.4688% | 13,679 |
| 64 | 27/128 | 21.0938% | 53.9062% | 62.5000% | 3.9062% | 13,613 |
| 96 | 33/128 | 25.7812% | 61.7188% | 72.6562% | 3.1250% | 13,418 |
| 128 | 28/128 | 21.8750% | 65.6250% | 75.0000% | 3.9062% | 12,899 |


The preregistered lexicographic rule selected **checkpoint-96**: it has the highest
canonical pass@1 (33/128, 25.78125%). Checkpoint-128 improved parseability and format
but fell to 28/128, so it does not displace checkpoint-96. Dev has one candidate per
problem; pass@4 and pass@10 are null/unavailable by protocol.

## Resources and artifacts

Training resource telemetry records 0.295499
GPU-hours, CNY 2.624028, peak
nvidia-smi VRAM 11247 MiB and mean
utilization 44.502%.
Per-checkpoint dev GPU telemetry was not persisted by the frozen runner, so dev-only
GPU-hours/cost are unavailable rather than zero. Independent post-release evidence
shows 0 MiB, 0% utilization and no compute process. End-to-end launch-to-release wall
time was 5,025 seconds (1.395833 hours); CNY12.395 at ¥8.88/hour includes the
post-finalization IPC wait and is not the training-only cost.

All four checkpoints are adapter-only (`base_weights_included=false`) and include
optimizer, scheduler, RNG, runtime/counter, curriculum cursor, completion/metric prefix
and SHA inventory. The complete run backup is
`/root/autodl-fs/math-rlvr-backups/grpo_v2_seed42_20260726T044303Z.tar.gz` with SHA `af88cd652ef1ff1a23ff34a728fafb24e62c55ac68b83472575dea90c3d2a6f2`. The non-overwriting post-release archive is `/root/autodl-fs/math-rlvr-backups/grpo_v2_seed42_20260726T044303Z.postrelease.tar.gz` with SHA `52ccacdccfd07259993ae3075301fdbb50ab00e628954871809ac8319e239fcb`.

Figures in [`figures/`](figures/) are reconstructed from `update_metrics.csv`,
`reward_group_statistics.csv`, `dev_checkpoint_metrics.csv`, and
`status_distribution.csv`; they are not primary evidence. Hidden test was not run.
