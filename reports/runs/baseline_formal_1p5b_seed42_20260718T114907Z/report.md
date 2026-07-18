# Failed Qwen 1.5B frozen baseline attempt

- Status: **FAILURE — no scientific baseline result**
- Run ID: `baseline_formal_1p5b_seed42_20260718T114907Z`
- Seed: `42`
- Command attempts: `1`; automatic retries: `0`
- Seed 123: **not executed**
- Base model only: true; adapter/LoRA loaded: false

## Failure and preserved evidence

The first planned completion reached `model.generate` and formal reward evaluation,
then failed before `rows.append` at:

```text
RealFormalEvaluationBackend.generate
  -> evaluation.to_dict()["components"]
  -> KeyError: 'components'
```

`RewardEvaluation.to_dict()` exposes flattened component fields such as
`answer_block_component`, `valid_answer_component`, and `verifier_detail`; it
does not expose a top-level `components` key. This is a real model-bound CLI wiring
error affecting result truthfulness and completion-data integrity, so the suite stopped
and no code was modified.

The run's persisted completion count is 0/800. Exactly one generation call completed
before the failing serialization statement, but its completion IDs, text, and token
count were never appended or saved. Generated tokens are therefore
`null/unavailable` with that reason; they must not be reported as zero.

## Scientific metrics

No greedy, sampled pass@1, pass@4, GSM8K, MATH500, level, format, validity, verifier,
length, EOS, or truncation result exists. All are unavailable because the run failed
before a completion row or aggregate metric was persisted. Independently, the frozen
protocol defines greedy accuracy as `null/unavailable` because it has no separate
greedy completion.

Trainer, adapter, LoRA, backward, optimizer, checkpoint, PPO, and GRPO counters are
all zero.

## Resource and release evidence

- Resource window: 6.312120 seconds
- Peak `nvidia-smi` memory: 3,801 MiB
- Peak/mean GPU utilization: 39% / 12%
- GPU-hours: 0.001753367
- Cost at CNY 8.88/hour: CNY 0.015570
- PyTorch allocator: unavailable because the exception bypassed the success-only
  persistence path
- Independent post-exit GPU check: 0 MiB, 0% utilization, no compute process

The resource figure is rebuilt only from the preserved `resource_metrics.csv`.

## Artifact and backup verification

All checksums in the original failure run passed. The verified persistent archive is:

`/root/autodl-fs/math-rlvr-backups/baseline_formal_1p5b_seed42_20260718T114907Z.failure.tar.gz`

SHA256:
`b32174ddea42dd458a86a20aa53948b8b56fcf838f996fced22cd2648a0bd6d4`

The archive contains 14 entries and no model weights/cache. This failure does not
authorize a retry, seed 123, baseline aggregation, PPO, or GRPO.
