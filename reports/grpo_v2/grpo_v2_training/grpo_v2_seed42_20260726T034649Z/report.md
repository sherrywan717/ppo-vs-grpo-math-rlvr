# Stage R.2 GRPO-v2 seed-42 attempt

Run `grpo_v2_seed42_20260726T034649Z` is an immutable engineering failure before
training and is excluded from scientific analysis. The newly authorized command ran
exactly once from commit `999faa507fbca3bbb97e3bd37253e0f2a972f45b`; automatic
retries were zero.

## Passed preflight

The capacity-reconciled pre-model audit passed all 512 frozen curriculum prompts and
128 frozen dev prompts. Training/dev maxima were 918/453 tokens under the 928-token
prompt cap, `prompt + 256 <= 1,184` held for every row, truncation was zero, and hidden
test access was zero. Config and runtime-registry identities matched the frozen Stage
R.1 values.

## Failure boundary

The runtime verified the immutable warm-start checkpoint/adapter and recorded that no
SFT optimizer or scheduler state would be inherited. Trainer construction returned,
but its optimizer had not yet been created. The subsequent fresh-optimizer audit read
`trainer.optimizer.state`, producing:

```text
AttributeError: 'NoneType' object has no attribute 'state'
```

The failure is after Trainer construction and before `train()`, generation, backward,
an optimizer step, or fresh-optimizer parameter-set verification. Consequently the run
does not prove the GRPO fresh-optimizer assembly contract and contains no scientific
learning evidence.

## Counters and metrics

- Updates/microsteps/optimizer/global steps: 0/0/0/0
- Unique prompts/completions/generated tokens: 0/0/0
- Checkpoints/dev evaluations/dev completions: 0/0/0
- Hidden-test accesses: 0
- Reward, loss, entropy, grad norm, KL, ratio, clip, format, parseable and canonical
  metrics: `null`, unavailable because no update completed
- Worker resource-monitor wall time: 3.858843 seconds
- Peak nvidia-smi memory / mean utilization: 3,597 MiB / 1.8182%
- GPU-hours / CNY: 0.001071901 / ¥0.009518

## Artifact and safety status

All 15 raw run checksums passed. The verified failure archive contains 17 lightweight
entries, no checkpoint and no model weights:

`/root/autodl-fs/math-rlvr-backups/grpo_v2_seed42_20260726T034649Z.failure.tar.gz`

SHA256:
`7c4a7c367723c47c13d0b3d4f4810478196716f69a39f4c27761ef88a28d1f50`.

After worker exit, GPU memory returned to 0 MiB with no compute process. The prior
capacity-failure run remains separate and immutable. Neither failed run may be resumed,
combined, or included in scientific statistics.

The unique next blocker is a separately authorized CPU-only diagnosis and minimal
repair of the Trainer/fresh-optimizer initialization boundary. No further GPU attempt
is authorized by this report.
