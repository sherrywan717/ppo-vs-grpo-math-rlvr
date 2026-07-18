# Formal PPO seed 42 — failed attempt evidence

- Run ID: `ppo_formal_1p5b_seed42_20260718T150510Z`
- Status: engineering failure; excluded from scientific aggregation
- Failure: checkpoint-8 serialization required a Trainer `grad_norm` field that TRL did not expose.
- Retry count: 0; no GRPO, seed-123, validation, or final-test command ran.

## Truthful evidence boundary

The live launcher reached the displayed 8/32 progress boundary and showed `episode=128`, but the runtime failed before it appended completion, metric, or verifier primary evidence. The finalized JSONL files and formal counters are therefore 0 rows/0 updates. Generated-token count is `null/unavailable`, not zero. The live display cannot support scientific reward, loss, entropy, completion, or token claims.

No checkpoint validation ran. The formal seed-42 test baseline remains a separate 800-completion test result and is not a validation delta.

## Partial checkpoint

`checkpoint-8` contains only policy/value adapter files and the scalar head. It lacks optimizer, scheduler, RNG, runtime/counter, comparison-prefix, and trusted inventory state. It is not a valid checkpoint and must never be resumed or evaluated. No full base-model weights are present. See `checkpoint_partial_inventory.csv`.

## Resources and release

- Wall time: 181.951470 s
- GPU-hours: 0.050542075
- Cost at CNY 8.88/GPU-hour: CNY 0.448814
- Peak nvidia-smi VRAM: 28,655 MiB
- Mean GPU utilization: 36.0585%
- Peak PyTorch allocated/reserved: 25,277.702 / 27,978 MiB
- After process exit: 0 MiB and no compute process
- Verified failure backup: `/root/autodl-fs/math-rlvr-backups/ppo_formal_1p5b_seed42_20260718T150510Z.failure.tar.gz`
- Backup SHA256: `76896c5b3db3ee4439566b8b68c0cad798af5b5610f393138aa23eba6c40debb`

## Unique blocker

Before any new GPU authorization, a bounded CPU-only repair must make missing optional `grad_norm` serialize as `null`/`available=false` with a reason, and must prove per-update completion/metric evidence is persisted before checkpoint serialization can fail. This failed run and partial checkpoint stay immutable.
