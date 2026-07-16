# Matched 0.5B suite state reconciliation

`Matched 0.5B pilot - not the final benchmark`

Primary evidence resolves `ppo_matched_0p5b_seed42_20260716T114710Z` as
`execution_success/learning_signal_present`. The explicit launcher exit code was
not persisted and is unavailable, but the success manifest, finite metrics, exact
counters, safe checkpoint, verified Git-safe checksums, verified persistent archive,
and post-process GPU release satisfy the scientific-success contract.

| Position | Algorithm | Seed | Run ID | Scientific status | Completions | Optimizer/global | Checkpoint | Backup | Commit |
|---:|---|---:|---|---|---:|---|---|---|---|
| 1 | PPO | 42 | `ppo_matched_0p5b_seed42_20260716T114710Z` | execution_success/learning_signal_present | 16 | 1/1 | safe adapter/head checkpoint-1 | verified `dd1833ea…b7f6` | `8db7ce6` |
| 2 | GRPO | 42 | `grpo_matched_0p5b_seed42_20260716T115219Z` | execution_success/learning_signal_present | 16 | 1/1 | safe adapter checkpoint-1 | actual hash `6c31e369…2fb`; stale sidecar | `c048802` |
| 3 | GRPO | 123 | `grpo_matched_0p5b_seed123_20260716T115714Z` | execution_success/learning_signal_present | 16 | 1/1 | safe adapter checkpoint-1 | actual hash `63d4e598…6f3`; stale sidecar | `463881f` |
| 4 | PPO | 123 | `ppo_matched_0p5b_seed123_20260716T120000Z` | execution_success/learning_signal_present | 16 | 1/1 | safe adapter/head checkpoint-1 | actual hash `2ac5a29a…f9`; stale sidecar | `5489154` |

All four runs have exact 16 comparison keys and checksum-verified full/Git-safe
evidence. The stale backup sidecars are retained as warnings; the actual archive
hashes agree with committed assessments.

Historical engineering failures excluded from scientific aggregation:

- `ppo_matched_0p5b_seed42_20260714T073357Z`
- `ppo_matched_0p5b_seed42_20260714T082003Z`
- `ppo_matched_0p5b_seed42_20260714T085240Z`
- `ppo_matched_0p5b_seed42_20260716T111934Z`
