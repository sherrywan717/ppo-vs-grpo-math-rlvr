# Suite Recovery Assessment (CPU-only)

Reconciled from HEAD `54891542497a3b4e1c3317f0d7fe764ee34d0498` on `pivot/math-rlvr` using `evidence_source_precedence.md`.

The microbatch correction is already present in commit `3c88fb1` (`fix: count
PPO microbatches from backward events`). Its Accelerate 1.14.0 CPU evidence is
recorded in `reports/pilot_0p5b/accelerate_microbatch_semantics.{md,json}` and
its static backup sidecar verifies.

## Authoritative matched-suite state

| Position | Algorithm | Seed | Run ID | Attempted | Scientific status | Completions | Tokens | Optimizer/global | Checkpoint | Backup | Commit |
|---:|---|---:|---|---|---|---:|---:|---|---|---|---|
| 1 | PPO | 42 | `ppo_matched_0p5b_seed42_20260716T114710Z` | yes | execution_success/learning_signal_present | 16 | 574 | 1/1 | safe adapter/head checkpoint-1 | verified archive | `8db7ce6` |
| 2 | GRPO | 42 | `grpo_matched_0p5b_seed42_20260716T115219Z` | yes | execution_success/learning_signal_present | 16 | 545 | 1/1 | safe checkpoint-1 | verified archive | `c048802` |
| 3 | GRPO | 123 | `grpo_matched_0p5b_seed123_20260716T115714Z` | yes | execution_success/learning_signal_present | 16 | 724 | 1/1 | safe checkpoint-1 | verified archive | `463881f` |
| 4 | PPO | 123 | `ppo_matched_0p5b_seed123_20260716T120000Z` | yes | execution_success/learning_signal_present | 16 | 565 | 1/1 | adapter/head checkpoint-1 | archive exists; sidecar stale | `463881f` |
| 5 | PPO | 2026 | — | no | not executed | — | — | — | — | — | — |
| 6 | GRPO | 2026 | — | no | not executed | — | — | — | — | — | — |

The four protected historical PPO seed-42 failures (`073357Z`, `082003Z`, `085240Z`, and `111934Z`) remain excluded from any aggregate. The successful `114710Z` run is not one of those failures. The seed-123 PPO evidence is complete: four batch-4 backward events,
prompt-major 16 keys, finite required metrics, and post-run GPU release. Its
Git-safe artifact directory passes its own checksum manifest. The archive's
actual SHA256 is `2ac5a29a453e8e71bda7aea2d498e798b0f6d29393c20286bd0e585fe7f326f9`;
the pre-existing sidecar contains a stale hash and therefore does not verify.
No backup or historical artifact was changed.

Frozen config hashes, manifest identity, prompt/reward/parser/verifier hashes,
and the fixed suite order remain unchanged. Current GPU state was 0 MiB with no
compute process. Worktree changes are limited to the pre-existing documentation,
registry, and Git-safe seed-123 report; no source or frozen contract changed.

Next unexecuted command is the frozen PPO seed-2026 command. It was not run.
