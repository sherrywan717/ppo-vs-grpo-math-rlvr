# Stage H.1: optional PPO telemetry and incremental evidence repair

## Outcome

The bounded CPU-only repair is complete. It changes no formal scientific identity and
does not authorize a GPU run.

## Root cause and repair

The failed run's TRL 0.24.0 per-update rows exposed neither `grad_norm` nor
`train/grad_norm`. `_normal_metrics()` nevertheless used the required-number parser,
and `FormalProgressGuard` also listed aggregate grad norm as required. Aggregate,
policy, and value grad norms now use the existing availability contract: a finite
provided value and exact raw key are preserved; a missing key becomes JSON `null`,
`available=false`, an explicit reason, and `raw_metric_key=null`; NaN/Inf still fails
closed.

The real backend previously waited for `trainer.train()` to return before replaying all
updates into `FormalRuntimeObserver`. The PPO guarded trainer now invokes the existing
observer immediately after the per-update log row is appended. The observer validates
counters, ordered comparison keys, token IDs/masks/counts, rewards and metrics, then
atomically rewrites the existing `completions.jsonl` and `metrics.jsonl` prefixes using
`math_rlvr.artifacts.manager.atomic_text`. This occurs before Trainer invokes the
step-8/16/24/32 checkpoint callback. No artifact type or checkpoint format was added.

At step 8 the expected durable prefix is exactly eight metric rows and 128 ordered
completion rows. A fake checkpoint-callback failure preserves those rows and counters.
A successful callback continues to the existing trusted adapter/head plus optimizer,
scheduler, RNG, runtime/prefix and inventory checkpoint path; full base weights remain
forbidden.

## Verification

- 23 directly related pytest cases passed: grad norm present/missing/non-finite,
  guarded PPO update-callback order, 8-update/128-row failure persistence, 32-step fake
  finalization, and existing same-run resume.
- Ruff passed on all affected files.
- compileall passed on affected modules and tests.
- Formal PPO dry-run passed with `no training started`.
- Full pytest was intentionally not run under the experiment-first constraint.
- Historical failed-run checksums passed unchanged.
- Frozen suite/config/model/data/prompt/reward/parser/verifier identities are unchanged.
- CUDA/model/tokenizer/generation/Trainer/backward/optimizer activity was zero.

Machine-readable details: `ppo_optional_metrics_evidence_fix.json`.
