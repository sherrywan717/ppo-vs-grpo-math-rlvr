# Next task: repair incremental formal checkpoint cadence

Status: second formal PPO seed-42 attempt preserved complete training evidence but
failed before checkpoint validation. CPU repair is not yet authorized.

The only real blocker is in `CompletedTrainerBackend.execute`: PPO incrementally
persisted observer state through update 32, then the post-training replay loop began
the scheduled checkpoint sequence at step 8. `FormalProgressGuard.record_checkpoint`
correctly rejected checkpoint 8 because the observer's current update was already 32.

The next task is a bounded CPU-only repair that keeps checkpoint and validation events
aligned with their corresponding incremental update. It must preserve the existing
checkpoint format, frozen identities, evidence order, budgets, and validation protocol;
it must not add guards, schemas, metrics, fallbacks, or unrelated tests. Direct fake
coverage should prove checkpoint 8 is recorded at update 8, validation follows the
checkpoint, and 32 completed updates do not replay checkpoint 8 against update 32.

Immutable failed run `ppo_formal_1p5b_seed42_20260719T131800Z` remains excluded from
scientific aggregation. Its four checkpoint directories are not authorized for resume
or evaluation. No PPO rerun, CUDA/model load, generation, checkpoint evaluation,
GRPO, seed 123, baseline, validation, or final test is authorized by this file.
