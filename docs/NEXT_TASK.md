# Next task: Stage L2 formal GRPO seed-42 final evaluation

Status: all four active formal training/checkpoint-validation results are complete, and
PPO seed-42 fixed checkpoint-32 held-out final evaluation succeeded as
`ppo_final_formal_1p5b_seed42_20260721T022152Z`. Its 800-row result is scientific; the
externally interrupted 429-row process `ppo_final_formal_1p5b_seed42_20260720T052931Z`
is immutable and excluded.

The only next task is a separately authorized fixed checkpoint-32 final evaluation:

- algorithm: GRPO
- seed: 42
- training run: `grpo_formal_1p5b_seed42_20260720T031006Z`
- checkpoint: trusted `checkpoint-32` policy adapter
- checkpoint artifact-manifest SHA256: `c0c1a3dc04b28d8d42463009b728b3ffd144e677c35e5d3ace2749fc925fa65a`
- evaluation config: `configs/formal_1p5b/evaluation.json`
- evaluation config canonical SHA256: `d8ba5ab80ab0553d2ec7246fb4876956dcbc5dd0bcf8642fd33c4ec19da6fe44`
- frozen protocol: 400 test problems plus the independent fixed 100-problem pass@4
  pool, 800 completions
- checkpoint choice: fixed at step 32 before training; validation/test cannot change it

Before execution require a new explicit GPU authorization, clean worktree, exact
checkpoint/config/suite/model/prompt/reward/parser/verifier identities, local-only
snapshot, idle H800, and a new run ID. The authorization must freeze the exact command,
one-attempt scope, progress, failure, artifact, and backup contracts.

Preserve all PPO42 final-evaluation evidence and both outage/success backups. Do not run
PPO/GRPO seed-123 final evaluation, CPU final aggregation, seed 2026, baseline, training,
or any automatic retry/next stage.

This file authorizes no CUDA initialization, model loading, generation, evaluation, or
training.
