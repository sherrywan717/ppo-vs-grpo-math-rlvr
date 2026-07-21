# Next task: Stage L3 formal GRPO seed-123 final evaluation

Status: all four active formal training/checkpoint-validation results are complete.
Fixed checkpoint-32 held-out final evaluations are complete for PPO seed 42 and GRPO
seed 42. Stage L2 run `grpo_final_formal_1p5b_seed42_20260721T034104Z` completed
800/800 candidates and is included in the scientific aggregate.

The only next task is a separately authorized fixed checkpoint-32 final evaluation:

- algorithm: GRPO
- seed: 123
- training run: `grpo_formal_1p5b_seed123_20260720T035927Z`
- checkpoint: trusted `checkpoint-32` policy adapter
- evaluation config: `configs/formal_1p5b/evaluation.json`
- evaluation config canonical SHA256:
  `d8ba5ab80ab0553d2ec7246fb4876956dcbc5dd0bcf8642fd33c4ec19da6fe44`
- frozen protocol: 400 test problems plus the independent fixed 100-problem pass@4
  pool, 800 completions
- checkpoint choice: fixed at step 32 before training; validation/test cannot change it

Before execution require a new explicit GPU authorization, clean worktree, exact
checkpoint/config/suite/model/prompt/reward/parser/verifier identities, local-only
snapshot, idle H800, and a new run ID. That authorization must freeze the exact command,
one-attempt scope, progress, failure, artifact, and backup contracts.

Preserve Base, PPO42, and GRPO42 final evidence and backups. Do not run PPO seed-123
final evaluation, CPU final aggregation, seed 2026, baseline, training, or any automatic
retry/next stage.

This file authorizes no CUDA initialization, model loading, generation, evaluation, or
training.
