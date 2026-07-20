# Next task: Stage L formal PPO seed-42 final evaluation

Status: all four active formal training runs and their frozen checkpoint validations are complete. PPO seed 123 succeeded as `ppo_formal_1p5b_seed123_20260720T043732Z`; the four-run training/validation aggregate is descriptive and no final test has run.

The only next task is the first separately authorized fixed step-32 final evaluation:

- algorithm: PPO
- seed: 42
- checkpoint source: trusted step-32 policy adapter linked by `ppo_formal_1p5b_seed42_composite_20260720T020928Z`
- evaluation config: `configs/formal_1p5b/evaluation.json`
- evaluation config canonical SHA256: `d8ba5ab80ab0553d2ec7246fb4876956dcbc5dd0bcf8642fd33c4ec19da6fe44`
- frozen protocol: 400 test problems plus the fixed 100-problem pass@4 subset, 800 completions
- checkpoint choice: fixed at step 32 before training; validation and test baseline cannot change it

Before execution require a new explicit GPU authorization, clean worktree, exact checkpoint/config/suite/model/prompt/reward/parser/verifier identities, local-only/offline snapshot, idle H800, and a new non-conflicting run ID. The exact executable command and one-attempt scope must be frozen in that authorization from the existing formal evaluation CLI; do not infer or run it from this handoff.

Preserve every training, validation, baseline, failure, checkpoint, and backup artifact. Do not run the other three final evaluations, CPU final aggregation, seed 2026, baseline reruns, or any training automatically.

This file authorizes no CUDA initialization, model loading, generation, evaluation, or training.
