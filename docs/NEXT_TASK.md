# Next task: Stage J formal GRPO seed 123

Status: formal PPO seed 42 is scientifically complete through recovered validation, and
formal GRPO seed 42 completed successfully in
`grpo_formal_1p5b_seed42_20260720T031006Z`. The same-seed descriptive review is
complete; no final test ran and no configuration changed.

The only next task is a separately authorized formal GRPO seed-123 run using:

- config: `configs/formal_1p5b/resolved/grpo_seed_123.json`
- config SHA256: `cc95138f50f37fafa76766d3a08b0995ffd5e0bf87cd7b9050acedb5e0bbc75e`
- model revision: `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`
- 32 updates, 512 training completions, 131,072 generated-token cap
- checkpoints and frozen 64-problem validation at steps 8/16/24/32

Before execution require new explicit GPU authorization, clean worktree, frozen identity
checks, local-only/offline snapshot, idle H800, and a new non-conflicting run ID. Preserve
all PPO/GRPO seed-42 runs, checkpoints, reports and backups. Do not start PPO seed 123,
final test, baseline reruns, or any later stage automatically.

This file authorizes no CUDA initialization, model loading, generation, or training.
