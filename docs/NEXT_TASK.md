# Next task: Stage K formal PPO seed 123

Status: formal GRPO seed 123 completed successfully in
`grpo_formal_1p5b_seed123_20260720T035927Z`. Formal PPO/GRPO seed 42 and both formal
GRPO seeds are now complete. The two-seed GRPO stability review is descriptive; no
final test ran and no configuration changed.

The only next task is a separately authorized formal PPO seed-123 run using:

- config: `configs/formal_1p5b/resolved/ppo_seed_123.json`
- config SHA256: `3d6cc1f30f7b72bfadb5191613298ac3f64a1ba3c699cc8d1e30ce147218c15e`
- model revision: `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`
- 32 updates, 512 training completions, 131,072 generated-token cap
- checkpoints and frozen 64-problem validation at steps 8/16/24/32

Before execution require new explicit GPU authorization, clean worktree, frozen identity
checks, local-only/offline snapshot, idle H800, and a new non-conflicting run ID. Preserve
all prior runs, checkpoints, reports and backups. Do not start final test, seed 2026,
baseline reruns, or any later stage automatically.

This file authorizes no CUDA initialization, model loading, generation, or training.
