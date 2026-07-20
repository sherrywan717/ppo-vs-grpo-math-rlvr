# Next task: Stage I formal GRPO seed 42

Status: PPO seed-42 is `scientifically_complete_with_recovered_validation`. Stage H.4
corrected prospective formal PPO/GRPO `valid_answer_rate` telemetry from the existing
flat `valid_answer_component` evidence. Targeted CPU tests and both formal dry-runs
passed; frozen scientific identities and historical artifacts did not change.

The only next task is a separately authorized formal GRPO seed-42 run using:

- config: `configs/formal_1p5b/resolved/grpo_seed_42.json`
- config SHA256: `3371d23166d01834c67830eb8dfd51a02d4af483687b0b29e941194174099199`
- model revision: `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`
- 32 updates, 512 training completions, 131,072 generated-token cap
- checkpoints and frozen 64-problem validation at steps 8/16/24/32

Before execution, require a new explicit GPU authorization, clean worktree, frozen
identity checks, local-only snapshot, offline mode, idle H800, and a non-conflicting new
run ID. Do not modify the PPO run or recovered validations. Do not start seed 123,
baseline, final test, PPO rerun/resume, or any later stage automatically.

This file authorizes no CUDA initialization, model loading, generation, or training.
