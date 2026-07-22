# Experiment design

## Frozen identities

- Model: Qwen/Qwen2.5-1.5B-Instruct at `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`.
- Prompt: `prompt_v2_formal_math`.
- Reward: `shaped_v3_domain` (0.05 answer block, 0.05 strict protocol, 0.10 domain-valid answer, 0.80 canonical correctness).
- Policy LoRA: r16/alpha32/dropout0 on q/k/v/o.
- Sampling: temperature 0.8, top-p 0.95, max prompt 832, max completion 256.
- Seeds: active 42 and 123; 2026 remains reserved/not scheduled.

## Matched budget

Each PPO/GRPO run performs 32 optimizer/global updates, creates 512 training completions, and is guarded by a 131,072 training-token cap. Both checkpoint and validate at steps 8, 16, 24, and 32. Validation completions/tokens are accounted separately.

PPO uses one response for each rollout row, a policy adapter, a distinct value adapter, and scalar value head. GRPO uses four completions per prompt group and relative group advantages without a value head. PPO and GRPO loss values are therefore not compared directly.

## Evaluation

Checkpoint validation is 64 problems with one candidate each; pass@4 is unavailable by protocol. Formal test always uses fixed checkpoint-32, never validation-selected weights. Its sampled pass@1 pool is 400×1; its independent pass@4 pool is 100×4. The total is 800 candidates per model/seed.

## Status of the matrix

All four training/validation runs are complete. Base/PPO/GRPO final test is complete only for seed 42. GRPO123 and PPO123 final tests are `deferred_not_executed`; no value is inferred from validation.
