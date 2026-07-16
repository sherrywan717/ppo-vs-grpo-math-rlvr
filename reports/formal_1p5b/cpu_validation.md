# Formal 1.5B CPU validation

Validation date: 2026-07-16 UTC. No 1.5B weights/tokenizer were downloaded or loaded;
CUDA remained uninitialized; no generation or real PPO/GRPO Trainer ran.

- `compileall`: passed.
- Ruff: passed.
- Pytest: 400 tests passed with three pre-existing real tiny-CPU backward/optimizer
  tests deselected to honor this turn's stricter no-backward rule. The same guard's
  fake 128-event/32-optimizer-group path passed.
- `check_env`: passed; `cuda_initialized=false`, `model_or_tokenizer_loaded=false`.
- Manifest validation: passed; 128 train, 64 validation, 400 test, zero frozen-contract
  drift.
- Active formal dry-runs: PPO42, GRPO42, GRPO123, PPO123 all passed without model load.
- Evaluation dry-runs: baseline seeds 42/123 and final PPO42/GRPO123 passed.
- Fake formal PPO/GRPO: 32 updates, 512 completions, checkpoint and validation steps
  8/16/24/32, final artifacts, same-run resume continuity, overflow failure, and
  failure backup passed.
- Six resolved descriptor files retain their Stage E SHA256 values; seed 2026 remains
  `reserved_not_scheduled` and cannot resolve to an active runtime contract.
- `git diff --check`: passed.

The three deselected tests are
`test_real_accelerate_cpu_ga4_updates_only_on_fourth_microbatch`,
`test_consumed_single_batch_disables_end_of_dataloader_early_sync`, and
`test_guarded_trainer_shim_counts_real_backward_and_underlying_step`. They are not
failures; they intentionally execute tiny CPU optimization, which this authorization
forbids.
