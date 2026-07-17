# Stage E.1 CPU-only formal model-bound CLI wiring validation

Status: implementation and controlled CPU acceptance passed on 2026-07-17 UTC. This
stage created no scientific result and authorized no model download, CUDA/model-load
sanity, baseline generation, evaluation, PPO, or GRPO run.

## Implemented boundary

- PPO requires `--execute --confirm-formal-ppo`; GRPO requires
  `--execute --confirm-formal-grpo`; evaluation requires
  `--execute --confirm-formal-evaluation` plus explicit `base`, `ppo`, or `grpo` mode.
- Execute selection is bound to the exact repository-relative path, raw config SHA256,
  active-suite membership, validated `MAIN_FORMAL` scope, and the existing
  `ExpectedRunContract`. Absolute aliases, symlinked paths, unknown hashes, wrong
  algorithms, and both seed-2026 descriptors fail before snapshot/model handling.
- All real paths require both offline variables and the exact canonical local
  `Qwen/Qwen2.5-1.5B-Instruct` revision. A missing snapshot fails before CUDA or model
  load, with no network fallback.
- PPO reuses the existing distinct policy/value builders, policy/value LoRA plus scalar
  head optimizer-role audit, adapter-disabled reference semantics, parameter-free
  domain reward, ordered prompt-major loader, and the sole TRL 0.24.0 shim. PPO never
  receives `num_generations`.
- GRPO reuses the identical policy LoRA, parameter-free domain reward, exact four-way
  groups, policy-only optimizer audit, and adapter-only checkpoint contract.
- Baseline loads no adapter. PPO/GRPO evaluation validates a compliant checkpoint and
  selects only `policy_adapter`; PPO value adapter/head are never supplied to generation.
- Formal run artifacts now bind scope, ExpectedRunContract, prompt preflight,
  authorization, model roles, config/suite identity, exact counters, completions,
  metrics, checkpoint inventory, validation evidence, and nullable telemetry reasons.
  Per-problem evaluation names distinguish `sampled_pass_at_1` and `pass_at_4`.
  The frozen protocol has no separate greedy completion, so greedy accuracy is stored
  as `null`, `available=false`, with an explicit reason rather than fabricated as zero.

## Controlled CPU verification

- `compileall`, Ruff, `git diff --check`: passed.
- Full pytest: 436 passed with two TRL deprecation warnings and no failures.
- Environment and manifest checks: passed; `cuda_initialized=false` and
  `model_or_tokenizer_loaded=false`.
- Four active training dry-runs, one baseline dry-run, and one final-evaluation
  dry-runs passed.
- Fake PPO and GRPO finalized 32 updates, 512 exact completions, and checkpoint/
  validation steps 8/16/24/32. Fake baseline, 64-row validation, and 800-completion
  final evaluation finalized the existing schema.
- Fresh-process PPO and evaluation dry-runs imported none of Torch, Transformers, TRL,
  or PEFT.
- Frozen active-suite/config hashes remained unchanged. No historical 0.5B artifact
  was modified.

## Explicit execution accounting

For the controlled Stage E.1 acceptance gate: real Qwen model/tokenizer loads = 0,
generation calls = 0, real Qwen Trainer calls = 0, backward calls = 0, optimizer steps
= 0, and CUDA initialization = false.

During an intermediate regression selection and the final full pytest, the pre-existing
`test_guarded_trainer_shim_counts_real_backward_and_underlying_step` was inadvertently
included. It executed synthetic tiny CPU backward/optimizer work only: no Qwen
model or tokenizer, CUDA, generation, scientific run, or artifact. This is disclosed
rather than incorrectly claiming the entire working session had zero CPU backward.

Stage E.1 does not validate GPU memory, real model compatibility, throughput, or
scientific learning. Those require separately authorized future stages.
