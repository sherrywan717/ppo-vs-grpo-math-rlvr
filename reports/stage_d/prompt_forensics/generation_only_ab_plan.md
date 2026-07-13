# Generation-only v0/v1 A/B diagnostic — guarded runner implemented, not executed

The CPU-only guarded runner is `python -m math_rlvr.evaluation.prompt_ab` and accepts
only `configs/diagnostics/prompt_ab.yaml`. Dry-run uses the config alone. Real generation
requires independent `--generate-only --confirm-prompt-diagnostic`; the GRPO
`--confirm-single-update` flag is invalid. A clean branch, both offline flags, the exact
validated local Qwen 0.5B snapshot, and the full immutable budget must pass before the
delayed runtime module imports CUDA or Transformers model code.

A is `prompt_v0_grpo_smoke`; B is the unactivated
`prompt_v1_strict_concise`. Both use the same two Countdown train problem IDs, base BF16
model without adapter, seed 42, temperature 0.8, top-p 0.95, top-k inherited from the pinned local model generation config, repetition
penalty 1.0, and four completions per problem. Matched seeds 42–49 are reset separately
for Python, PyTorch CPU, and PyTorch CUDA for each condition, so v0 RNG consumption cannot
affect v1.

The immutable cap is 16 completions, 128 tokens each, at most 2,048 generated tokens,
120 seconds, and 3.5 GiB nvidia-smi process memory. The runner asserts `eval()`, frozen
parameters, `torch.inference_mode()`, and zero Trainer/train/backward/optimizer/training
step/checkpoint/model writes. It never activates v1 or retries.

v1 may only enter a later GRPO review when its complete-envelope rate exceeds v0, it has
at least one complete envelope, truncation does not increase, and at least one problem
group has nonzero reward variance. All-WRONG_ANSWER output is explicitly a possible
no-advantage result, not evidence of learning.

Expected/worst wall time remains 40/120 seconds. Expected peak/stop-gate VRAM is
2.5/3.5 GiB. At CNY 8.88/GPU-hour, expected/worst cost is CNY 0.099/CNY 0.296. This plan
has not been executed and is not a statistically significant experiment.


Before any separately authorized execution, the versioned evidence capability manifest
must pass all seven fields. A non-CUDA parent spawns one fixed worker; the worker records
paired and per-problem evidence plus PyTorch allocator cleanup, and the parent verifies
process exit/nvidia-smi absence/baseline memory restoration before cross-file validation,
success-or-failure backup, and publication. This adds evidence gates only: prompts,
problems, seeds 42–49, sampling, 16/2,048 budget, 120 seconds, and 3.5 GiB remain frozen.
