# Repository Guidelines

## Project Mission

This repository is the artifact-first Math RLVR project. Its goal is a reproducible PPO-versus-GRPO comparison for few-shot mathematical reasoning, with matched prompts, reward contracts, completion/token budgets, and auditable run artifacts. Formal experiments target `Qwen/Qwen2.5-1.5B-Instruct`; `Qwen/Qwen2.5-0.5B-Instruct` is the bounded smoke-test model. Countdown is verifier/smoke data, while GSM8K and MATH are training/evaluation data and MATH500 is held out from training.

Never advance to a new paid/GPU stage implicitly. Model download, CUDA initialization, model loading, generation, and PPO/GRPO updates each require the user's explicit scope. A successful stage does not authorize the next stage.

## Current Baseline and Milestones

The active branch is `pivot/math-rlvr`. Important milestones are:

- `5a10cbae2abcb066b423b10ff9d327ad1483b75c` — artifact-first Stage D infrastructure, frozen configs, reports, CPU gates, tokenizer audit, trainer builders, and shared PPO/GRPO prompt renderer.
- `6daca223bd17ddc9201e0b8dc7cdc3c677db9b39` — successful Qwen 0.5B CUDA/model-load sanity report.
- The local 0.5B snapshot is revision `7ae557604adf67be50417f59c2c2f167def9a775` under `/root/autodl-tmp/cache/huggingface`; never copy model weights into Git, run artifacts, backups, or `/root/autodl-fs`.
- CPU tokenizer audit, static gates, and CUDA load sanity have passed. No PPO update has been executed; the second bounded GRPO smoke executed exactly one optimizer update but remained a failed run.
- The guarded GRPO runner requires `--execute --confirm-single-update`; either missing flag fails before delayed model/CUDA imports. Exact completion tokens are guarded through the isolated TRL 0.24.0 shim.

Read `memory.md` before changing execution code or launching another run; it records measured results and known pitfalls.

## Project Structure

Keep application code in `src/math_rlvr/`, tests in `tests/`, reusable entry points in `scripts/`, configs in `configs/`, and Git-safe reports in `reports/`. Runtime datasets, caches, runs, outputs, and checkpoints belong only under `/root/autodl-tmp`. Full run artifacts use `/root/autodl-tmp/runs/math_rlvr/<run_id>/`; Git-safe summaries use `reports/runs/<run_id>/`; static backups use `/root/autodl-fs/math-rlvr-backups/` and must exclude model caches and weights.

The shared prompt contract is one system/user chat followed by an open assistant turn. The required completion envelope is exactly one `<reasoning>...</reasoning>` block followed by exactly one terminal `<answer>...</answer>` block. PPO and GRPO must use the same renderer from `math_rlvr.prompt`.

## Build, Test, and CPU Gates

No command in this section should load a model or initialize CUDA. Run from the repository root:

- `python -m compileall src scripts tests`
- `ruff check .`
- `pytest -q`
- `PYTHONPATH=src python scripts/check_env.py`
- `PYTHONPATH=src python scripts/validate_manifests.py`
- `PYTHONPATH=src python -m math_rlvr.training.ppo --config configs/smoke/ppo.yaml`
- `PYTHONPATH=src python -m math_rlvr.training.grpo --config configs/smoke/grpo.yaml`

The PPO/GRPO commands above are dry-run preflights only. They must not call `train`, `generate`, or `optimizer.step`, and must not load a model. Tests that inject a fake trainer must use the CPU configuration path so TRL does not probe BF16 GPU support.

## Experiment Contracts

Use BF16 LoRA, never QLoRA, vLLM, bitsandbytes, or newly downloaded dependencies unless separately approved. Policy LoRA is `r=16`, alpha 32, dropout 0 on q/k/v/o projections. The PPO value adapter is `r=8`, alpha 16 on q/v plus a trainable scalar score head. Keep prompt/completion limits and budgets in the checked-in configs, and align comparisons using actual completions and generated tokens rather than trainer steps.

The fixed formal reward is format 0.10, parse/semantic validity 0.10, and correctness 0.80. Infrastructure errors abort instead of becoming reward zero. Verifiers must not use `eval`, `exec`, dynamic imports, subprocess execution, or generated-code execution.

## Artifact and Safety Rules

Use `ArtifactManager` and `ResourceMonitor` for GPU runs. Every run must have bounded wall time, completion/token limits, explicit stop reasons, checksums, environment metadata without full environment dumps, resource CSV/JSONL, summaries, logs, and plots. On OOM, NaN/Inf, revision mismatch, target mismatch, timeout, or artifact failure, stop immediately and do not retry unless explicitly authorized.

Never commit model/cache files, full run archives, checkpoints, auth files, tokens, proxy credentials, complete environment dumps, or large binaries. Git-safe reports and small plots may be committed after secret, size, and `git diff --check` audits. Verify archives with both `tar -tzf` and `sha256sum -c`; exclude even empty checkpoint directories when the stage promises no checkpoints.

Do not batch-delete files or directories: `rm -rf`, recursive `rm`, and recursive `rmdir` are prohibited. If deletion is necessary, confirm and remove one explicit path. Do not execute generated code unless a verified isolation backend is active; ordinary subprocess execution is not a security sandbox.

## Coding and Review Style

Use Python 3.12, spaces, Ruff, descriptive `snake_case` functions/modules and `PascalCase` types. Keep modules focused and add deterministic tests for every behavior change. Preserve user changes and avoid unrelated cleanup. Use short imperative commit subjects, preferably Conventional Commits. Pull requests should state motivation, implementation, verification commands, configuration changes, artifact impact, and follow-up work.

### GRPO first-failure repair gate

The first real GRPO attempt, `grpo_single_update_qwen25_05b_20260713T050407Z`, remains a failed immutable historical run at commit `ebc926c432d6778c3b057a0b7b518f7f2eaea5ed`. Its trainer-construction failure was traced to comparing a validated local snapshot path against the original repository ID; artifact finalization separately attempted to serialize the BudgetGuard clock callable.

Execution code now uses the frozen `ValidatedModelSource` boundary: exact 0.5B repo/revision, canonical cache structure, local-only resolver equality, required files, and Qwen2 causal-LM config identity. `BudgetGuard.snapshot()` is the only counter serialization contract and contains primitive JSON values, never callbacks or runtime objects. These repairs do not authorize a real GRPO rerun or PPO.

### GRPO second-failure evidence repair

The second real GRPO attempt, `grpo_single_update_qwen25_05b_20260713T053852Z`, completed the frozen 8-completion/687-token/4-microstep/1-update budget but remained failure because the old checkpoint allowlist rejected the 7,441-byte Trainer metadata file `training_args.bin`. Preserve both failed runs and their backups as immutable evidence.

The repaired runner trusts only the Trainer-created top-level `checkpoint-1`, allows only a canonical non-symlink `training_args.bin` of at most 1 MiB, rejects duplicate adapters, and never deserializes metadata. The sole TRL shim owns ordered completion IDs/masks/text/reward evidence. Frozen beta is 0.0, so KL must be null with a reason. Allocator peaks and nvidia-smi remain distinct. This CPU repair is not authorization to rerun GRPO or start PPO.

### GRPO third-failure allocator repair

The third real GRPO attempt, `grpo_single_update_qwen25_05b_20260713T061248Z`, remains an immutable failed run at commit `438569d97a8636ea6ad13394920663016e01282e`. It stopped before model loading or generation because `CudaAllocatorEvidence` passed the literal string `"cuda:0"` to PyTorch allocator APIs; the resolved training config had no device field.

Allocator device handling is now centralized through `normalize_cuda_device_index`: accepted CUDA values are normalized to a validated non-boolean integer index, while callables, CPU devices, GPU display names, malformed strings, negative values, and out-of-range indices fail closed. Allocator API calls use only that integer; `device_label` and `device_name` are evidence fields only. CPU dry-runs never resolve an implicit current device. This repair does not authorize GRPO or PPO.

### CUDA allocator probe result

The one-shot minimal allocator probe `cuda_allocator_probe_20260713T063028Z` passed on H800 device index 0 after the normalization repair. A 1 MiB uint8 tensor produced 1/2 MiB current allocated/reserved and 1/2 MiB peak allocated/reserved; after release and `empty_cache`, current allocated/reserved returned to zero. The probe loaded no model/tokenizer/data and performed no generation, training, optimizer update, checkpoint, or network access. It does not authorize a GRPO rerun or PPO.

### Successful GRPO smoke and prompt forensic gate

The evidence-complete GRPO run `grpo_single_update_qwen25_05b_20260713T063829Z` succeeded at commit `85776a8290f736b0469f377b0a3d3c4b86cdc7a1`: 2 prompts, 8 completions, 687 generated tokens, 4 microsteps, and one optimizer/global step. All eight outputs were strict format errors, producing zero reward variance and no learning signal. Preserve the run, checkpoint, reports, and backup unchanged.

Historical/production prompt behavior remains versioned as `prompt_v0_grpo_smoke`.
After the separately authorized successful generation-only A/B diagnostic,
`prompt_v1_strict_concise` is approved only as the shared Qwen 0.5B PPO/GRPO smoke
prompt; its production status remains not approved. Both smoke paths must resolve the
same prompt version/hash/renderer and render the same `MathProblem` byte-identically.
Main/formal 1.5B configs remain unactivated. The strict parser and reward/verifier
contract are unchanged.

### Guarded prompt A/B implementation gate

The independent module `math_rlvr.evaluation.prompt_ab` accepts only
`configs/diagnostics/prompt_ab.yaml`. Real generation requires both dedicated flags,
a clean branch, both offline variables, the exact validated local Qwen 0.5B snapshot,
and the fixed 16-completion/2,048-token/120-second/3.5-GiB contract before delayed model
imports. It uses the base model only; Trainer, LoRA, train, backward, optimizer,

### Prompt A/B cleanup semantics

The immutable run `prompt_ab_qwen25_05b_20260713T101918Z` remains failure. Its worker allocator bytes were not persisted because close raised before returning evidence, but parent and manual post-run checks showed PID exit, no compute process and 0 MiB baseline/post memory. Future isolated-worker diagnostics treat allocator current memory as pre-exit evidence and warning only; the non-CUDA parent's post-exit PID/process/baseline check is authoritative. This does not activate v1 or authorize another A/B, GRPO or PPO run.

### Prompt v1 smoke-only activation

The successful rerun `prompt_ab_qwen25_05b_20260713T105428Z` produced v0: 8/8
FORMAT_ERROR and zero nonzero-variance groups; v1: 6 FORMAT_ERROR, 2
INVALID_EXPRESSION, 25% complete envelopes, and two nonzero-variance groups. v1 still
had zero valid-expression rate, number-usage accuracy, pass@1, and pass@4. Therefore
`prompt_v1_strict_concise` is frozen as `approved_for_smoke` and
`not_approved` for production. Only the 0.5B PPO/GRPO smoke selectors use it; v0 and
main/formal configs remain preserved. The next executable step requires a new explicit
GRPO single-update authorization. PPO remains unauthorized and must never start
automatically.
