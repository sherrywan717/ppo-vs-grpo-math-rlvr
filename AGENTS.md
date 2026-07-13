# Repository Guidelines

## Project Mission

This repository is the artifact-first Math RLVR project. Its goal is a reproducible PPO-versus-GRPO comparison for few-shot mathematical reasoning, with matched prompts, reward contracts, completion/token budgets, and auditable run artifacts. Formal experiments target `Qwen/Qwen2.5-1.5B-Instruct`; `Qwen/Qwen2.5-0.5B-Instruct` is the bounded smoke-test model. Countdown is verifier/smoke data, while GSM8K and MATH are training/evaluation data and MATH500 is held out from training.

Never advance to a new paid/GPU stage implicitly. Model download, CUDA initialization, model loading, generation, and PPO/GRPO updates each require the user's explicit scope. A successful stage does not authorize the next stage.

## Current Baseline and Milestones

The active branch is `pivot/math-rlvr`. Important milestones are:

- `5a10cbae2abcb066b423b10ff9d327ad1483b75c` — artifact-first Stage D infrastructure, frozen configs, reports, CPU gates, tokenizer audit, trainer builders, and shared PPO/GRPO prompt renderer.
- `6daca223bd17ddc9201e0b8dc7cdc3c677db9b39` — successful Qwen 0.5B CUDA/model-load sanity report.
- The local 0.5B snapshot is revision `7ae557604adf67be50417f59c2c2f167def9a775` under `/root/autodl-tmp/cache/huggingface`; never copy model weights into Git, run artifacts, backups, or `/root/autodl-fs`.
- CPU tokenizer audit, static gates, and CUDA load sanity have passed. No PPO or GRPO optimizer update has been executed.

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
