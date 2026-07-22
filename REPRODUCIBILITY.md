# Reproducibility

This document separates public reproduction from paid experiment authorization. Dry-runs and report rebuilding are CPU-only. Real model loading, generation, training, validation, or final evaluation requires an exact local snapshot and explicit authorization.

## Environment

- Python 3.12
- PyTorch/Transformers/TRL version ranges declared in `pyproject.toml`
- Model: `Qwen/Qwen2.5-1.5B-Instruct`
- Revision: `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`
- Formal configs: `configs/formal_1p5b/`
- Active suite canonical SHA256: `1d7c29f76d9bfbf11e1838cd6b0bc8f3da6d0133e0605420e9ed838de729d600`
- Active suite raw SHA256: `11869c63f4365aee5d4bf8e13fe263c9d0397164a18a88b419da07218f6a2017`

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
PYTHONPATH=src python scripts/check_env.py
PYTHONPATH=src python scripts/validate_manifests.py
```

The manifests record deterministic selections from GSM8K, MATH, and MATH500. Dataset/cache paths are machine-local and intentionally not committed. Resolve them outside the repository and verify the checked-in manifest hashes before any experiment.

## CPU-only preflight

```bash
PYTHONPATH=src python -m math_rlvr.training.ppo --config configs/formal_1p5b/resolved/ppo_seed_42.json
PYTHONPATH=src python -m math_rlvr.training.grpo --config configs/formal_1p5b/resolved/grpo_seed_42.json
```

These commands omit dual confirmation and must not load a model or initialize CUDA.

## Formal execution templates

See the README for training, checkpoint-validation, final-evaluation, and trusted same-run resume templates. Do not substitute external checkpoints. All real runs use `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `local_files_only=true`, a clean branch, a fresh run directory, and zero automatic retries.

## Rebuild reports and figures

```bash
python scripts/build_portfolio_v1.py
```

The script reads only committed files under `reports/formal_1p5b/metrics/`. It writes ten PNGs under `reports/formal_1p5b/figures/`. The release manifest records source relationships and SHA256 values.

## Audit invariants

- Training: 32 optimizer/global updates and 512 rollout completions per run.
- Training tokens: independently counted and capped at 131,072.
- Validation: 64 single-candidate completions at checkpoints 8/16/24/32; excluded from the training token budget.
- Final: 400×1 pass@1 candidates plus an independent 100×4 pass@4 pool; 800 completions total.
- Missing metrics: JSON null / unavailable / explicit reason, never zero substitution.
- Checkpoints: adapters, scalar head where applicable, and trusted resume state; never full base weights.
- Test: fixed checkpoint-32 only; no tuning or checkpoint selection.

## Public/private boundary

Git contains code, configs, tests, summaries, per-problem/candidate evidence, small images, and checksum manifests. Model caches, full checkpoints, optimizer state, credentials, complete environment dumps, and large run archives remain outside Git. See [release/remote_artifacts.md](release/remote_artifacts.md).
