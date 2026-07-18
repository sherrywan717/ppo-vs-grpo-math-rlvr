# Next task: Stage H — execute formal PPO seed 42

Status: technically ready, not yet authorized. This file identifies the only next
project task; it does not itself authorize CUDA, model loading, generation, backward,
or optimizer execution.

## Goal

Execute exactly one frozen formal PPO seed-42 run through the existing Stage E.1
model-bound CLI. Do not redesign infrastructure, change scientific identities, run
GRPO, evaluate a checkpoint, or continue automatically after success/failure.

## Frozen authorization identity

- Branch: `pivot/math-rlvr`
- Model/revision: `Qwen/Qwen2.5-1.5B-Instruct` /
  `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`
- Config: `configs/formal_1p5b/resolved/ppo_seed_42.json`
- Config raw SHA256:
  `1093e87a8363a0a2a6ab640a6f723c04cb6cfb22edef2e38a8c3a0062693ec43`
- Active-suite raw SHA256:
  `11869c63f4365aee5d4bf8e13fe263c9d0397164a18a88b419da07218f6a2017`
- Active-suite canonical SHA256:
  `1d7c29f76d9bfbf11e1838cd6b0bc8f3da6d0133e0605420e9ed838de729d600`
- Prompt/reward/parser/verifier/data and step-32 selection remain frozen.
- Shared `max_prompt_length=832`; `max_completion_length=256`.

## Exact command after explicit authorization

```bash
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
PYTHONPATH=src \
python -m math_rlvr.training.ppo \
  --config configs/formal_1p5b/resolved/ppo_seed_42.json \
  --execute \
  --confirm-formal-ppo
```

The runtime creates `/root/autodl-tmp/runs/math_rlvr/ppo_formal_1p5b_seed42_<UTC>/`.
A new run starts without `--resume-checkpoint`. Resume is allowed only after a later
explicit authorization and only from the same run's trusted checkpoint 8/16/24.

## Preflight before the single attempt

Verify branch, expected HEAD, clean worktree, exact suite/config hashes, pinned
canonical local snapshot, both offline variables, idle H800/no compute process,
writable storage, disk capacity, and absence of credentials in output. Do not download,
upgrade dependencies, or use network fallback. Automatic retries are zero.

## Frozen run contract

- 128 ordered training problems; 32 updates.
- Four prompts × four responses per update; exactly 512 completions.
- Actual generated-token hard cap 131,072.
- PPO rollout 16, microbatch 4, GA4, one PPO epoch and one minibatch per update.
- Exactly 32 optimizer/global steps.
- Checkpoint and validation milestones 8/16/24/32.
- Policy LoRA r16/alpha32/dropout0 on q/k/v/o.
- Separate value base with q/v r8 adapter and scalar head.
- Frozen adapter-disabled reference; parameter-free canonical reward/verifiers.
- Git-safe/final checkpoint contains adapters/head only; full base weights are forbidden.
- Training recovery state remains only in the run and persistent backup and must pass
  same-run path, filename, size, manifest hash, identity, and counter-prefix checks.

## Evidence that must be preserved

Follow `docs/PORTFOLIO_DELIVERABLES.md`. Save per-update PPO policy/value/total loss,
reward, advantage/return availability, KL, clip fraction, ratio, entropy definition and
availability, grad norms, LR, completion length/EOS/truncation/diversity, generated and
cumulative tokens, validation metrics, VRAM/time/GPU-hours/CNY, and every completion
with reward/parser/verifier evidence. Do not add a model forward solely for optional
entropy. Missing/unreliable values remain null/unavailable with exact reasons, never 0.

Test data is forbidden for tuning or checkpoint selection. Step 32 remains fixed for
final evaluation regardless of validation results.

## Stop rules and handoff

OOM, NaN/Inf in required evidence, identity drift, unfair model-role assembly,
checkpoint/base-weight violation, counter/token mismatch, artifact failure, or GPU
safety failure stops the single attempt without retry. Optional telemetry and
presentation issues are warnings.

After the attempt, whether success or failure, release the GPU; verify artifacts and
backup; update `PROJECT_HANDOFF.md`, append `memory.md`, replace this file with the
single next task or blocker, update `reports/formal_1p5b/run_registry.csv`, and update
the concise README. Stop. A PPO success does not authorize GRPO seed 42.
