# Math RLVR Stage E handoff

## Purpose and authority

This repository is a portfolio-grade, artifact-first comparison of PPO and GRPO for
verifiable mathematical-reasoning post-training with Qwen 1.5B. The goal is a
reproducible engineering and scientific result, not a trainer demo.

Use this authority order when sources disagree:

1. Git commits, actual configs/manifests, and saved run artifacts
2. `PROJECT_HANDOFF.md`
3. `docs/NEXT_TASK.md`
4. `AGENTS.md` and `memory.md`
5. Historical chat content

Git and original artifacts always win. Never rewrite historical artifacts to make a
derived document consistent.

## Required reading

Read these files before continuing Stage E:

1. `PROJECT_HANDOFF.md`
2. `docs/NEXT_TASK.md`
3. `docs/PORTFOLIO_DELIVERABLES.md`
4. `AGENTS.md`
5. `memory.md`

## Verified repository state

- Branch: `pivot/math-rlvr`
- Code/runtime baseline HEAD: `8ab031e567f877d48af75adb0ea5a6fba9e8bf55`
- Worktree at handoff start: clean
- This handoff is documentation-only; it changes no frozen config or runtime code.

## Completed stages

- Math RLVR pivot to an artifact-first PPO-versus-GRPO comparison.
- 0.5B Stage D PPO and GRPO technical smokes.
- Matched 0.5B pilot with six valid single-update runs: PPO/GRPO at seeds 42, 123,
  and 2026.
- Stage E formal 1.5B experiment and data/reward/evaluation freeze.
- Four-run formal budget amendment, preserving the original six-run decision as
  history.
- CPU-only formal multi-update and evaluation runtime contract with fake 32-step
  finalization.

## What the 0.5B pilot established

The matched pilot established that the execution, comparison-key, checkpoint,
artifact, resource, and cost paths work across both algorithms. All six valid runs
completed their frozen one-update contracts. Canonical pass@1/pass@4 was zero for
every run. It therefore does not demonstrate learning, statistical significance, or
PPO/GRPO superiority and is not the final benchmark.

## Frozen 1.5B scientific identity

- Model: `Qwen/Qwen2.5-1.5B-Instruct`
- Revision: `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`
- Prompt: `prompt_v2_formal_math`
- Prompt SHA256: `89e459da827474d9bcc66e4407b06b5f8a968ce10d0be92e830c59fd9830a994`
- Reward: `shaped_v3_domain`
- Reward SHA256: `b9eda9520bb0271e28f6c209db85a408cdc0a65c2d403871b2b0fcc06e06a463`
- Parser: `strict_completion_parser_v1`
- Parser SHA256: `655c30f20c677ead5728b402a1b6d5a4d4cefe54e4c1b34abebdafe41f3ba0ad`
- Verifier bundle: `gsm8k_math_domain_router_v1`
- Verifier SHA256: `ac3603158e31c8603c21e5d33445745bb56f3ccf946b055db9544a3dbc5886fd`
- GSM8K verifier SHA256: `91f9de474df89f63cd208a5621fbb7a678dadbe73e4a3f3426afb5f59fbe4b50`
- MATH verifier SHA256: `0a4fb547959d1edc3c157392fb49f209a140981757f39a0c25c454868e8aefa7`

The model is BF16 with identical policy LoRA for PPO/GRPO: r16, alpha32, dropout0,
q/k/v/o. PPO's separate value base, value LoRA, and scalar head are an explicit
algorithm-required difference.

## Active four-run suite

- Canonical active-suite SHA256:
  `f6de8c555a70837d08c1e34e13a738a21ce247b3a09531df6244a2f1d3ef53bd`
- Active-suite file: `configs/formal_1p5b/active_suite.json`
- Active-suite file SHA256:
  `a78df532c2d31a11a63790993d9ce2b1425844c46d5013fec6820a3609dffc49`

Frozen execution order:

1. PPO seed 42 — `configs/formal_1p5b/resolved/ppo_seed_42.json` —
   `717502aa665e9d5ef967e04a5ab27aa53329ccb061bda228db3c715f4dab967b`
2. GRPO seed 42 — `configs/formal_1p5b/resolved/grpo_seed_42.json` —
   `6776f8894e9ac725a39748b06b57b62782cea2dab61faf51fd3cc3ceb5ae58bf`
3. Seed-42 validation and learning-signal review
4. GRPO seed 123 — `configs/formal_1p5b/resolved/grpo_seed_123.json` —
   `4ce0918f7284220c36555b9f23db181354168ebe252d7244ac3ac9587be236fa`
5. PPO seed 123 — `configs/formal_1p5b/resolved/ppo_seed_123.json` —
   `a68524e85e427e335abf6447aa2cc391686fd3aa4da6d42efb0e522beec1a0b3`
6. Frozen final evaluation of all four step-32 checkpoints
7. CPU-only aggregation

Seed 2026 configs remain byte-preserved as `reserved_not_scheduled`. They are not in
the active registry, execution queue, costs, evaluations, or final statistics.

## Training and evaluation contracts

Per active training run:

- 128 ordered training problems; 32 updates
- 4 unique prompts x 4 responses per update
- 16 completions per update; exactly 512 completions per run
- Maximum completion length 256; generated-token cap 131,072
- 32 optimizer steps and 32 global steps
- Checkpoint and validation steps: 8, 16, 24, 32
- Automatic retries: 0

Evaluation quantities:

- Baseline: two untrained-model seeds x 800 completions = 1,600
- Validation: four runs x four checkpoints x 64 completions = 1,024
- Final: four step-32 checkpoints x 800 completions = 3,200
- Final test: GSM8K 200 + MATH500 200; fixed pass@4 subset 50 + 50
- Test results never select prompts, rewards, hyperparameters, or checkpoints.

## Current blocker and next stages

Stage E.1 CPU wiring is complete: exact active-config routing, dual confirmations,
pinned local-only snapshot validation, delayed PPO/GRPO/evaluation assembly, 32-step
runtime connection, portfolio metric availability, and trusted same-run resume are
implemented. Recovery state remains only in the full run and persistent backup;
Git-safe/final exports remain adapter/head-only and full Qwen weights are forbidden.
Final CPU gates, commits, and static backup are the remaining administration work.

Future stages, each separately authorized:

1. Download the pinned 1.5B snapshot.
2. Run CUDA/model-load sanity.
3. Run the two-seed untrained baseline.
4. Run PPO seed 42, then GRPO seed 42.
5. Review seed-42 validation and learning signal.
6. Run GRPO seed 123, then PPO seed 123.
7. Run frozen final evaluation for four checkpoints.
8. Aggregate CPU-only and write the final report.

## Storage and cost

- Formal configs/manifests: `configs/formal_1p5b/`
- Git-safe formal plans/results: `reports/formal_1p5b/`
- Git-safe run reports: `reports/runs/<run_id>/`
- Full runs: `/root/autodl-tmp/runs/math_rlvr/<run_id>/`
- Dataset manifests: `/root/autodl-tmp/datasets/math_rlvr/manifests/`
- Model cache: `/root/autodl-tmp/cache/huggingface/` (never back up or commit)
- Persistent backups: `/root/autodl-fs/math-rlvr-backups/`

Full Stage E planning estimate: 6.8767 expected / 14.0 ceiling GPU-hours and
CNY 61.06 expected / CNY 124.32 ceiling at CNY 8.88 per GPU-hour.

Only issues affecting training results, PPO/GRPO fairness, safety, recoverability, or
report truthfulness may block. Nullable noncritical telemetry, optional plots, PTY
capture, SVG whitespace, and CSV line-ending issues are warnings. All historical
runs, checkpoints, reports, checksums, and pilot aggregates are immutable.

This document intentionally contains no token, auth material, proxy value, full
environment dump, or credential.
