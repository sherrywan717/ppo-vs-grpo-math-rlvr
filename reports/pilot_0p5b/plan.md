# Matched 0.5B PPO/GRPO pilot plan

`Matched 0.5B pilot - not the final benchmark`

## Stage D closure

Stage D technical smoke is complete. Accepted immutable evidence is PPO `ppo_single_update_qwen25_05b_20260714T051538Z` and GRPO `grpo_single_update_qwen25_05b_20260713T122258Z`; PPO's nullable `val/ratio_var` is a `nonessential_telemetry_warning`. Algorithm-effect comparison remains not established. Neither historical run may be rerun or rewritten. All checked-in Markdown is valid UTF-8; terminal/attachment mojibake, if any, is a display warning only.

## Purpose and boundary

The pilot tests whether PPO and GRPO can produce paired, credible artifacts under identical 0.5B model revision, prompt/reward/parser/verifier identities, sampling, policy LoRA, seeds, completion/token caps, and one-update budgets. It is not the final benchmark and cannot establish that either algorithm is better. Countdown ends after this pilot; a later 1.5B GSM8K+MATH stage needs separate design and authorization.

## Frozen data and identity

The ordered manifest selects the first four frozen Countdown train records: `countdown:train:0` through `countdown:train:3`. Selection is original-order only and uses no historical reward or model-quality signal. The manifest contains no gold answer or construction. Its SHA256 is `0235210e038bc27ebf2e7218691f36f09c8e11f0bbc743f46a5318a279f6bc1f`.

Both algorithms use Qwen 0.5B revision `7ae557604adf67be50417f59c2c2f167def9a775`, prompt `prompt_v1_strict_concise`, reward `shaped_v2_staged`, temperature 0.8, top-p 0.95, length 128, and q/k/v/o LoRA r16 alpha32 dropout0. Parser and Countdown verifier identities are canonical JSON contract hashes, independent of comments and Markdown.

## Matched budget

Per algorithm and seed: four unique prompts, four responses per prompt, 16 completions, 2,048 generated-token hard cap, one outer/optimizer/global update, zero retries, and one independent checkpoint. Across six runs: 96 completions and a 12,288-token hard cap. Reports must show both completion-matched and actual generated-token-normalized metrics.

PPO resolves to per-device batch 4 × GA4 = rollout batch 16, one epoch, one minibatch, four microbatches, local rollout forward batch 4, and no `num_generations` PPOConfig field. GRPO resolves to generation batch 16, four generations for each of four prompts, per-device batch 4, GA4, four steps per generation, and one iteration/update.

## Fixed run order

1. seed 42 — PPO
2. seed 42 — GRPO
3. seed 123 — GRPO
4. seed 123 — PPO
5. seed 2026 — PPO
6. seed 2026 — GRPO

Each is a new process from the fixed base snapshot, with no inherited checkpoint, its own run ID/checkpoint/full backup, and no retry. Only correctness, fairness, safety, or report-truthfulness failures stop the suite.

## Metrics and artifacts

Every run saves all 16 completion texts and token IDs, four problem groups, canonical/staged reward evidence, variance/zero-advantage, pass@1/pass@4, protocol validity, reward statistics, available losses/grad/entropy/KL, token/resource/cost evidence, and checkpoint inventory/SHA. Missing metrics are null/unavailable with reasons, never zero-filled. Three-seed aggregation reports raw values, mean, and standard deviation only; statistical-significance claims are forbidden.

Figures are generated only from saved CSV/JSON for reward, protocol validity, group variance, loss/grad, tokens, and resource/cost views.

## Resource estimate

Observed Stage D resources plus the parsed microbatch/value-model contracts yield a planning estimate of 39 seconds/7 GiB for PPO and 20 seconds/3.5 GiB for GRPO. Per-run 2× ceilings are 78 seconds/14 GiB and 40 seconds/7 GiB. Six-run expected total is 177 seconds, 0.04917 GPU-hours, and ¥0.4366; the planning ceiling is 354 seconds, 0.09833 GPU-hours, and ¥0.8732 at ¥8.88/GPU-hour. These are estimates; actual saved evidence and hard guards are authoritative.

## GPU-suite blockers

- TRL PPO 0.24.0 uses `DataLoader(shuffle=True)`. A reviewed sequential rollout/evidence path must prove the frozen prompt-major repetitions and `problem_id::generation_index` keys.
- Stage D guarded runtimes still enforce historical 4/8-completion shapes. They must be parameterized and fake-tested for 16 without changing the immutable historical paths.

The six resolved configs are accepted by exact path, seed, content, and SHA during CPU preflight. Real pilot execution remains fail-closed until these correctness blockers are removed in a separately reviewed CPU change and the user explicitly authorizes GPU execution. No additional confirmation flag is planned.

## Future command sequence (not authorized)

These commands are exact path/seed templates but remain blocked by the two runtime correctness items above and require a new explicit GPU authorization. Do not run them during configuration freeze.

```bash
PYTHONPATH=src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m math_rlvr.training.ppo --config configs/pilot/resolved/ppo_seed_42.json --execute --confirm-single-update
PYTHONPATH=src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m math_rlvr.training.grpo --config configs/pilot/resolved/grpo_seed_42.json --execute --confirm-single-update
PYTHONPATH=src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m math_rlvr.training.grpo --config configs/pilot/resolved/grpo_seed_123.json --execute --confirm-single-update
PYTHONPATH=src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m math_rlvr.training.ppo --config configs/pilot/resolved/ppo_seed_123.json --execute --confirm-single-update
PYTHONPATH=src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m math_rlvr.training.ppo --config configs/pilot/resolved/ppo_seed_2026.json --execute --confirm-single-update
PYTHONPATH=src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m math_rlvr.training.grpo --config configs/pilot/resolved/grpo_seed_2026.json --execute --confirm-single-update
```
