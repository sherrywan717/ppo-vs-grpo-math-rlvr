# PPO vs GRPO for Few-Shot Math RLVR at 1.5B

A reproducible comparison of PPO and GRPO sample efficiency, stability, and generalization on `Qwen/Qwen2.5-1.5B-Instruct`. Formal training uses GSM8K and MATH Level 1–3; evaluation uses GSM8K test and MATH500. Countdown is smoke/verifier-only and is never a headline benchmark.

## Frozen design

Both algorithms use seed `20260712`, the same frozen manifests, prompt envelope, BF16 policy LoRA (`r=16`, alpha 32, dropout 0; q/k/v/o projections), four completions, temperature 0.8, top-p 0.95, prompt length 512, completion length 384, and the same verifier/reward policy. Comparisons align actual completions and generated tokens, not trainer steps.

Formal reward is fixed: format 0.10, parse/semantic validity 0.10, correctness 0.80. Thus correct is 1.00, parseable wrong is 0.20, format-correct parse failure is 0.10, and format error is 0.00. Infrastructure errors abort.

PPO uses a separate sequence-classification value model from the policy checkpoint, value LoRA `r=8`, alpha 16 on q/v projections, and a trainable scalar head. This phase defines and tests contracts only; no model is loaded and training remains disabled.

## Data and layout

Frozen manifests live under `/root/autodl-tmp/datasets/math_rlvr/manifests`; Hugging Face cache is `/root/autodl-tmp/cache/huggingface`. `src/math_rlvr/` contains schema, prompt, verifiers, rewards, rollout accounting, metrics, and preflight entry points. `src/math_rlvr/execution/` is retained only as legacy/out-of-scope history and is not imported by the math pipeline.

The output contract is exactly one `<reasoning>...</reasoning>` block followed by exactly one terminal `<answer>...</answer>` block. Only answer content is verified. Verifiers never call `eval`, `exec`, dynamic imports, subprocesses, or generated code.

## GRPO single-update smoke contract

The checked-in GRPO smoke YAML is the configuration source of truth: two unique prompts, four generations per prompt, generation batch 8, micro-batch 2, gradient accumulation 4, one iteration, one global/optimizer step, eight completions, and a 1,024 generated-token hard cap. TRL 0.24.0 must infer `steps_per_generation=4`; never configure both that field and `generation_batch_size`. This is an integration smoke contract, not a formal experiment result.

## Shared smoke prompt

The Qwen 0.5B PPO and GRPO smoke configs select the same frozen
`prompt_v1_strict_concise` renderer. Its candidate status is `approved_for_smoke`, but
its production status is `not_approved`: the matched generation-only diagnostic raised
complete-envelope compliance from 0% to 25% and created nonzero reward variance in both
Countdown groups, while valid-expression, number-usage, pass@1, and pass@4 remained 0.
`prompt_v0_grpo_smoke` remains unchanged for historical replay, and main/formal 1.5B
configs do not activate v1.

PPO and GRPO must resolve and report the same `prompt_version`, `prompt_sha256`, and
`renderer_version`; rendering the same `MathProblem` must be byte-identical. See
`docs/smoke-prompt-fairness.md`. This selector change authorizes no GPU execution and
does not authorize PPO.

## Guarded GRPO execution

The default GRPO CLI is dry-run only. `--execute` by itself still fails closed. The only real smoke path requires the frozen smoke config and both flags:

```bash
PYTHONPATH=src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m math_rlvr.training.grpo --config configs/smoke/grpo.yaml --execute --confirm-single-update
```

Before delayed model imports, the CLI requires a clean `pivot/math-rlvr` worktree, the fixed local revision, and the complete batching/budget contract. `trl_compat.py` is the sole TRL 0.24.0 private-hook shim and exact token accounting uses completion IDs/masks, not decode/re-tokenize. The artifact state is fail-closed: success requires complete artifacts, adapter-only checkpoint inventory, tar backup, and verified SHA256. It never advances to PPO automatically.

## CPU-only checks

```bash
python -m compileall src tests
ruff check .
pytest -q
python scripts/check_env.py
python scripts/validate_manifests.py
make smoke-ppo
make smoke-grpo
make main-ppo
make main-grpo
```

All four algorithm targets are static preflights and refuse to train. Model/tokenizer download and CUDA initialization are outside this phase.

## Metrics

Both runs record pass@1/pass@4; GSM8K, MATH500, and per-Level accuracy; format, parse, expression, and number-usage validity; reward, completions, generated tokens, completion length, wall time, KL, entropy, peak VRAM, GPU-hours, and CNY cost. PPO additionally reports value loss/explained variance; GRPO reports zero-variance group rate.

## GRPO evidence and checkpoint safety

The single-update runner uses the Trainer-created top-level `checkpoint-1` as its sole authoritative checkpoint. It never performs a second manual `save_model`. The exact `training_args.bin` basename is accepted only as non-symlink regular trainer metadata directly under that checkpoint, capped at 1 MiB and hashed without deserialization; arbitrary `.bin` files remain forbidden.

The sole TRL 0.24.0 shim binds completion IDs/masks, exact mask-derived token counts, Unicode decoded text, exact verifier input, and ordered reward results into eight JSONL records. Missing or reordered evidence fails closed. The frozen config resolves to `beta=0.0`, so KL is represented as unavailable with `null` and an explicit reason. PyTorch allocator peaks are recorded separately from nvidia-smi. See `docs/artifact-schema.md` and `docs/checkpoint-safety.md`.

## Guarded generation-only prompt diagnostic

The independent v0/v1 diagnostic defaults to a CPU-only static preflight:

```bash
PYTHONPATH=src python -m math_rlvr.evaluation.prompt_ab --config configs/diagnostics/prompt_ab.yaml
```

Real generation is not authorized by the training flags. It requires both
`--generate-only --confirm-prompt-diagnostic`, a clean worktree, offline mode, and the
exact local Qwen 0.5B snapshot. It uses the BF16 base model in eval/inference mode with
all parameters frozen, matched seeds across prompt variants, 16 completions, and a 2,048
token cap. Trainer, LoRA, train, backward, optimizer, checkpoint/model writes, retries,
and automatic v1 activation are fail-closed.

The candidate decision is diagnostic only: v1 must improve complete-envelope rate, yield
at least one envelope, avoid higher truncation, and create nonzero within-problem reward

`docs/prompt-diagnostic-artifact-schema.md`. A versioned capability manifest must prove
paired artifacts, per-problem rewards, allocator evidence, failure backup, post-worker
GPU verification, and cross-file consistency before the fixed worker may start. The
non-CUDA parent launches one fixed spawned worker, then verifies PID exit, absence from
the nvidia-smi compute list, and restoration to baseline before final backup/publication.
