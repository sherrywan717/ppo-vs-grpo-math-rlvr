# PPO vs GRPO for Few-Shot Math RLVR at 1.5B

A reproducible comparison of PPO and GRPO sample efficiency, stability, and generalization on `Qwen/Qwen2.5-1.5B-Instruct`. Formal training uses GSM8K and MATH Level 1–3; evaluation uses GSM8K test and MATH500. Countdown is smoke/verifier-only and is never a headline benchmark.

## Frozen design

Both algorithms use seed `20260712`, the same frozen manifests, prompt envelope, BF16 policy LoRA (`r=16`, alpha 32, dropout 0; q/k/v/o projections), four completions, temperature 0.8, top-p 0.95, prompt length 512, completion length 384, and the same verifier/reward policy. Comparisons align actual completions and generated tokens, not trainer steps.

Formal reward is fixed: format 0.10, parse/semantic validity 0.10, correctness 0.80. Thus correct is 1.00, parseable wrong is 0.20, format-correct parse failure is 0.10, and format error is 0.00. Infrastructure errors abort.

PPO uses a separate sequence-classification value model from the policy checkpoint, value LoRA `r=8`, alpha 16 on q/v projections, and a trainable scalar head. This phase defines and tests contracts only; no model is loaded and training remains disabled.

## Data and layout

Frozen manifests live under `/root/autodl-tmp/datasets/math_rlvr/manifests`; Hugging Face cache is `/root/autodl-tmp/cache/huggingface`. `src/math_rlvr/` contains schema, prompt, verifiers, rewards, rollout accounting, metrics, and preflight entry points. `src/math_rlvr/execution/` is retained only as legacy/out-of-scope history and is not imported by the math pipeline.

The output contract is exactly one `<reasoning>...</reasoning>` block followed by exactly one terminal `<answer>...</answer>` block. Only answer content is verified. Verifiers never call `eval`, `exec`, dynamic imports, subprocesses, or generated code.

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
