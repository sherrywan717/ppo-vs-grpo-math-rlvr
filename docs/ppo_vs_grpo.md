# PPO versus GRPO

## What is comparable

PPO and GRPO share model revision, training/validation examples and order, prompt, policy LoRA, sampling, reward/verifier, 32 updates, 512 completions, token cap, and checkpoint cadence. Training curves are aligned by actual generated tokens and updates.

## What is algorithm-specific

PPO learns a separate value adapter and scalar head, reports policy/value/total losses, and incurs substantially higher peak memory. GRPO derives relative advantages within four-completion prompt groups and has no value model/value loss. Their loss values and native entropy definitions are not directly comparable.

## Observed result

At seed 42, held-out pass@1 is Base 4.0%, PPO 3.75%, and GRPO 7.0%. Paired GRPO improvements over Base and PPO are positive with intervals above zero; pass@4 intervals cross zero. At step-32 validation, GRPO exceeds PPO in both seeds (5 vs 2 correct at seed 42; 6 vs 3 at seed 123). This is consistent evidence under a small budget, not a universal superiority claim.

GRPO also used about 8.5–10.9 GiB peak VRAM and ¥2.75–¥2.93 per formal training-plus-validation run, versus PPO's 51.9–52.6 GiB and ¥3.78–¥4.26. The value-model architecture explains much of the resource difference.

See the [seed-42 final comparison](../reports/formal_1p5b/13_seed42_final_comparison.md) and [four-run aggregate](../reports/formal_1p5b/10_four_run_training_validation_aggregate.md).
