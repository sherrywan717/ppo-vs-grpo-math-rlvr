# Next task: separately authorize a fresh Stage R GRPO-v2 seed-42 run

Stage R.1 is complete. The exact pinned tokenizer/runtime renderer passed all 512 training and 128 dev prompts before model-bound execution. Capacity is frozen at prompt 928, completion 256 and sequence ceiling 1,184. GRPO config SHA is `ce3883b0326492b9109963e8d95496936aa3b3b8670cb9d3b4e9346f65c8cc93` and runtime registry canonical SHA is `fad035928e6fdc285ec290d295f4d481700c04ac7f5639f41d3e3ac8a0451beb`. The immutable failed run `grpo_v2_seed42_20260726T030733Z` must not be resumed or reused.

No GPU work is currently authorized. After explicit authorization, create a new run ID and execute exactly once from update 0:

```bash
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
PYTHONPATH=src \
python -m math_rlvr.training.grpo_v2 \
  --config configs/grpo_v2/grpo_v2_seed42.json \
  --warmstart-checkpoint /root/autodl-tmp/runs/math_rlvr/warmstart_grpo_v2_seed42_20260722T051218Z/checkpoint-16 \
  --run-dir /root/autodl-tmp/runs/math_rlvr/<NEW_GRPO_V2_RUN_ID> \
  --execute \
  --confirm-grpo-v2
```

The future run remains 128 updates, 512 microsteps, 512 unique curriculum prompts, 2,048 completions, 524,288 training-token cap, and checkpoint/dev cadence 32/64/96/128. Hidden test remains inaccessible. Do not launch it without new explicit GPU authorization.
