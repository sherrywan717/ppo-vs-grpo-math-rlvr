# Next task: execute the frozen seed-42 GRPO-v2 run

Stage Q has frozen the guarded model-bound runtime. The sole next task, only after new explicit GPU authorization, is one seed-42 GRPO-v2 run from the immutable warm-start checkpoint-16 adapter. Do not run it implicitly.

Frozen command:

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

Contract: seed 42; 128 updates; 512 microsteps; 512 unique curriculum prompts once; 2,048 completions; 524,288 training-token cap; checkpoints and independent 128-problem dev at 32/64/96/128. The initial policy loads only the warm-start adapter; GRPO initializes a fresh optimizer/scheduler. Hidden test remains sealed. Automatic retry is zero.
