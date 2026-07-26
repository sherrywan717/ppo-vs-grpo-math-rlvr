# Next task: directly reauthorize one fresh GRPO-v2 seed-42 run

Stage R.3 is complete. The native lazy optimizer lifecycle is now correctly audited:
constructor-time `optimizer=None` is accepted, exact policy-LoRA/fresh state is checked
at `on_train_begin`, and first-step state/scheduler/counters are checked at
`on_step_end`. No SFT optimizer or scheduler is inherited. Frozen scientific
identities and both immutable failed attempts are unchanged.

下一步直接重新授权真实GRPO-v2训练。

After new explicit GPU authorization, create a unique run ID and execute exactly once
from update 0:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src python -m math_rlvr.training.grpo_v2   --config configs/grpo_v2/grpo_v2_seed42.json   --warmstart-checkpoint /root/autodl-tmp/runs/math_rlvr/warmstart_grpo_v2_seed42_20260722T051218Z/checkpoint-16   --run-dir /root/autodl-tmp/runs/math_rlvr/<NEW_GRPO_V2_RUN_ID>   --execute   --confirm-grpo-v2
```

The run remains 128 updates, 512 microsteps, 512 unique prompts, 2,048 completions,
524,288 training-token cap, and checkpoint/dev cadence 32/64/96/128. Do not resume or
reuse either failed run. Hidden test remains sealed. This document does not authorize
the GPU command by itself.
