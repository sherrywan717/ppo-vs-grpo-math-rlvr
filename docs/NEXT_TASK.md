# Next task: execute one GRPO-v2 warm-start seed 42 run

Only a new explicit GPU authorization may execute the frozen warm-start. It must begin from a clean `improve/grpo-v2` worktree and the exact committed config/runtime registry. It does not authorize dev evaluation or GRPO-v2.

```bash
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
PYTHONPATH=src \
python -m math_rlvr.training.warmstart \
  --config configs/grpo_v2/warmstart_seed42.json \
  --run-dir /root/autodl-tmp/runs/math_rlvr/<NEW_WARMSTART_RUN_ID> \
  --execute \
  --confirm-grpo-v2-warmstart
```

Expected contract: seed42; 256 unique samples; one epoch; batch4/GA4/effective16; 64 microsteps; 16 optimizer/global/scheduler steps; BF16 policy LoRA with 4,358,144 trainables; prompt/target/actual-sequence caps 928/640/1,088; no truncation; checkpoint-16 adapter plus trusted resume state; no full base weights; zero retries. Stop after the warm-start and GPU-release/backup checks.

Stage O.3 changes only the future hidden-evaluation contract: the active 100-problem subset uses one shared n=10 candidate batch and exact unbiased pass@1/pass@4/pass@10 estimators. The O.2 50-problem design is superseded before evaluation. This does not change or authorize the warm-start command above.
