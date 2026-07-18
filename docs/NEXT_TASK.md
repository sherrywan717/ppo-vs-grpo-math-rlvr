# Next task: execute a new formal PPO seed 42 attempt

Status: CPU repair verified; GPU execution not yet authorized.

The only next task is a new, single-attempt formal PPO seed-42 run using the unchanged
frozen command and config. It must use a new run ID and must not resume or reuse the
partial checkpoint from `ppo_formal_1p5b_seed42_20260718T150510Z`.

Before execution, verify the new repair commit, clean worktree, frozen suite/config
SHAs, pinned offline snapshot, idle H800, baseline artifacts, writable storage, and the
historical failed run checksum. Then a new explicit authorization may permit exactly:

```bash
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
PYTHONPATH=src \
python -m math_rlvr.training.ppo \
  --config configs/formal_1p5b/resolved/ppo_seed_42.json \
  --execute \
  --confirm-formal-ppo
```

The frozen contract remains 32 updates, 512 rollout completions, 131,072 generated
training-token cap, and checkpoint/validation at 8/16/24/32. Automatic retries are
zero. PPO success or failure does not authorize GRPO, seed 123, or final test.

This file does not itself authorize CUDA, model loading, generation, training,
backward, optimizer, checkpoint validation, or any GPU command.
