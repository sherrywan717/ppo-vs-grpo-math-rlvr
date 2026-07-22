# Next task: Stage O GRPO-v2 warm-start execution

Stage N is complete and CPU-only. The single next task requires new explicit authorization: implement/verify the model-bound warm-start entrypoint, perform the pinned local-tokenizer target-length audit, and execute exactly one seed-42 one-epoch warm-start from `configs/grpo_v2/warmstart_seed42.json`.

Frozen intended command (not executable or authorized in Stage N):

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src \
python -m math_rlvr.training.warmstart_v2 \
  --config configs/grpo_v2/warmstart_seed42.json \
  --execute --confirm-grpo-v2-warmstart
```

Before GPU execution, Stage O must add or verify that exact guarded entrypoint without altering the frozen data/scientific contract, audit all 256 rendered target lengths with the pinned local tokenizer, and confirm every target fits 256 tokens. If any target is over cap, stop for user adjudication; do not truncate or silently alter data. Stage O does not authorize GRPO-v2, dev checkpoint selection, or hidden test.

Future GRPO command, requiring a separate authorization after warm-start/dev evidence is complete:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src \
python -m math_rlvr.training.grpo_v2 \
  --config configs/grpo_v2/grpo_v2_seed42.json \
  --warmstart-checkpoint <TRUSTED_WARMSTART_CHECKPOINT> \
  --execute --confirm-formal-grpo-v2
```
