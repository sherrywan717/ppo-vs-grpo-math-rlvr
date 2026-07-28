# Next task: execute the remaining three frozen hidden-test roles

Stage S.3 recovered Base metrics only from immutable run
`base_hidden_grpo_v2_seed42_20260728T073339Z`. The original run remains
`engineering_failure_after_generation_during_metric_finalization`; its supplemental
composite is `scientifically_complete_with_recovered_metric_finalization`. Base must
never be generated again.

The only next task requires new explicit GPU authorization for these three commands,
in this exact order and with one common new UTC suite identifier:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src \
python -m math_rlvr.evaluation.grpo_v2_hidden \
  --config configs/grpo_v2/hidden_test_evaluation.json \
  --role old_grpo_v1 \
  --checkpoint /root/autodl-tmp/runs/math_rlvr/grpo_formal_1p5b_seed42_20260720T031006Z/checkpoint-32 \
  --run-dir /root/autodl-tmp/runs/math_rlvr/old_grpo_v1_hidden_grpo_v2_seed42_<UTC_TIMESTAMP> \
  --execute --confirm-grpo-v2-hidden
```

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src \
python -m math_rlvr.evaluation.grpo_v2_hidden \
  --config configs/grpo_v2/hidden_test_evaluation.json \
  --role warmstart_only \
  --checkpoint /root/autodl-tmp/runs/math_rlvr/warmstart_grpo_v2_seed42_20260722T051218Z/checkpoint-16 \
  --run-dir /root/autodl-tmp/runs/math_rlvr/warmstart_only_hidden_grpo_v2_seed42_<UTC_TIMESTAMP> \
  --execute --confirm-grpo-v2-hidden
```

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src \
python -m math_rlvr.evaluation.grpo_v2_hidden \
  --config configs/grpo_v2/hidden_test_evaluation.json \
  --role selected_grpo_v2 \
  --checkpoint /root/autodl-tmp/runs/math_rlvr/grpo_v2_seed42_20260726T044303Z/checkpoint-96 \
  --run-dir /root/autodl-tmp/runs/math_rlvr/selected_grpo_v2_hidden_grpo_v2_seed42_<UTC_TIMESTAMP> \
  --execute --confirm-grpo-v2-hidden
```

Do not rerun Base, change checkpoints/sampling/contracts, or use recovered results to
alter the remaining sequence. This document itself authorizes no GPU work.
