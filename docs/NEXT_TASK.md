# Next task: execute the two frozen matched dev-v2 evaluations

Stage P warm-start training is immutable and must not be rerun. Stage P.1 Phase A has frozen the shared evaluator at config SHA `8501bfb945f85dda895d9278bb5d1d74a5d9c2c0791f9daa7cb0152d25e02528`.

Execute exactly once, in order, with no retry:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src python -m math_rlvr.evaluation.grpo_v2_dev --config configs/grpo_v2/dev_evaluation_seed42.json --mode base --run-dir /root/autodl-tmp/runs/math_rlvr/base_dev_grpo_v2_seed42_20260722T060500Z --execute --confirm-grpo-v2-dev
```

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src python -m math_rlvr.evaluation.grpo_v2_dev --config configs/grpo_v2/dev_evaluation_seed42.json --mode warmstart --checkpoint /root/autodl-tmp/runs/math_rlvr/warmstart_grpo_v2_seed42_20260722T051218Z/checkpoint-16 --run-dir /root/autodl-tmp/runs/math_rlvr/warmstart_dev_grpo_v2_seed42_20260722T060500Z --execute --confirm-grpo-v2-dev
```

Each run is 128 single-candidate completions. Do not run GRPO-v2, hidden test, warm-start training, or any retry.
