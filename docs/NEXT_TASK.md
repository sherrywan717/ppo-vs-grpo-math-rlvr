# Next task: Stage H.3 PPO seed-42 validation-only recovery

Status: Stage H.2 CPU repair and read-only eligibility audit passed. GPU validation is
not yet authorized.

The only next task is to execute the existing PPO seed-42 checkpoints in strict order
8, 16, 24, 32 against the frozen 64-problem validation set. Every command requires a
new explicit authorization, a unique UTC timestamp in its new direct-child run
directory, one attempt, and zero automatic retries. A failure stops the sequence.

The original training run and checkpoints are immutable inputs:

- Run: `ppo_formal_1p5b_seed42_20260719T131800Z`
- Training: 32 updates, 512 completions, 51,369 rollout tokens
- Validation-only eligible: true
- Training resume authorized: false
- Training rerun required: false
- Original status: `engineering_failure_after_training / validation_pending`
- Scientific aggregate inclusion: false until all four validations and reconciliation pass

Each validation has 64 completions. Its completions/tokens belong only to the
validation ledger and never alter the 512-completion/131,072-token training budget.
The commands load only the checkpoint's policy adapter for generation; value adapter,
scalar head, optimizer and training state are not loaded for evaluation.

### Checkpoint 8

```bash
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
PYTHONPATH=src \
python -m math_rlvr.evaluation.formal \
  --config configs/formal_1p5b/evaluation.json \
  --phase validation \
  --algorithm ppo \
  --seed 42 \
  --mode ppo \
  --checkpoint-step 8 \
  --checkpoint /root/autodl-tmp/runs/math_rlvr/ppo_formal_1p5b_seed42_20260719T131800Z/checkpoint-8 \
  --run-dir /root/autodl-tmp/runs/math_rlvr/ppo_validation_formal_1p5b_seed42_step8_<UTCSTAMP> \
  --execute \
  --confirm-formal-evaluation
```

### Checkpoint 16

```bash
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
PYTHONPATH=src \
python -m math_rlvr.evaluation.formal \
  --config configs/formal_1p5b/evaluation.json \
  --phase validation \
  --algorithm ppo \
  --seed 42 \
  --mode ppo \
  --checkpoint-step 16 \
  --checkpoint /root/autodl-tmp/runs/math_rlvr/ppo_formal_1p5b_seed42_20260719T131800Z/checkpoint-16 \
  --run-dir /root/autodl-tmp/runs/math_rlvr/ppo_validation_formal_1p5b_seed42_step16_<UTCSTAMP> \
  --execute \
  --confirm-formal-evaluation
```

### Checkpoint 24

```bash
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
PYTHONPATH=src \
python -m math_rlvr.evaluation.formal \
  --config configs/formal_1p5b/evaluation.json \
  --phase validation \
  --algorithm ppo \
  --seed 42 \
  --mode ppo \
  --checkpoint-step 24 \
  --checkpoint /root/autodl-tmp/runs/math_rlvr/ppo_formal_1p5b_seed42_20260719T131800Z/checkpoint-24 \
  --run-dir /root/autodl-tmp/runs/math_rlvr/ppo_validation_formal_1p5b_seed42_step24_<UTCSTAMP> \
  --execute \
  --confirm-formal-evaluation
```

### Checkpoint 32

```bash
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
PYTHONPATH=src \
python -m math_rlvr.evaluation.formal \
  --config configs/formal_1p5b/evaluation.json \
  --phase validation \
  --algorithm ppo \
  --seed 42 \
  --mode ppo \
  --checkpoint-step 32 \
  --checkpoint /root/autodl-tmp/runs/math_rlvr/ppo_formal_1p5b_seed42_20260719T131800Z/checkpoint-32 \
  --run-dir /root/autodl-tmp/runs/math_rlvr/ppo_validation_formal_1p5b_seed42_step32_<UTCSTAMP> \
  --execute \
  --confirm-formal-evaluation
```

After all four succeed, CPU-only reconciliation may append validation artifacts and
update the derived registry/handoff. It must not rewrite the original failure summary,
training evidence, checkpoints, or checksums. The existing formal test baseline remains
separate and must not be used as a 64-problem validation delta or for checkpoint
selection.

Frozen planning estimate for all four validations: approximately 20 minutes,
0.3333 GPU-hours and CNY 2.96; ceiling 40 minutes, 0.6667 GPU-hours and CNY 5.92 at
CNY 8.88/GPU-hour.

This file authorizes no CUDA/model load, validation, PPO rerun/resume, GRPO, seed 123,
baseline, final test, or automatic retry.
