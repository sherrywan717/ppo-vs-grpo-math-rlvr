# PPO seed-42 validation-only recovery eligibility

## Decision

- Original run: `ppo_formal_1p5b_seed42_20260719T131800Z`
- Original status: `engineering_failure_after_training`
- Training contract: **complete**
- Training evidence: **complete**
- Checkpoint contract: **complete**
- Validation contract: **pending**
- Validation-only eligible: **true**
- Training resume authorized: **false**
- Training rerun required: **false**

This is a read-only Stage H.2 audit. It does not modify or relabel the original failed
run, its summaries, checkpoint files, checksums, or backup. Scientific success remains
pending until all four separately authorized validation runs complete and their
artifacts are reconciled.

## Cadence root cause and repair

The PPO callback incrementally advanced the training cursor from 1 through 32 and
atomically persisted each update. Checkpoint and validation events were intentionally
deferred until the Trainer returned, but the old guard compared replayed checkpoint 8
with the current training cursor 32.

Training progress now remains the monotonic 1..32 update/optimizer/global/completion/
token contract. Checkpoint and validation cadence use their own ordered cursors
`[8,16,24,32]`; a deferred step may be less than the completed training cursor, but it
cannot be out of order, duplicated, illegal, ahead of training, or validated without
its same-step trusted checkpoint. Validation rows and tokens do not alter training
counters or the 512/131,072 training budget.

## Existing training evidence

The formal guard successfully replayed all original evidence:

- 32 metric rows and 512 completion rows
- update/optimizer/global: 32/32/32
- exact generated tokens: 51,369
- complete ordered problem/generation comparison keys
- completion token IDs, masks and exact counts valid
- reward and canonical verifier evidence present

The original run remains an engineering failure after training because validation is
still 0/4. The generic failure summary is not rewritten.

## Checkpoint audit

| Step | Completion prefix | Metric prefix | Training tokens | Total bytes | Artifact manifest SHA256 |
|---:|---:|---:|---:|---:|---|
| 8 | 128 | 8 | 13,468 | 66,535,669 | `805f2916...f914` |
| 16 | 256 | 16 | 25,216 | 67,271,976 | `56d8c908...34d9` |
| 24 | 384 | 24 | 39,158 | 68,021,100 | `854a8c02...8ca9c` |
| 32 | 512 | 32 | 51,369 | 68,703,758 | `18534747...1952` |

All four checkpoints passed the existing formal inventory and evaluation-selection
validators. Each contains policy/value adapters, scalar head, optimizer, scheduler,
Python/NumPy/PyTorch CPU/CUDA RNG evidence, runtime/trainer counters and evidence
prefixes. Model/config/suite/prompt/reward/parser/verifier identities match. No full
base-model weights are present.

Original `checksums.sha256` file SHA256:
`43295b905f4175a41de21cd41e71e1e42d687c80a411af0421f91ecc3133e372`.
Immutable failure backup SHA256:
`f63812afed44cdc9f0fcafdf0931454548da1a4ce145840ebf91bb6fa5a6d7c5`.

## Future validation-only commands

Each command requires a new explicit GPU authorization, a unique replacement for
`<UTCSTAMP>`, one attempt, and zero automatic retries. Each writes a new evaluation
run and does not modify the training run.

Step 8:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src python -m math_rlvr.evaluation.formal --config configs/formal_1p5b/evaluation.json --phase validation --algorithm ppo --seed 42 --mode ppo --checkpoint-step 8 --checkpoint /root/autodl-tmp/runs/math_rlvr/ppo_formal_1p5b_seed42_20260719T131800Z/checkpoint-8 --run-dir /root/autodl-tmp/runs/math_rlvr/ppo_validation_formal_1p5b_seed42_step8_<UTCSTAMP> --execute --confirm-formal-evaluation
```

Step 16:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src python -m math_rlvr.evaluation.formal --config configs/formal_1p5b/evaluation.json --phase validation --algorithm ppo --seed 42 --mode ppo --checkpoint-step 16 --checkpoint /root/autodl-tmp/runs/math_rlvr/ppo_formal_1p5b_seed42_20260719T131800Z/checkpoint-16 --run-dir /root/autodl-tmp/runs/math_rlvr/ppo_validation_formal_1p5b_seed42_step16_<UTCSTAMP> --execute --confirm-formal-evaluation
```

Step 24:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src python -m math_rlvr.evaluation.formal --config configs/formal_1p5b/evaluation.json --phase validation --algorithm ppo --seed 42 --mode ppo --checkpoint-step 24 --checkpoint /root/autodl-tmp/runs/math_rlvr/ppo_formal_1p5b_seed42_20260719T131800Z/checkpoint-24 --run-dir /root/autodl-tmp/runs/math_rlvr/ppo_validation_formal_1p5b_seed42_step24_<UTCSTAMP> --execute --confirm-formal-evaluation
```

Step 32:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src python -m math_rlvr.evaluation.formal --config configs/formal_1p5b/evaluation.json --phase validation --algorithm ppo --seed 42 --mode ppo --checkpoint-step 32 --checkpoint /root/autodl-tmp/runs/math_rlvr/ppo_formal_1p5b_seed42_20260719T131800Z/checkpoint-32 --run-dir /root/autodl-tmp/runs/math_rlvr/ppo_validation_formal_1p5b_seed42_step32_<UTCSTAMP> --execute --confirm-formal-evaluation
```

The frozen plan estimates these four 64-completion validations at approximately
20 minutes / 0.3333 GPU-hours / CNY 2.96 total, with a planning ceiling of
40 minutes / 0.6667 GPU-hours / CNY 5.92 at CNY 8.88 per GPU-hour.

## Current boundary

Stage H.2 ran CPU-only: CUDA/model/tokenizer/generation/Trainer/backward/optimizer and
validation counts all remained zero. This report authorizes no validation, PPO rerun,
training resume, GRPO, seed 123, baseline, or final test.
