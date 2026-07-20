# Next task: Stage I formal GRPO seed 42

Status: PPO seed-42 training plus recovered checkpoint validation is scientifically
complete as a transparent composite. The original PPO training run remains immutable
with its historical engineering-failure status; it must not be rerun or resumed.

The only next task is a new formal GRPO seed-42 training run from update 0, after a new
explicit GPU authorization. This file records scope only and does not authorize CUDA,
model loading, generation, or training.

Frozen identity:

- Config: `configs/formal_1p5b/resolved/grpo_seed_42.json`
- Config SHA256: `3371d23166d01834c67830eb8dfd51a02d4af483687b0b29e941194174099199`
- Active suite raw SHA256:
  `11869c63f4365aee5d4bf8e13fe263c9d0397164a18a88b419da07218f6a2017`
- Active suite canonical SHA256:
  `1d7c29f76d9bfbf11e1838cd6b0bc8f3da6d0133e0605420e9ed838de729d600`
- Model revision: `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`
- Budget: 32 updates, 512 training completions, 131,072 training rollout tokens
- Checkpoint/validation cadence: 8, 16, 24, 32
- Attempt: one; automatic retries: zero

Future authorized command:

```bash
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
PYTHONPATH=src \
python -m math_rlvr.training.grpo \
  --config configs/formal_1p5b/resolved/grpo_seed_42.json \
  --execute \
  --confirm-formal-grpo
```

Before any future execution, verify clean branch/HEAD, exact config/suite/model and
scientific identities, canonical local snapshot/offline mode, idle H800, writable run
and backup paths, and a non-conflicting new run ID. Do not expand tests or engineering
infrastructure when no correctness, fairness, recovery, safety, or evidence-truth
blocker exists.

PPO rerun/resume, seed 123, baseline, final test, and automatic progression beyond
GRPO seed 42 remain unauthorized. The formal test baseline is independent of the
64-problem checkpoint validation and must not be used for checkpoint selection.
