# Code RLVR: PPO vs. GRPO

A portfolio-oriented, reproducible comparison of PPO and GRPO reinforcement-learning
post-training for code generation. The main model is
`Qwen/Qwen2.5-Coder-1.5B-Instruct`; bounded smoke tests use the 0.5B variant before any
paid main run. Phase 1 is scaffolding only: trainer adapters intentionally refuse to train.

## Experiment Design

PPO and GRPO have separate entry points but share `dataset.py`, `prompt.py`, the deterministic
reward/verifier, and evaluation aggregation. A fair comparison fixes dataset split, prompt,
sampling budget, BF16 LoRA targets, seed, reward weights, evaluator, and training-token budget.
Report quality together with GPU hours and estimated cost rather than comparing step count alone.

| Run | Model | Samples | Steps | Generation | Checkpoints | GPU budget |
|---|---|---:|---:|---:|---:|---:|
| Smoke PPO/GRPO | Qwen2.5-Coder-0.5B | 32 | 5 | 128 | 1 | 0.25 h each |
| Main PPO | Qwen2.5-Coder-1.5B | 2,000 | 500 | 512 | 2 | 8 h |
| Main GRPO | Qwen2.5-Coder-1.5B | 2,000 | 500 | 512 | 2 | 8 h |

Budgets are hard planning ceilings, not runtime guarantees. Replace the placeholder CNY cost
caps with the current AutoDL hourly price before training.

## Layout

- `configs/`: bounded smoke/main, path, reward, and evaluation settings.
- `src/code_rlvr/training/`: independent PPO and GRPO entry points.
- `src/code_rlvr/{dataset,prompt}.py`: shared input pipeline contract.
- `src/code_rlvr/{rewards,verifier,evaluation}/`: shared RLVR and metrics.
- `src/code_rlvr/execution/`: fail-closed isolation capability and execution API.
- `tests/`: offline unit tests; `reports/`: experiment report template.

All runtime artifacts belong under `/root/autodl-tmp`: Hugging Face cache in
`/root/autodl-tmp/huggingface`, data in `/root/autodl-tmp/code-rlvr-data`, outputs in
`/root/autodl-tmp/code-rlvr-outputs`, and checkpoints in
`/root/autodl-tmp/code-rlvr-checkpoints`.

## Recommended Environment

The container currently has Python 3.12, PyTorch 2.8.0+cu128, and CUDA 12.8 runtime packages.
The recorded ranges are `transformers>=4.55,<5`, `trl==0.24.0`, `peft>=0.17,<1`,
`accelerate>=1.10,<2`, and `datasets>=4,<5`. These lower bounds represent the contemporary
PyTorch 2.8/Python 3.12 generation; upper bounds avoid known major-version migrations. TRL is
locked exactly because 0.24.0 exports `PPOConfig`, `PPOTrainer`, `GRPOConfig`, and `GRPOTrainer`,
while the previously resolved 0.29.1 no longer exposed the expected stable PPO API. Its published
requirements (`accelerate>=1.4.0`, `datasets>=3.0.0`, and `transformers>=4.56.1`) are satisfied by
this environment. No `bitsandbytes` dependency is included because this design uses BF16 LoRA,
not QLoRA.

### PPO reward compatibility decision

TRL 0.24.0's `GRPOTrainer` accepts callable `reward_funcs`, which maps naturally to the shared
verifier score once safe code execution exists. Its `PPOTrainer` instead requires both a neural
`reward_model` and a separate `value_model`; the current verifier produces post-generation scalar
rewards and is not a Transformers reward model. A reviewed adapter must define how verifier scores
are aligned to response tokens/batches and how value learning is supplied without changing reward
semantics. Until that design has fixtures and CPU tests, the PPO entry point remains preflight-only.

Installation is deliberately not performed in phase 1. A future isolated environment may use:

```bash
python -m venv /root/autodl-tmp/venvs/code-rlvr
python -m pip install -r requirements/dev.txt
python -m pip freeze > requirements/lock.txt
```

## Safe Checks and Commands

These commands are offline and do not load a model:

```bash
make check-env
make static
make test                 # after dev dependencies are installed
make smoke-ppo            # configuration preflight only in phase 1
make smoke-grpo           # configuration preflight only in phase 1
```

Future reviewed training commands will be separate and explicit:

```bash
python -m code_rlvr.training.ppo --config configs/main/ppo.yaml --execute
python -m code_rlvr.training.grpo --config configs/main/grpo.yaml --execute
```

They currently stop with an error even when `--execute` is supplied, preventing accidental GPU
spend before trainer implementation and review.

## Code Execution Safety

Generated code is untrusted. The executor defaults to `safe_backend=None` and refuses execution.
`subprocess`, even with a timeout, is not a sandbox. A future backend must verify filesystem and
network isolation, syscall restrictions, process/memory/time limits, and cleanup. If capability
verification fails, reward and evaluation must remain unavailable rather than falling back to host
execution. AutoDL does not support nested Docker, and this project does not assume it does.

## Reproducibility Sequence

1. Run `make check-env` and archive its output.
2. Install and lock dependencies only in an approved setup phase.
3. Add a reviewed isolation backend and tiny, versioned dataset fixture.
4. Run both 0.5B smoke configs and verify reward/evaluation parity.
5. Record the current hourly rate, approve budgets, then run matched 1.5B experiments.
6. Fill in `reports/experiment_report.md` with metrics, costs, configs, and commit IDs.
