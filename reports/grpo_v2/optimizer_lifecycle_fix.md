# Stage R.3 lazy optimizer lifecycle fix

Stage R.3 fixes only the premature GRPO-v2 optimizer audit that caused immutable run
`grpo_v2_seed42_20260726T034649Z` to fail before training. No model, data, curriculum,
prompt, reward, parser/verifier, LoRA, sampling, capacity, budget, checkpoint, dev or
hidden-test contract changed.

## Root cause and new timing

Transformers 4.57.6 creates and Accelerate-prepares the optimizer and scheduler inside
the native training loop. Therefore `trainer.optimizer is None` immediately after
Trainer construction is valid. The old runtime incorrectly read
`trainer.optimizer.state` at that point.

The runtime now records the post-constructor state as:

- `lifecycle=lazy_not_initialized`
- `optimizer_present=false`
- `scheduler_present=false`
- `sft_optimizer_state_loaded=false`

The first audit runs at native `on_train_begin`, after optimizer/scheduler creation and
preparation but before the first update. It validates the exact policy-LoRA parameter
union, excludes frozen/base roles, accepts an empty AdamW state as fresh, rejects
preexisting moment state for a new run, and verifies the scheduler is at its initial
position. The second audit runs at native `on_step_end` after the first optimizer and
scheduler steps and global-step increment. It verifies state materialization,
unchanged parameter identity, scheduler advancement, and optimizer/global/update
counters equal to one.

Accelerate wrappers are safely unwrapped only to inspect the underlying optimizer.
Wrapper object identity is not used as an equivalence gate.

## Tiny native lifecycle evidence

One authorized CPU-only synthetic Transformers Trainer ran one backward/AdamW step:

- Trainer constructor optimizer: `None`
- `on_train_begin` optimizer state entries: 0
- exact trainable role: one synthetic LoRA parameter; frozen base excluded
- after first step state entries: 1
- scheduler `last_epoch`: 0 → 1
- optimizer/global/update counters: 1/1/1
- CUDA initialized before/after: false/false

The SFT optimizer/scheduler is never loaded for a fresh GRPO-v2 run. Same-run resume
state restoration is deferred to the same native `on_train_begin` lifecycle boundary,
after optimizer/scheduler creation.

## CPU validation

- Focused tests: 4 passed.
- Affected Ruff: passed.
- Affected compileall: passed.
- GRPO-v2 dry-run: passed; model loads, generation, train, backward and optimizer
  execution all zero; CUDA false.
- Manifest validation: passed.
- Independent `check_env`: `cuda_initialized=false`,
  `model_or_tokenizer_loaded=false`.
- `git diff --check`: passed.

Frozen SHA values remain:

- GRPO config:
  `ce3883b0326492b9109963e8d95496936aa3b3b8670cb9d3b4e9346f65c8cc93`
- Dev config:
  `cafd9f4945a31a9befcf90ae1524107e086f0820178447e8b5767cf19c2ffa59`
- Runtime registry raw:
  `32d83b2ac2e7bb64cbab3d09cec3f2834baca0e46df416e9c407ebf7bcf3fd3b`
- Runtime registry canonical:
  `fad035928e6fdc285ec290d295f4d481700c04ac7f5639f41d3e3ac8a0451beb`
- Curriculum:
  `7f7dcfa1218828e72dd6d42783bc2c790897c7e2a8f2f84d59ce2189710e3b41`

The Stage R.2 failure run and checksum evidence remain immutable.

下一步直接重新授权真实GRPO-v2训练。
