# Stage Q: guarded model-bound GRPO-v2 runtime freeze

Stage Q is complete as a CPU-only/static/fake gate. It did not initialize CUDA, load a model or tokenizer, generate, call a real Trainer, run backward, or step an optimizer.

## Frozen execution boundary

- Config: `configs/grpo_v2/grpo_v2_seed42.json` — `059553888fdc997a5b9f214fde526d4be8c309ca84abe212c243fd74305b1b66`
- Runtime registry: `43ef900265e37a355d7edf271384a5f7c84166a17b378034349c344228dab3fa`
- Initial policy: immutable warm-start checkpoint-16 policy adapter `44066dd13...72b9`; checkpoint artifact `507749d3...92f0`.
- The SFT optimizer/scheduler are never loaded into a new GRPO-v2 run. GRPO creates a fresh optimizer/scheduler over the exact policy-LoRA trainable set. Same-run resume restores only a project-created GRPO-v2 optimizer/scheduler.
- Frozen training: seed 42, 512 unique curriculum prompts once each, 128 updates, 512 microsteps, 2,048 completions, and a 524,288 generated-token cap.
- Checkpoint and matched dev cadence: 32/64/96/128. Each dev pass has 128 single-candidate rows and an independent completion/token ledger.
- Resume is allowed only from the same run at checkpoint 32/64/96 with exact config/model/data/curriculum/warm-start identity and inventory SHA validation.
- Hidden test is not imported by the training contract and cannot enter checkpoint selection.

## Evidence and safety

Each successful update atomically rewrites the validated completion/metric prefix before a checkpoint callback. Completion evidence binds curriculum position, problem/prompt identity, generation index, token IDs/mask/count, text, EOS/truncation, reward components and canonical verifier status. Update evidence preserves reward groups/variance, zero-signal diagnostics, loss, entropy definition/raw key, optional telemetry availability, generated tokens, optimizer/global/microstep counters and learning rate. Missing optional telemetry is `null` with `available=false` and a reason.

Checkpoints contain only policy LoRA plus trusted optimizer, scheduler, Python/NumPy/PyTorch CPU/CUDA RNG, Trainer/runtime counters, curriculum cursor, completion/metric prefixes and SHA inventory. Full base-model weights and PPO value roles are rejected.

## CPU validation

- 49 targeted tests passed across the new runtime, existing formal checkpoint/runtime, and matched-dev safety suites.
- Exact 128 updates / 512 microsteps / 2,048 completions passed; 127/129 and budget overflow failed closed.
- Checkpoint 32/64/96 same-run resume passed; continuous 128 and 64+resume fake final state matched exactly.
- Four independent 128-row dev ledgers passed without changing training counters.
- Ruff, affected compileall, GRPO-v2 dry-run, environment check, manifest validation, `git diff --check`, and secret/large-file audit passed.
- Full pytest was intentionally not run under the Stage Q scope.
