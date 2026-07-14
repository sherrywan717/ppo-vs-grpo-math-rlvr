# Matched 0.5B pilot execution-contract fix

`Matched 0.5B pilot - not the final benchmark`

This CPU-only repair addresses exactly two blockers: PPO prompt-major sequential rollout and protected 4/8/16 completion evidence profiles. It did not initialize CUDA, load a model/tokenizer, generate text, run a real Trainer, call backward/optimizer, alter frozen configs/manifests, or modify Stage D history.

## TRL 0.24.0 PPO risk and replacement

TRL 0.24.0 constructs the PPO training `DataLoader` with `shuffle=True`, then prepares model, optimizer and that loader together. `train()` later assigns `dataloader = self.dataloader`, repeatedly yields from it, and consumes `data = next(iter_dataloader)`. A fixed seed makes a shuffle reproducible, but it does not make it equal to the frozen prompt-major comparison order.

The guarded subclass now replaces only `self.dataloader`, immediately after `PPOTrainer.__init__` returns. It creates an explicit `SequentialSampler` loader with batch size 16, `drop_last=True`, `num_workers=0`, and the trainer's existing collator wrapped only to preserve episode metadata. It calls the trainer's existing `accelerator.prepare_data_loader`; it does not prepare the model or optimizer again.

The prepared first batch is validated field-by-field for `episode_position`, `problem_id`, `generation_index`, pair key, problem hash, rendered prompt hash, seed and algorithm. A proxy repeats the same validation on the iterator actually consumed by `train()`. Random sampling, Accelerator-side reordering, swaps, missing/duplicate rows and hash drift fail closed.

Fake-Accelerator tests for seeds 42, 123 and 2026 observed `RandomSampler` before replacement and `SequentialSampler` afterward. Each prepared batch contained the exact 16 keys in [ppo_episode_order.csv](ppo_episode_order.csv).

## Protected evidence profiles

| Profile | Config authorization | Prompts × responses | Completions | Token cap | Update / optimizer / global |
|---|---|---:|---:|---:|---:|
| PPO Stage D smoke | exact `configs/smoke/ppo.yaml` SHA256 | 4 × 1 | 4 | 512 | 1 / 1 / 1 |
| GRPO Stage D smoke | exact `configs/smoke/grpo.yaml` SHA256 | 2 × 4 | 8 | 1,024 | 1 / 1 / 1 |
| PPO matched pilot | exact allowlisted resolved config SHA256 | 4 × 4 | 16 | 2,048 | 1 / 1 / 1 |
| GRPO matched pilot | exact allowlisted resolved config SHA256 | 4 × 4 | 16 | 2,048 | 1 / 1 / 1 |

`ExpectedRunContract` is immutable and selected only by exact repository path plus raw config SHA256. It also binds model/revision/local-only/BF16, policy LoRA, sampling, prompt, reward, parser, verifier and manifest identities. Main/1.5B configs and arbitrary CLI limits have no profile.

Online overflow fails before an update. Finalization requires equality, not `>=`: PPO smoke 4 succeeds while 3/5 fail; GRPO smoke 8 succeeds while 7/9 fail; both pilot profiles succeed at 16 while 15/17 fail. Tokens remain mask-derived. Metrics receive guard-derived completion/token/update/optimizer/global counters so metrics, summary and run manifest cannot silently disagree.

GRPO retains its four-generations-per-prompt batching semantics. Its actual completion order is mapped to `problem_id::generation_index` and must equal the same 16-key set used by PPO, once each; see [comparison_keys.csv](comparison_keys.csv).

## Frozen identity and historical preservation

- Pilot manifest SHA256 remains `0235210e038bc27ebf2e7218691f36f09c8e11f0bbc743f46a5318a279f6bc1f`.
- All six resolved config SHA256 values remain unchanged; they are listed in [execution_contract_fix.json](execution_contract_fix.json).
- Stage D smoke config SHA256 values remain `547e6736…9f668` for PPO and `068ff8d7…d8979` for GRPO.
- Existing protected Stage D artifact hash tests still pass. No file under `reports/runs/` was edited.

## Validation state

Targeted execution-contract and Stage D regression tests passed (120 tests), as did
321 full tests, full Ruff, compileall, `check_env`, manifest validation, six pilot
dry-runs, two Stage D dry-runs, fake PPO/GRPO pilot execute, and 16-row
CSV/JSON/figure finalization. `check_env` reported `cuda_initialized=false` and
`model_or_tokenizer_loaded=false`; real generation, Trainer, backward and optimizer
calls were zero.

No correctness blocker remains among the two items in this repair scope. The GPU suite
is still unauthorized: it requires a future explicit user authorization and fresh
mutable preflight. No GPU pilot run was executed by this repair.
