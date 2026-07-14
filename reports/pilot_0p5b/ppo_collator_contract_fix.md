# PPO pilot collator mapping contract fix

## Outcome

The CPU-only repair passed. The failure was a Python type-contract mismatch, not a
TRL padding or tokenization failure. No frozen experiment identity or budget changed.

## Exact reproduction

Using frozen PPO seed 42, the real 16-row pilot dataset, the fixed local tokenizer,
TRL/Transformers 0.24.0/4.57.6 and a CPU DataLoader:

- Base collator: `DataCollatorWithPadding`
- Input feature type: `dict`
- Input keys: `input_ids` plus the eight protected episode metadata fields
- Base return type: `BatchEncoding`
- Base mapping status: `MutableMapping=true`, `dict=false`
- Base keys: `input_ids`, `attention_mask`
- Shapes: both `[16, 161]`
- Dtypes: both `torch.int64`
- Exact old failure: `TRLContractError: PPO data collator must return a mapping`

The exception occurred in `OrderedMetadataCollator.__call__` after the base collator
returned but before metadata attachment. The old wrapper incorrectly required a
concrete `dict` even though `BatchEncoding` is the standard mutable Mapping.

## Minimal repair

Only `src/math_rlvr/training/trl_compat.py` changed. The wrapper now accepts the
actual `MutableMapping`; prepared-batch extraction accepts `Mapping`. It mutates and
returns the original `BatchEncoding`, so `input_ids`, `attention_mask`, padding,
dtypes, tokenization and response boundaries are unchanged. Episode metadata is
removed before base collation, reattached afterward, and never expanded into
policy/value model kwargs.

The CPU Accelerator-prepared batch remains a `BatchEncoding` Mapping. The fixed
loader remains `SequentialSampler`, batch size 16, `drop_last=true`, `num_workers=0`,
and the first/train-consumed batch retains all 16 prompt-major comparison keys.

## Verification

- 47 targeted collator/order/scope/evidence tests passed.
- 369 full pytest tests passed.
- Ruff, compileall, check_env, manifest validation and `git diff --check` passed.
- Three PPO and three GRPO pilot dry-runs passed.
- Stage D PPO/GRPO dry-runs passed.
- Fake 16-completion pilot PPO/GRPO finalization passed.
- CUDA initialized: false.
- Real model load, generation, Trainer.train, backward and optimizer steps: zero.

The Stage D four-row collator regression passed and the GRPO path was unchanged.
Main/formal rejection remains active. All six pilot config hashes, the pilot manifest,
prompt/reward/parser/verifier identities and budgets remain frozen.

Both historical failures remain immutable and excluded from scientific aggregation:

- `ppo_matched_0p5b_seed42_20260714T073357Z`
- `ppo_matched_0p5b_seed42_20260714T082003Z`

Stage A is technically complete. The separately authorized new GPU suite may begin
only after the repair commit, verified static backup, clean worktree and fresh GPU,
offline snapshot, frozen identity, prompt-scope and collator preflights.
