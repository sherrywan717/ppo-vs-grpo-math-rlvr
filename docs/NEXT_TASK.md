# Next task: Stage H.1 — CPU-only PPO evidence/checkpoint repair

Status: blocked on one bounded execution-chain defect. This file does not authorize
CUDA initialization, model loading, generation, training, resume, or another attempt.

## Failure evidence

The immutable run `ppo_formal_1p5b_seed42_20260718T150510Z` reached the live step-8 checkpoint boundary and
failed because formal checkpoint metric normalization treated absent TRL `grad_norm`
as required. Finalized completion/metric/verifier JSONL files are empty and counters
are zero, so generated tokens and scientific training metrics are unavailable. The
partial checkpoint is not resume-capable. See
`reports/runs/ppo_formal_1p5b_seed42_20260718T150510Z/report.md`.

## Sole repair scope

Perform a minimal CPU-only repair that:

1. Reuses the existing metric-availability contract so missing optional `grad_norm`
   becomes `value=null`, `available=false`, with the exact reason and original-key
   evidence; it must not become zero.
2. Ensures each completed update's completion, reward/verifier, metric, comparison-key,
   token, update, optimizer, and global-step evidence is appended durably before a
   later checkpoint serialization failure can erase or misreport the prefix.
3. Uses a fake formal PPO path through step 8 with missing grad norm to verify 128
   ordered completion rows, eight metric/update/optimizer/global-step records, truthful
   generated-token accounting, and a trusted resume-capable checkpoint-8 inventory.
4. Verifies the checkpoint contains policy/value adapters and scalar head plus trusted
   recovery state, but no full base-model weights.

Do not introduce a new checkpoint format, metric, schema, fallback, broad guard, or
unrelated test. Preserve all frozen config/suite/model/data/prompt/reward/parser/
verifier/LoRA/sampling/budget SHAs and keep the failed run immutable.

## Allowed verification

Run only affected targeted CPU tests, Ruff on affected files, compileall on affected
modules, and the formal PPO dry-run. CUDA/model/tokenizer/generation/Trainer/backward/
optimizer activity must remain zero. Do not run the full pytest suite unless separately
requested.

After a repair commit, update the handoff/memory/registry and stop. A new real PPO
attempt requires a new explicit authorization; GRPO seed 42 remains unauthorized.
