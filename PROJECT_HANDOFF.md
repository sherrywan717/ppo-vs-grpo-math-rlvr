# Math RLVR formal 1.5B handoff

## Purpose and authority

This repository is an artifact-first, portfolio-grade fair comparison of PPO and GRPO
for mathematical RLVR on Qwen 1.5B. Authority order is Git/configs/manifests/original
artifacts, this handoff, `docs/PORTFOLIO_DELIVERABLES.md`, `docs/NEXT_TASK.md`,
`AGENTS.md`/`memory.md`, then historical chat. Never rewrite primary evidence to make
a derived document agree.

## Verified repository state

- Branch: `pivot/math-rlvr`
- Stage H.4 repair base HEAD: `68f50a5fea9ec1183adabf04fc60d8daf1dc7d16`
- Worktree: clean at Stage H.4 start; expected clean after the repair commit.
- Model: `Qwen/Qwen2.5-1.5B-Instruct`
- Revision: `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`
- Canonical local snapshot:
  `/root/autodl-tmp/cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306`

## Current formal identity

- Evaluation raw SHA256:
  `85100dd0f613f295a7219a45a42a03e3ad4a45e24893c7f296e1d8da9a1f4a35`
- Evaluation canonical SHA256:
  `d8ba5ab80ab0553d2ec7246fb4876956dcbc5dd0bcf8642fd33c4ec19da6fe44`
- Active-suite raw SHA256:
  `11869c63f4365aee5d4bf8e13fe263c9d0397164a18a88b419da07218f6a2017`
- Active-suite canonical SHA256:
  `1d7c29f76d9bfbf11e1838cd6b0bc8f3da6d0133e0605420e9ed838de729d600`

Active training runs, in frozen order:

1. PPO seed 42 — `configs/formal_1p5b/resolved/ppo_seed_42.json` —
   `1093e87a8363a0a2a6ab640a6f723c04cb6cfb22edef2e38a8c3a0062693ec43`
2. GRPO seed 42 — `configs/formal_1p5b/resolved/grpo_seed_42.json` —
   `3371d23166d01834c67830eb8dfd51a02d4af483687b0b29e941194174099199`
3. GRPO seed 123 — `configs/formal_1p5b/resolved/grpo_seed_123.json` —
   `cc95138f50f37fafa76766d3a08b0995ffd5e0bf87cd7b9050acedb5e0bbc75e`
4. PPO seed 123 — `configs/formal_1p5b/resolved/ppo_seed_123.json` —
   `3d6cc1f30f7b72bfadb5191613298ac3f64a1ba3c699cc8d1e30ce147218c15e`

Seed 2026 remains `reserved_not_scheduled`. Prompt, reward, parser, verifier, data,
sampling, LoRA, completion/token/update budgets, and fixed step-32 selection remain
frozen. Test data is never used for tuning or checkpoint selection.

## Completed execution stages

Stage E.1 formal model-bound CLI wiring and trusted same-run resume are complete.
Stage F is complete:

- Snapshot download commit: `5bbf913358f018e413ea70ef3ce34fa38afcfa1d`
- CUDA sanity commit: `2a86af7572d4f6b1419b1012b9b19f50cf9cbade`
- Sanity run: `cuda_load_sanity_qwen25_1p5b_20260718T113620Z`
- BF16 local-only load and two finite forwards passed; generation/training/backward/
  optimizer/checkpoint counters were zero; post-process GPU release passed.

## Immutable failed baseline attempts

Both attempts are engineering failures and `included_in_scientific_aggregate=false`:

- `baseline_formal_1p5b_seed42_20260718T114907Z`: reward-evidence serialization
  attempted `evaluation.to_dict()["components"]`; 0/800 persisted completion rows.
  Repair commit: `cce3a212f7f5a60edbf43ffd6eef4794850173f6`.
- `baseline_formal_1p5b_seed42_20260718T120909Z`: old 512-token prompt cap rejected
  `math:HuggingFaceH4/MATH-500:test:219` at exactly 800 tokens after 642/800 rows.
  The 642 rows and 74,968 exact tokens remain immutable and are never reused.

Their run directories, checksums, and verified failure backups remain primary evidence.
They are excluded from all baseline metrics and figures.

## Post-freeze prompt-length capacity amendment

A pinned-tokenizer audit covered 1,192 actual-mode rows / 592 unique frozen problems.
Maximum prompt lengths were train 713, validation 339, and test/overall 800. Three
unique problems exceeded the old 512 cap. Commit
`edecfcf503cff8ee8aef3c7ef2136dae04e192b7` publicly amends the shared evaluation,
PPO, and GRPO cap from 512 to 832; 832 + unchanged completion 256 = 1,088, below the
32,768 model context.

This is capacity-only. No prompt text/token IDs, data/order, sampling, reward,
parser/verifier, LoRA, optimizer, or completion/token/update budget changed. Previously
fitting prompt token IDs are unchanged, and PPO/GRPO retain the same protocol. See
`reports/formal_1p5b/prompt_length_amendment.{md,json}`.

## Scientific baseline result

Only these successful post-amendment runs enter the scientific aggregate:

- Seed 42: `baseline_formal_1p5b_seed42_20260718T125833Z` — 800/800,
  96,150 tokens, sampled pass@1 0.040, pass@4 0.100.
- Seed 123: `baseline_formal_1p5b_seed123_20260718T133624Z` — 800/800,
  91,651 tokens, sampled pass@1 0.025, pass@4 0.060.

Greedy accuracy is `null/unavailable` because the frozen protocol has no separate
greedy completion. Aggregate report: `reports/formal_1p5b/01_baseline_results.md`.
Commit: `287f7d313c5ad8ac1500eb416eeacd605c3298f3`.

## Stage H PPO seed-42 failed attempt

The one authorized command ran exactly once as `ppo_formal_1p5b_seed42_20260718T150510Z` and failed at the
step-8 checkpoint callback: `FormalRuntimeError: formal Trainer did not expose required
grad norm`. The missing TRL field was optional under the authorization contract and
should have been persisted as null/unavailable, not treated as a training blocker.

The finalized completion/metric/verifier JSONL files are empty and formal counters are
zero. Live stdout reached the displayed 8/32 boundary and `episode=128`, but those are
not persisted primary evidence; generated tokens and scientific training metrics are
therefore unavailable. No validation ran. The run is an immutable engineering failure
and `included_in_scientific_aggregate=false`.

The partial `checkpoint-8` contains policy/value adapters plus scalar head, but no
optimizer, scheduler, RNG, runtime/counter, comparison-prefix, or trusted inventory
state. It is not resume-capable and must not be evaluated. No full base-model weights
were saved. The verified failure archive is
`/root/autodl-fs/math-rlvr-backups/ppo_formal_1p5b_seed42_20260718T150510Z.failure.tar.gz`, SHA256
`76896c5b3db3ee4439566b8b68c0cad798af5b5610f393138aa23eba6c40debb`. Measured usage was 181.951470 seconds, 0.050542075 GPU-hours,
CNY 0.448814, peak 28,655 MiB nvidia-smi VRAM; post-process GPU state was 0 MiB and no
compute process.

## Stage H.1 repair complete

From pre-repair HEAD `108aa260481710ceb90080200af348f7a0ec0765`, the bounded CPU
repair made missing aggregate/policy/value grad norms nullable with raw-key evidence and
moved validated atomic completion/metric prefix persistence to every PPO update before
the checkpoint callback. A fake step-8 callback failure retains 128 completions, eight
metric rows, and matching counters. Existing trusted checkpoint/resume structure is
unchanged. Twenty-three related pytest cases, affected Ruff/compileall, and formal PPO
dry-run passed; full pytest was not run under the experiment-first limit.

The historical failed run remains immutable and excluded. Its checksums and every
frozen SHA remain unchanged. No CUDA, model/tokenizer, generation, real Trainer,
backward, optimizer, baseline, validation, or final test ran in Stage H.1.

## Stage H PPO seed-42 second failed attempt

The newly authorized command ran exactly once as
`ppo_formal_1p5b_seed42_20260719T131800Z`; automatic retries were zero. Incremental
primary evidence confirms 32 updates, 32 optimizer/global steps, 512 training
completions, and 51,369 training rollout tokens. Checkpoint directories 8/16/24/32
were written with role-separated adapters/head, trusted training state, artifact
manifests, and `base_weights_included=false`.

After training returned, the backend replayed scheduled checkpoint 8 while the
incremental observer was already at update 32. The cadence guard rejected the mismatch
before any frozen 64-problem validation ran. The attempt is therefore an immutable
engineering failure and `included_in_scientific_aggregate=false`; none of its
checkpoints was authorized for resume or evaluation at failure time. Stage H.2 later
made the four trusted checkpoints eligible for the separately authorized validation-only
recovery recorded below; training resume remains unauthorized. The generic exception
`final_summary.json` has zero counters because no success result object reached
finalization, while the authoritative JSONL prefixes, failure report, and checkpoint
manifests agree at 32/512/51,369.

Measured usage was 751.268899 seconds, 0.208685805 GPU-hours, CNY 1.853130,
53,151 MiB peak nvidia-smi VRAM, and 34.8984% mean utilization. After worker exit the
GPU was 0 MiB with no compute process. The verified failure archive is
`/root/autodl-fs/math-rlvr-backups/ppo_formal_1p5b_seed42_20260719T131800Z.failure.tar.gz`,
SHA256 `f63812afed44cdc9f0fcafdf0931454548da1a4ce145840ebf91bb6fa5a6d7c5`.

## Stage H.2 cadence repair and recovery eligibility

Training updates remain a monotonic 1..32 cursor with unchanged optimizer/global,
completion-key, and token-budget checks. Checkpoint and validation use independent
ordered cursors `[8,16,24,32]`; online and deferred execution are both supported.
A validation step requires its same-position trusted checkpoint, while validation
rows/tokens never modify training counters or budgets.

Read-only replay validated 32 metric rows, 512 completion rows, 51,369 tokens, and all
comparison keys. Checkpoints 8/16/24/32 passed actual file SHA256, inventory, prefix,
same-run and frozen-identity validation, and existing formal evaluation selection. No
base weights are present. Therefore `validation_only_eligible=true`,
`training_rerun_required=false`, and `training_resume_authorized=false`. The
original run remains `engineering_failure_after_training / validation_pending` and
is not yet a scientific success.

Nineteen targeted CPU/fake tests, affected Ruff/compileall, PPO dry-run, and validation
protocol dry-run passed. CUDA/model/tokenizer/generation/Trainer/backward/optimizer and
real validation counts were zero. See
`reports/formal_1p5b/ppo_seed42_validation_recovery_eligibility.{md,json}`.

## Stage H.3 validation-only recovery complete

Four independently backed-up runs evaluated checkpoints 8/16/24/32 in order against
the frozen 64-problem validation manifest. They completed 64 rows each and generated
7,533 / 7,848 / 7,663 / 7,497 tokens. Sampled pass@1 was 4.6875% / 3.125% /
3.125% / 3.125%; pass@4 is null/unavailable because validation has one candidate per
problem. Total recovery usage was 0.271514 GPU-hours and CNY 2.411047; all artifact
checksums/backups verified and the GPU returned to 0 MiB with no compute process.

The original training run remains unchanged as
`engineering_failure_after_training / validation_pending` and excluded as an
individual run. The transparent composite linking its complete 32-update training
evidence, four checkpoint SHAs, and four validation artifact SHAs is
`scientifically_complete_with_recovered_validation`. Training was not rerun or
resumed, final test was not run, and checkpoint selection was not changed. See
`reports/formal_1p5b/03_ppo_training.md` and
`reports/formal_1p5b/ppo_seed42_composite_result.json`.

## Stage H.4 valid-answer telemetry repair

The stale `components.valid_answer` lookup was replaced by the existing flat
`valid_answer_component` evidence through one shared PPO/GRPO definition. It counts
positive extracted-answer verifier components over all update completions; it is not
canonical parseable rate. Numerator, denominator, raw source, definition version,
availability, and reason are persisted. Missing/zero-denominator values remain null,
and contradictory aggregates fail finalization. Reward scalar and all optimization
inputs are unchanged.

Forty-four targeted CPU tests, affected Ruff/compileall, and PPO/GRPO formal dry-runs
passed. Historical PPO artifacts/checksums remain unchanged; the recovered composite
stays `scientifically_complete_with_recovered_validation` and needs no rerun.

## Unique next task

The only next task is separately authorized formal GRPO seed 42 from the frozen
`configs/formal_1p5b/resolved/grpo_seed_42.json` SHA256
`3371d23166d01834c67830eb8dfd51a02d4af483687b0b29e941194174099199`.
Do not start it automatically. PPO rerun/resume, seed 123, baseline, final test, and
automatic stage advancement remain unauthorized. Exact scope is in `docs/NEXT_TASK.md`.

All historical runs and successful baselines remain immutable. Missing or unreliable
metrics remain null/unavailable with reasons; noncritical telemetry and presentation
issues are warnings only.
