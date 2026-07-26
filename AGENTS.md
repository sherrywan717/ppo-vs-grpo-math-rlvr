# Repository Guidelines

## Project Mission

This repository is the artifact-first Math RLVR project. Its goal is a reproducible PPO-versus-GRPO comparison for few-shot mathematical reasoning, with matched prompts, reward contracts, completion/token budgets, and auditable run artifacts. Formal experiments target `Qwen/Qwen2.5-1.5B-Instruct`; `Qwen/Qwen2.5-0.5B-Instruct` is the bounded smoke-test model. Countdown is verifier/smoke data, while GSM8K and MATH are training/evaluation data and MATH500 is held out from training.

Never advance to a new paid/GPU stage implicitly. Model download, CUDA initialization, model loading, generation, and PPO/GRPO updates each require the user's explicit scope. A successful stage does not authorize the next stage.

## Portfolio v1 and current mainline

Portfolio v1 freezes four completed formal training/checkpoint-validation runs and the seed-42 Base/PPO/GRPO held-out comparison. GRPO123 and PPO123 final evaluations are `deferred_not_executed`; never infer them from validation or encode them as zero. Public headline claims must remain limited to the paired seed-42 result and the descriptive two-seed validation direction. Pass@1 and pass@4 use independent candidate pools.

The next mainline is a separately versioned GRPO-v2 design phase. It must not edit v1 artifacts/configs or tune from the published held-out test. Credentials, model cache, full checkpoints, optimizer state, and large run archives never enter GitHub. Every public figure must be reproducible from committed CSV/JSON. See `docs/PORTFOLIO_DELIVERABLES.md`, `ARTIFACT_INDEX.md`, and `docs/grpo_v2_roadmap.md`.

## Current Baseline and Milestones

The active improvement branch is `improve/grpo-v2`; portfolio v1 remains frozen on `main`/`v0.1.0-formal-rlvr`. Important milestones are:

- `5a10cbae2abcb066b423b10ff9d327ad1483b75c` — artifact-first Stage D infrastructure, frozen configs, reports, CPU gates, tokenizer audit, trainer builders, and shared PPO/GRPO prompt renderer.
- `6daca223bd17ddc9201e0b8dc7cdc3c677db9b39` — successful Qwen 0.5B CUDA/model-load sanity report.
- The local 0.5B snapshot is revision `7ae557604adf67be50417f59c2c2f167def9a775` under `/root/autodl-tmp/cache/huggingface`; never copy model weights into Git, run artifacts, backups, or `/root/autodl-fs`.
- CPU tokenizer audit, static gates, and CUDA load sanity have passed. Current accepted one-update smokes are GRPO `grpo_single_update_qwen25_05b_20260713T122258Z` and PPO `ppo_single_update_qwen25_05b_20260714T051538Z`; neither is a formal algorithm comparison.
- The guarded GRPO runner requires `--execute --confirm-single-update`; either missing flag fails before delayed model/CUDA imports. Exact completion tokens are guarded through the isolated TRL 0.24.0 shim.

Read `memory.md` before changing execution code or launching another run; it records measured results and known pitfalls.

## Project Structure

Keep application code in `src/math_rlvr/`, tests in `tests/`, reusable entry points in `scripts/`, configs in `configs/`, and Git-safe reports in `reports/`. Runtime datasets, caches, runs, outputs, and checkpoints belong only under `/root/autodl-tmp`. Full run artifacts use `/root/autodl-tmp/runs/math_rlvr/<run_id>/`; Git-safe summaries use `reports/runs/<run_id>/`; static backups use `/root/autodl-fs/math-rlvr-backups/` and must exclude model caches and weights.

The shared prompt contract is one system/user chat followed by an open assistant turn. The required completion envelope is exactly one `<reasoning>...</reasoning>` block followed by exactly one terminal `<answer>...</answer>` block. PPO and GRPO must use the same renderer from `math_rlvr.prompt`.

## Build, Test, and CPU Gates

No command in this section should load a model or initialize CUDA. Run from the repository root:

- `python -m compileall src scripts tests`
- `ruff check .`
- `pytest -q`
- `PYTHONPATH=src python scripts/check_env.py`
- `PYTHONPATH=src python scripts/validate_manifests.py`
- `PYTHONPATH=src python -m math_rlvr.training.ppo --config configs/smoke/ppo.yaml`
- `PYTHONPATH=src python -m math_rlvr.training.grpo --config configs/smoke/grpo.yaml`

The PPO/GRPO commands above are dry-run preflights only. They must not call `train`, `generate`, or `optimizer.step`, and must not load a model. Tests that inject a fake trainer must use the CPU configuration path so TRL does not probe BF16 GPU support.

## Experiment Contracts

Use BF16 LoRA, never QLoRA, vLLM, bitsandbytes, or newly downloaded dependencies unless separately approved. Policy LoRA is `r=16`, alpha 32, dropout 0 on q/k/v/o projections. The PPO value adapter is `r=8`, alpha 16 on q/v plus a trainable scalar score head. Keep prompt/completion limits and budgets in the checked-in configs, and align comparisons using actual completions and generated tokens rather than trainer steps.

The fixed formal reward is format 0.10, parse/semantic validity 0.10, and correctness 0.80. Infrastructure errors abort instead of becoming reward zero. Verifiers must not use `eval`, `exec`, dynamic imports, subprocess execution, or generated-code execution.

## Artifact and Safety Rules

Use `ArtifactManager` and `ResourceMonitor` for GPU runs. Every run must have bounded wall time, completion/token limits, explicit stop reasons, checksums, environment metadata without full environment dumps, resource CSV/JSONL, summaries, logs, and plots. On OOM, NaN/Inf, revision mismatch, target mismatch, timeout, or artifact failure, stop immediately and do not retry unless explicitly authorized.

Never commit model/cache files, full run archives, checkpoints, auth files, tokens, proxy credentials, complete environment dumps, or large binaries. Git-safe reports and small plots may be committed after secret, size, and `git diff --check` audits. Verify archives with both `tar -tzf` and `sha256sum -c`; exclude even empty checkpoint directories when the stage promises no checkpoints.

Do not batch-delete files or directories: `rm -rf`, recursive `rm`, and recursive `rmdir` are prohibited. If deletion is necessary, confirm and remove one explicit path. Do not execute generated code unless a verified isolation backend is active; ordinary subprocess execution is not a security sandbox.

## Coding and Review Style

Use Python 3.12, spaces, Ruff, descriptive `snake_case` functions/modules and `PascalCase` types. Keep modules focused and add deterministic tests for every behavior change. Preserve user changes and avoid unrelated cleanup. Use short imperative commit subjects, preferably Conventional Commits. Pull requests should state motivation, implementation, verification commands, configuration changes, artifact impact, and follow-up work.

### GRPO first-failure repair gate

The first real GRPO attempt, `grpo_single_update_qwen25_05b_20260713T050407Z`, remains a failed immutable historical run at commit `ebc926c432d6778c3b057a0b7b518f7f2eaea5ed`. Its trainer-construction failure was traced to comparing a validated local snapshot path against the original repository ID; artifact finalization separately attempted to serialize the BudgetGuard clock callable.

Execution code now uses the frozen `ValidatedModelSource` boundary: exact 0.5B repo/revision, canonical cache structure, local-only resolver equality, required files, and Qwen2 causal-LM config identity. `BudgetGuard.snapshot()` is the only counter serialization contract and contains primitive JSON values, never callbacks or runtime objects. These repairs do not authorize a real GRPO rerun or PPO.

### GRPO second-failure evidence repair

The second real GRPO attempt, `grpo_single_update_qwen25_05b_20260713T053852Z`, completed the frozen 8-completion/687-token/4-microstep/1-update budget but remained failure because the old checkpoint allowlist rejected the 7,441-byte Trainer metadata file `training_args.bin`. Preserve both failed runs and their backups as immutable evidence.

The repaired runner trusts only the Trainer-created top-level `checkpoint-1`, allows only a canonical non-symlink `training_args.bin` of at most 1 MiB, rejects duplicate adapters, and never deserializes metadata. The sole TRL shim owns ordered completion IDs/masks/text/reward evidence. Frozen beta is 0.0, so KL must be null with a reason. Allocator peaks and nvidia-smi remain distinct. This CPU repair is not authorization to rerun GRPO or start PPO.

### GRPO third-failure allocator repair

The third real GRPO attempt, `grpo_single_update_qwen25_05b_20260713T061248Z`, remains an immutable failed run at commit `438569d97a8636ea6ad13394920663016e01282e`. It stopped before model loading or generation because `CudaAllocatorEvidence` passed the literal string `"cuda:0"` to PyTorch allocator APIs; the resolved training config had no device field.

Allocator device handling is now centralized through `normalize_cuda_device_index`: accepted CUDA values are normalized to a validated non-boolean integer index, while callables, CPU devices, GPU display names, malformed strings, negative values, and out-of-range indices fail closed. Allocator API calls use only that integer; `device_label` and `device_name` are evidence fields only. CPU dry-runs never resolve an implicit current device. This repair does not authorize GRPO or PPO.

### CUDA allocator probe result

The one-shot minimal allocator probe `cuda_allocator_probe_20260713T063028Z` passed on H800 device index 0 after the normalization repair. A 1 MiB uint8 tensor produced 1/2 MiB current allocated/reserved and 1/2 MiB peak allocated/reserved; after release and `empty_cache`, current allocated/reserved returned to zero. The probe loaded no model/tokenizer/data and performed no generation, training, optimizer update, checkpoint, or network access. It does not authorize a GRPO rerun or PPO.

### Successful GRPO smoke and prompt forensic gate

The evidence-complete GRPO run `grpo_single_update_qwen25_05b_20260713T063829Z` succeeded at commit `85776a8290f736b0469f377b0a3d3c4b86cdc7a1`: 2 prompts, 8 completions, 687 generated tokens, 4 microsteps, and one optimizer/global step. All eight outputs were strict format errors, producing zero reward variance and no learning signal. Preserve the run, checkpoint, reports, and backup unchanged.

Historical/production prompt behavior remains versioned as `prompt_v0_grpo_smoke`.
After the separately authorized successful generation-only A/B diagnostic,
`prompt_v1_strict_concise` is approved only as the shared Qwen 0.5B PPO/GRPO smoke
prompt; its production status remains not approved. Both smoke paths must resolve the
same prompt version/hash/renderer and render the same `MathProblem` byte-identically.
Main/formal 1.5B configs remain unactivated. The strict parser and reward/verifier
contract are unchanged.

### Guarded prompt A/B implementation gate

The independent module `math_rlvr.evaluation.prompt_ab` accepts only
`configs/diagnostics/prompt_ab.yaml`. Real generation requires both dedicated flags,
a clean branch, both offline variables, the exact validated local Qwen 0.5B snapshot,
and the fixed 16-completion/2,048-token/120-second/3.5-GiB contract before delayed model
imports. It uses the base model only; Trainer, LoRA, train, backward, optimizer,

### Prompt A/B cleanup semantics

The immutable run `prompt_ab_qwen25_05b_20260713T101918Z` remains failure. Its worker allocator bytes were not persisted because close raised before returning evidence, but parent and manual post-run checks showed PID exit, no compute process and 0 MiB baseline/post memory. Future isolated-worker diagnostics treat allocator current memory as pre-exit evidence and warning only; the non-CUDA parent's post-exit PID/process/baseline check is authoritative. This does not activate v1 or authorize another A/B, GRPO or PPO run.

### Prompt v1 smoke-only activation

The successful rerun `prompt_ab_qwen25_05b_20260713T105428Z` produced v0: 8/8
FORMAT_ERROR and zero nonzero-variance groups; v1: 6 FORMAT_ERROR, 2
INVALID_EXPRESSION, 25% complete envelopes, and two nonzero-variance groups. v1 still
had zero valid-expression rate, number-usage accuracy, pass@1, and pass@4. Therefore
`prompt_v1_strict_concise` is frozen as `approved_for_smoke` and
`not_approved` for production. Only the 0.5B PPO/GRPO smoke selectors use it; v0 and
main/formal configs remain preserved. The next executable step requires a new explicit
GRPO single-update authorization. PPO remains unauthorized and must never start
automatically.

### Prompt v1 GRPO single-update result

The separately authorized run `grpo_single_update_qwen25_05b_20260713T112100Z`
completed the frozen v1 smoke budget: two prompts, eight completions, 276 generated
tokens, four microsteps, and exactly one optimizer/global step. The infrastructure
smoke passed, including the sole safe `checkpoint-1`, finalized evidence, verified
backup, and post-process GPU release.

The learning-signal smoke failed. All eight completions were `FORMAT_ERROR`; each
four-completion problem group had rewards `[0, 0, 0, 0]`, zero variance, and zero
advantage. Finite loss/grad/entropy evidence does not change that conclusion. Keep
`prompt_v1_strict_concise` smoke-only and production-unapproved. PPO remains
unauthorized and blocked pending user review; never enter PPO automatically.

### Staged shaped reward v2 gate

The CPU-only post-smoke intervention `shaped_v2_staged` is selected identically by
the Qwen 0.5B PPO/GRPO smoke configs only. It awards 0.05 each for one usable answer
block, strict protocol, safe valid expression, and exact Countdown number use, plus
0.80 only when the unchanged original canonical verifier returns `VERIFIED_PASS`.
Sparse reward, canonical status, formal metrics, strict parser/verifier semantics,
prompt v1, data, sampling, budgets, and main/formal 1.5B configs remain unchanged.

The immutable eight completions from
`grpo_single_update_qwen25_05b_20260713T112100Z` replay to staged group rewards
`[0.10, 0.10, 0.15, 0.00]` and `[0.10, 0.05, 0.10, 0.05]`, giving nonzero
variance in both groups while retaining eight canonical `FORMAT_ERROR` statuses.
This is a disclosed post-smoke intervention, not a retroactive result. A new GPU GRPO
single update requires separate authorization; PPO remains unauthorized.


### Staged-reward GRPO single-update result

The separately authorized run `grpo_single_update_qwen25_05b_20260713T122258Z` executed exactly once from commit `4f33a76449945e34e5ff4798a884208050fc562a` with frozen `prompt_v1_strict_concise` and `shaped_v2_staged`. Infrastructure, reward integration, and learning-signal smoke gates all passed: two prompts, eight online completions, 276 generated tokens, four microsteps, and exactly one optimizer/global step.

Online group rewards were `[0.10, 0.10, 0.15, 0.00]` and `[0.10, 0.05, 0.10, 0.05]`, with population variances `0.00296875` and `0.000625`; zero-advantage groups were 0. Loss was `0.6453`, finite nonzero grad norm was `4.648100852966309`, and entropy was `0.5070152431726456`. All eight canonical statuses remained `FORMAT_ERROR`, proving that the staged scalar supplied gradient signal without changing strict evaluation semantics. This is a single-update diagnostic, not evidence that the model learned Countdown and not a fair algorithm comparison with the pre-intervention run.

The sole `checkpoint-1` passed adapter-only inventory. PyTorch pre-exit allocator residue (64 MiB allocated / 108 MiB reserved) is a warning; after process exit, `nvidia-smi` reported 0 MiB and no compute process. PPO remains unauthorized and must not start automatically.

### Guarded PPO CPU implementation gate

The CPU-only guarded PPO runner is implemented for TRL 0.24.0. The frozen smoke
contract resolves to four unique prompts, one response per prompt, four completions,
at most 512 generated tokens, one PPO epoch, one minibatch, and exactly one
optimizer/update/global step. `generation.num_generations=4` is a shared-schema field
that TRL PPO ignores; the resolved contract must record that fact and must never infer
16 completions. The sole compatibility shim applies YAML top-p 0.95 to TRL's internal
generation config.

Policy and value models must be distinct objects loaded local-only from the same
validated 0.5B snapshot. Policy trainables are q/k/v/o LoRA only; value trainables are
q/v LoRA plus the scalar score head. The frozen reference uses the policy with its
adapter disabled, the verifier reward model is parameter-free, and the optimizer
parameter set must equal exactly the union of policy/value trainables. The sole custom
`checkpoint-1` is role-separated and adapter/head-only; full base or optimizer
weights are forbidden.

Real PPO requires `--execute --confirm-single-update`, a clean
`pivot/math-rlvr` branch, both offline variables, the fixed local snapshot, and a new
explicit user authorization. The CPU implementation/fake execute did not initialize
CUDA, load a model, generate a completion, or execute an optimizer update. Never run
the documented real PPO command automatically.


### Accepted PPO smoke and nullable telemetry repair

The user accepted immutable run `ppo_single_update_qwen25_05b_20260714T051538Z` as
`execution_success/nonessential_telemetry_warning`. It completed four prompts, four
responses/completions, 141 generated tokens, and exactly one PPO epoch, minibatch,
update, optimizer step, and global step. Reward population variance was
`0.00046875`; the checkpoint is role-separated adapter/head-only. The launcher exit
code 1 occurred after training when formal artifact finalization rejected TRL's
nonessential `val/ratio_var=NaN`. Never rerun this smoke or rewrite its original
failure report, launcher output, completions, checkpoint inventory, or other evidence.

The TRL compatibility shim has a narrow nullable nonessential telemetry allowlist.
Only `val/ratio_var` may normalize a non-finite value to standard JSON `null` with
`available=false`, its original raw key, non-finite classification, and an explicit
reason. It is never coerced to zero. Required metrics, rewards, losses, counters,
budgets, and all unreviewed metric keys remain fail-closed on NaN/Inf. Persisted
trainer history uses the sanitized copy; the original in-memory history is not
mutated.

### Stage D technical-smoke completion

`reports/stage_d/smoke_readiness_matrix.{json,md}` establishes that the accepted PPO
run has a current GRPO technical-smoke counterpart:
`grpo_single_update_qwen25_05b_20260713T122258Z`. Both used the exact fixed Qwen 0.5B
revision, `prompt_v1_strict_concise`, `shaped_v2_staged`, common sampling and policy
LoRA, completed a real update, produced nonzero reward variation, and passed
checkpoint safety. Stage D technical smoke is complete.

This is not an algorithm-effect result. PPO sampled four unique prompts once each;
GRPO sampled two prompts four times each, with different completion/token budgets and
variance aggregation. A future 0.5B matched pilot must freeze common prompt allocation,
actual completion/token budgets, explicit parser/verifier hashes, and multiple seeds,
then receive new explicit GPU authorization. Do not execute it automatically, and do
not enter 1.5B or formal training implicitly.


### Matched 0.5B pilot configuration freeze

The CPU-only matched pilot is frozen under `configs/pilot/`. Its manifest SHA256 is
`0235210e038bc27ebf2e7218691f36f09c8e11f0bbc743f46a5318a279f6bc1f` and selects
Countdown train IDs 0–3 in original order, without gold/construction fields or
performance-based selection. Each algorithm/seed resolves to four prompts × four
responses, 16 completions, a 2,048-token cap, and one outer/optimizer/global step.
Approved seeds are 42, 123, and 2026; only the six committed resolved JSON files and
their registry SHA256 values are accepted. Every report must say
`Matched 0.5B pilot - not the final benchmark`.

Parser `strict_completion_parser_v1` and verifier `countdown_ast_fraction_v1` hashes
come from canonical semantic JSON. Resolved configs and future run manifests record
both. PPOConfig never receives `num_generations`; its local batch is 4 × GA4 = 16 and
rollout forward batch is 4. GRPOConfig resolves generation batch 16, four generations,
per-device batch 4, GA4, and four steps per generation.

The two matched-pilot execution correctness items are resolved. For PPO pilot only,
the sole TRL compatibility shim replaces `self.dataloader` immediately after
`PPOTrainer.__init__` with an explicit `SequentialSampler`, batch size 16,
`drop_last=True`, and `num_workers=0`, using only the existing Trainer
`accelerator.prepare_data_loader`. Prepared-batch metadata and the actual iterator
consumed by `train()` must match all 16 prompt-major records exactly; model and
optimizer are never prepared twice.

PPO/GRPO evidence now uses immutable `ExpectedRunContract` profiles selected only by
exact repository config path and SHA256: Stage D PPO=4, Stage D GRPO=8, matched
PPO=16, matched GRPO=16. Profiles bind model, LoRA, sampling, prompt, reward,
parser/verifier, manifest, completion/token, and update identities. Overflow fails
online; final counts require equality. Main/1.5B and CLI-provided numeric widening are
rejected. Do not bypass the profile resolver or scatter additional TRL monkey patches.

The user subsequently authorized the frozen six-run suite. Its first and only executed
command, PPO seed 42 run `ppo_matched_0p5b_seed42_20260714T073357Z`, failed before
generation or training. The delayed PPO dataset renderer called
`render_training_prompt`, whose selector treats only `smoke-*` names as eligible for
the strict-concise prompt and rejected the valid pilot name as main/formal. Counters
were 0 completions, 0 tokens, and 0 update/optimizer/global steps; no checkpoint was
written. The worker exited, the GPU returned to 0 MiB with no compute process, and the
failure archive verified successfully.

The no-retry and suite-stop rules were applied: runs 2–6 were not executed. Preserve
this failed run and its reports/backup unchanged. The prompt-stage routing mismatch is
a real execution-chain blocker; any repair requires separate CPU-only authorization,
tests, a new commit, and a new explicit GPU authorization. Never retry this command,
continue the pilot suite, rerun Stage D, download 1.5B, or start formal training from
the present authorization.

### Pilot-aware prompt routing repair

The CPU-only repair after failed run
`ppo_matched_0p5b_seed42_20260714T073357Z` removes experiment-name prefix routing.
`ValidatedExperimentScope` is now selected only from an exact repository config path
and raw SHA256. Protected Stage D and matched-pilot scopes come from
`ExpectedRunContract`; exact main config paths resolve to `MAIN_FORMAL` without
gaining an execution profile. Unknown, spoofed or drifted scope evidence fails closed.

PPO and GRPO share the same explicit scope selector. Before any snapshot/model handling,
their execute paths render and validate 16 PPO Python message rows or four GRPO rows,
including frozen prompt hashes and all comparison keys. The delayed dataset builders
receive the same scope, and future artifacts persist `prompt_scope_preflight.json`.
Main configs renamed to `pilot-*` still reject smoke-only v1; Stage D and the matched
pilot allow v1; historical v0 replay is unchanged.

CPU gates passed with CUDA uninitialized and zero real model/tokenizer/generation/train
calls. The historical failed run, reports and verified backup remain immutable and are
not a scientific PPO result. This repair does not authorize retrying it, running any
pilot job, or entering 1.5B. A new suite requires a new explicit GPU authorization.


### PPO pilot collator mapping repair

The separately authorized new matched suite first exposed and preserved a second
immutable pre-generation failure, `ppo_matched_0p5b_seed42_20260714T082003Z`. The
base TRL/Transformers collator correctly returned a `BatchEncoding`, but the ordered
metadata wrapper incorrectly required concrete `dict`, raising before generation or
update. The run and its failure backup remain excluded from scientific aggregation.

The CPU-only repair is limited to `training/trl_compat.py`: the wrapper accepts the
actual mutable Mapping and prepared-batch validation accepts Mapping. It returns the
same `BatchEncoding`; token tensors, padding, dtypes and response boundaries are
unchanged. Metadata stays outside model kwargs. Real local-tokenizer and CPU
Accelerator tests verify the 16-row SequentialSampler order; full CPU gates passed.
This repair by itself changes no frozen config, manifest, identity or budget.

### PPO pilot epoch/minibatch guard repair

The immutable third pilot failure `ppo_matched_0p5b_seed42_20260714T085240Z` completed
16 rollouts and 574 tokens, then stopped when the old guard reported a second
epoch/minibatch. CPU audit proved the frozen JSON, builder kwargs and actual TRL 0.24.0
`PPOConfig` were all one epoch and one minibatch. TRL invoked the Accelerator optimizer
wrapper once for each of four microbatches; the old hook incorrectly counted those calls
as four logical loop entries. Preserve this run and the two earlier failed seed-42 runs
unchanged and exclude all three from scientific aggregation.

The sole TRL shim now records a logical `(outer_update, epoch_index, minibatch_index)`
only at `accelerator.sync_gradients`, after exactly four microbatch calls. Duplicate loop
keys are idempotent. Actual second epoch/minibatch/outer keys and a second optimizer step
still fail closed before a second synchronized update. Runtime Trainer-derived batch
fields are checked against the exact protected profile. Frozen configs, identities and
budgets are unchanged. Full CPU gates passed without CUDA, model loading, generation,
training, backward or a real optimizer step. The user's continuous authorization permits
one new PPO seed-42 run only after this repair is committed, backed up and the worktree
is clean; a failure stops the suite without retry.


### PPO pilot sync-boundary failure

After the CPU loop-budget repair, the newly authorized run `ppo_matched_0p5b_seed42_20260716T111934Z` executed once.
It produced 16 completions, 574 generated tokens and 16 rewards, then failed before an
optimizer/update/global step because the real Accelerator reported `sync_gradients=true`
on the first optimizer-wrapper call; the shim expected four calls before that boundary.
No completion rows, metrics or checkpoint were finalized, so this is not a scientific
PPO result and must remain excluded from aggregation. The verified failure archive SHA256
is `29a6e478a6692782700c23900b2c0836af5dd132961f835867834461490014e1`. GPU returned
to 0 MiB with no compute process. The command was not retried and runs 2--6 were not
executed. Preserve this run plus the three earlier PPO seed-42 failures unchanged. No
further repair or GPU execution is authorized by this stopped suite.


### Accelerate backward-event PPO guard

CPU reproduction with Accelerate 1.14.0 proved that the bottom optimizer is called
once, first observed with `sync_gradients=true`, after four ordinary GA4 microbatches.
It also reproduced TRL's consumed single-batch loader: default end-of-dataloader sync
would make every inner microbatch a real update. The sole PPO compatibility shim now
sets `sync_with_dataloader=false`, counts four actual `accelerator.backward` events
with batch sizes 4 and sync trace false/false/false/true, and separately guards one
bottom optimizer step. Epoch/minibatch/update/global caps remain 1. Frozen contracts
and four historical seed-42 failures are unchanged. A new real PPO seed-42 run is
authorized only after this CPU repair commit, verified static backup and clean GPU
preflight; any failure stops without retry or further repair.


### Matched PPO pilot seed 42 success

Run `ppo_matched_0p5b_seed42_20260716T114710Z` successfully completed the frozen matched PPO budget from commit
`3c88fb1b19a5dcfcde2592ac9a17a4f37596fb73`: 16 completions, 574 tokens, four
batch-4 backward events with sync false/false/false/true, and exactly one
epoch/minibatch/bottom-optimizer/update/global step. Completion evidence and the safe
policy/value-adapter plus scalar-head checkpoint are complete. All four reward groups
had nonzero variance, while canonical accuracy remained zero; this is a valid pilot
execution, not proof of learning or superiority. The verified backup SHA256 is
`dd1833ea6fa75a6a8af1d7fba366b05e498b0524ee726a526eb7fa294f89b7f6`. Parent GPU
release passed. Four historical PPO failures remain excluded. The active authorization
allows the suite to continue next with frozen GRPO seed 42, once this report is
committed and the worktree/GPU are clean.

### Matched GRPO pilot seed 42 success

Run `grpo_matched_0p5b_seed42_20260716T115219Z` executed its frozen command once and
completed 16 comparison-keyed completions, 545 generated tokens, four microsteps and
exactly one optimizer/update/global step. All four reward groups had nonzero variance;
canonical accuracy remained zero. The adapter-only checkpoint, full/Git-safe checksums,
verified independent backup and parent GPU release passed. This is a valid matched-pilot
single update, not proof of learning or GRPO superiority. The suite may proceed only to
the next frozen command, GRPO seed 123; the four historical PPO failures stay excluded.

### Matched GRPO pilot seed 123 success

Run `grpo_matched_0p5b_seed123_20260716T115714Z` executed once and passed with 16
completion/evidence rows, 724 generated tokens, four microsteps and one optimizer /
update / global step. All 16 comparison keys and four nonzero-variance groups passed;
canonical pass accuracy remained zero. The adapter-only checkpoint, checksums, backup
and GPU release passed. It is a pilot single update, not a learning or superiority
claim. Continue only with the next frozen command, PPO seed 123.

### Matched PPO pilot seed 123 success

Run `ppo_matched_0p5b_seed123_20260716T120000Z` executed once and passed: 16 rows,
565 tokens, four batch-4 backward events with sync false/false/false/true, and one
epoch/minibatch/bottom optimizer/update/global step. One reward group had zero variance;
the other three did not, and zero canonical accuracy remains an experimental outcome.
Finite losses, exact model-role optimizer union, safe checkpoint, verified backup and
GPU release passed. This is not proof of learning or PPO superiority. Proceed only to
the next frozen command, PPO seed 2026.

### Matched suite recovery reconciliation

Primary evidence establishes `ppo_matched_0p5b_seed42_20260716T114710Z` as the
valid suite-position-1 scientific success. A derived recovery row incorrectly called
it a historical failure and was corrected without modifying the run. Real historical
failures remain `073357Z`, `082003Z`, `085240Z`, and `111934Z`.

### Matched PPO pilot seed 2026 success

Run `ppo_matched_0p5b_seed2026_20260716T122924Z` passed with 16 completions,
512 tokens, four batch-4 backward events and exact one-step counters. Required
metrics, adapter/head checkpoint, backup and GPU release passed. All completions were
FORMAT_ERROR; this is a scientific outcome, not proof of learning or superiority.

### Matched GRPO pilot seed 2026 success

Run `grpo_matched_0p5b_seed2026_20260716T123716Z` passed with 16 completions,
703 tokens, four microsteps and one optimizer/update/global step. Required metrics,
group evidence, adapter checkpoint, backup and GPU release passed. This remains a
matched pilot result, not a final benchmark or superiority claim.

### Matched 0.5B pilot aggregate complete

All six frozen PPO/GRPO runs are scientific execution successes and are aggregated
in `reports/pilot_0p5b/final_report.md`. Historical engineering failures remain
excluded. Total measured usage was 0.023131886 GPU-hours and CNY 0.205411. Every
run had zero canonical pass@1/pass@4; the pilot validates execution and evidence,
not learning or algorithm superiority. The next appropriate action is CPU-only 1.5B
GSM8K+MATH configuration freezing, not automatic training.

### Stage E formal 1.5B configuration freeze

The CPU-only formal design is frozen under `configs/formal_1p5b/` for exact model
`Qwen/Qwen2.5-1.5B-Instruct` revision
`989aa7980e4cf806f80c7fef2b1adb7bc71aa306`. Stage E queried metadata only; it did not
download weights or load a model/tokenizer, initialize CUDA, generate, instantiate a
Trainer, run backward, or step an optimizer. Future execution must use a separately
authorized exact local snapshot with both offline variables and `local_files_only=true`.

Formal PPO/GRPO share `prompt_v2_formal_math`, `shaped_v3_domain`, policy LoRA r16/
alpha32/dropout0 on q/k/v/o, temperature 0.8, top-p 0.95, 256 completion tokens, and
seeds 42/123/2026. The reward is answer-block 0.05, strict protocol 0.05, domain-valid
answer 0.10, and canonical correctness 0.80. Countdown exact-number-usage is forbidden
for GSM8K/MATH. Canonical pass metrics ignore scalar shaping; `INFRA_ERROR` fails closed.

The formal data registry selects 64 GSM8K + 64 MATH train, 32+32 validation, 200 GSM8K
+ 200 MATH500 test, and fixed 50+50 pass@4 subsets. There is zero content-hash overlap
across core splits and zero train/validation overlap with all MATH500. The deterministic
32-update schedule is two GSM8K followed by two MATH records per update, preserving
within-domain manifest order. Preserve the disclosed historical validation provenance:
manifest `source_split=validation`, physical source split `train`, selection split
`validation`; never rewrite historical manifests to hide it.

Each algorithm/seed is frozen to 32 updates, 512 completions, a 131,072-token cap,
32 optimizer/global steps, and checkpoints at 8/16/24/32. PPO uses 512 prompt-major
episodes, rollout 16, batch4/GA4, one epoch, one minibatch, and never receives
`num_generations`. GRPO uses 128 prompts, generation batch16, four generations,
batch4/GA4, one iteration, no shuffle, drop-last, and zero workers. PPO's separate
value base, q/v r8 adapter and scalar head are an explicit cost/architecture difference;
checkpoints remain role-separated adapter/head-only.

Stage F pinned CUDA/model-load sanity and the shared untrained baseline are complete.
The remaining order is separately authorized: PPO42; GRPO42; seed-42 validation review;
then GRPO123, PPO123; fixed step-32 final test; CPU aggregate. Test data cannot tune the
prompt, reward, hyperparameters, or checkpoint. A successful stage never authorizes the
next one.

### Formal 1.5B four-run amendment

The six-run Stage E plan at commit `499fea9f6de6f991229f15b949e23e63c496e6cc`
remains historical evidence. The approved active suite is PPO42, GRPO42, a seed-42
learning-signal review, GRPO123, PPO123, four fixed step-32 final evaluations, and CPU
aggregation. After the disclosed prompt-length capacity amendment, the exact active-suite canonical
hash is `1d7c29f76d9bfbf11e1838cd6b0bc8f3da6d0133e0605420e9ed838de729d600`.
Seed-2026 descriptors remain byte-identical and `reserved_not_scheduled`; never place
them in the active registry, queue, costs, or aggregate without a new amendment.
Two-seed results may report raw values, mean, sample SD, paired deltas, and
problem-level bootstrap intervals, but not statistical significance or general PPO/
GRPO superiority. This CPU amendment authorizes no model download, CUDA, evaluation,
or training.

### Formal multi-update CPU runtime

The formal runtime accepts only the four active config path/SHA pairs. It requires 32
updates, 512 ordered completion keys, the 131,072-token cap, and checkpoints plus
validation at 8/16/24/32. Resume is same-run only and binds the exact identity/counter/
pair-key prefix. The TRL shim supports 32 PPO sequential batches, 128 GA4 backward
events, and append-only PPO/GRPO evidence without weakening Stage D profiles. Formal
evaluation freezes two baseline seeds and four step-32 final checkpoints. CPU fakes
may exercise the runtime; no model download, CUDA, generation, Trainer, backward, or
optimizer is authorized by this implementation commit.

### Formal 1.5B standing rules

The formal project goal is a fair, auditable PPO-versus-GRPO Math RLVR comparison on
`Qwen/Qwen2.5-1.5B-Instruct`. The active scientific suite has exactly four training
runs: PPO seed 42, GRPO seed 42, GRPO seed 123, and PPO seed 123. Seed 2026 remains
`reserved_not_scheduled`. Exact config path/SHA membership and execution order come
from `configs/formal_1p5b/active_suite.json`; never infer them from names or chat.

Frozen final test data is baseline/final-evaluation only. It must never select or tune
prompts, rewards, data, sampling, hyperparameters, checkpoints, stopping rules, or
code changes. Validation may provide the predeclared learning-signal review, while the
final checkpoint remains fixed at step 32.

Scientific metrics, exact greedy/sampled pass@1/pass@4 definitions, per-update and
per-problem evidence, CSV/JSON-rebuildable figures, Markdown analysis, time, VRAM,
GPU-hours, and CNY cost are core deliverables. Missing or unreliable metrics must be
standard JSON `null`/CSV empty with `available=false` and a reason; never fabricate
zero. The authoritative checklist is `docs/PORTFOLIO_DELIVERABLES.md`.

Only training correctness, PPO/GRPO fairness, safety, same-run recovery, or report
truthfulness can block. Optional telemetry, PTY capture, SVG whitespace, CSV CRLF,
and other noncritical guards are warnings. Do not expand infrastructure for them.

Every failed attempt is immutable evidence: never overwrite it, splice partial rows
into another run, or include it in scientific statistics. Every completed stage,
whether success or failure, must synchronize `PROJECT_HANDOFF.md`, `memory.md`,
`docs/NEXT_TASK.md`, `reports/formal_1p5b/run_registry.csv`, and the concise README
stage summary before handoff.

### Current formal handoff

Stage F snapshot download/CUDA sanity and the two-seed frozen base baseline are
complete. The current active-suite canonical SHA256 is
`1d7c29f76d9bfbf11e1838cd6b0bc8f3da6d0133e0605420e9ed838de729d600`.
Two historical seed-42 baseline failures remain excluded; the post-freeze prompt-cap
amendment is public and capacity-only. See `PROJECT_HANDOFF.md` and
`docs/NEXT_TASK.md` for current identities and the exact boundary.

When sources conflict, use this order: Git commits plus actual configs/manifests and
saved artifacts; `PROJECT_HANDOFF.md`; `docs/PORTFOLIO_DELIVERABLES.md`;
`docs/NEXT_TASK.md`; this file and `memory.md`; historical chat. Git and original
artifacts always prevail.

### First formal PPO seed-42 failure

The single authorized Stage H attempt `ppo_formal_1p5b_seed42_20260718T150510Z` is an immutable engineering
failure and is excluded from scientific aggregation. The real PPO loop reached the
live step-8 checkpoint boundary, then checkpoint serialization raised because TRL did
not expose `grad_norm` and the formal adapter incorrectly treated that optional metric
as required. Automatic retries were zero; GRPO and all later stages were not started.

Finalized completion, metric, and verifier JSONL files are empty and formal counters
are zero, so generated tokens, reward/loss/entropy histories, and completion counts are
unavailable rather than inferred from transient console output. The partial
`checkpoint-8` contains policy/value adapters and the scalar head only; it lacks
optimizer/scheduler/RNG/runtime/prefix inventory state and is not resume-capable. It
contains no full base-model weights. The verified failure backup SHA256 is
`76896c5b3db3ee4439566b8b68c0cad798af5b5610f393138aa23eba6c40debb`; post-process GPU release was 0 MiB/no compute process.

The unique next task is a bounded CPU-only repair: missing optional `grad_norm` must
serialize as null/unavailable with a reason, and append-only per-update primary evidence
must be persisted before checkpoint serialization can discard it. Preserve the failed
run and partial checkpoint unchanged. No new PPO attempt or GRPO run is authorized.

### Stage H.1 optional-telemetry/evidence repair

The bounded CPU-only repair after the first formal PPO seed-42 failure is complete.
TRL 0.24.0 exposed neither `grad_norm` nor `train/grad_norm`; aggregate/policy/value
grad norms are now nullable optional telemetry with explicit availability, reason, and
raw-key evidence, while provided NaN/Inf remains fail-closed. The guarded PPO per-update
log callback now validates and atomically persists the existing completion and metric
prefixes before checkpoint callbacks. A fake step-8 failure preserves 128 ordered
completion rows and eight metric/counter rows. No artifact type, checkpoint format,
scientific identity, budget, or historical run changed. A new real PPO seed-42 attempt
requires explicit authorization; never start it or GRPO automatically.

### Second formal PPO seed-42 failure

The separately authorized run `ppo_formal_1p5b_seed42_20260719T131800Z` executed once
from clean HEAD `1d31f56386857909c881bba1a7c5302166cf9682`. Incremental evidence
preserved all 32 updates, 512 completions, and 51,369 training tokens, and trusted
checkpoint directories 8/16/24/32 contain no base weights. After training, the backend
replayed checkpoint 8 while the observer already held update 32, so the cadence guard
failed before any checkpoint validation. Preserve this run and backup unchanged and
exclude it from scientific aggregation. Do not resume/evaluate its checkpoints or run
another GPU job automatically without a later explicit authorization.

### Stage H.2 checkpoint/validation cadence repair

Training progress and checkpoint-validation cadence now have independent ordered
state. Training remains exactly updates 1..32 with unchanged optimizer/global,
completion-key and token-budget checks. Checkpoint and validation steps remain exactly
8/16/24/32, once each and in order; validation requires the corresponding trusted
same-run checkpoint but may run online or after training completes. Validation
completions/tokens never alter training counters.

A read-only audit of `ppo_formal_1p5b_seed42_20260719T131800Z` validated 32 metric
rows, 512 completion rows, 51,369 tokens and all four checkpoint inventories, SHA256
values, prefixes and frozen identities. The run remains
`engineering_failure_after_training / validation_pending`, while its checkpoints are
eligible only for separately authorized validation-only evaluation. Training rerun is
not required and training resume is not authorized. Never modify the original run or
start validation/GRPO automatically; follow `docs/NEXT_TASK.md`.

### Stage H.3 recovered PPO checkpoint validation

The separately authorized validation-only sequence evaluated the trusted PPO seed-42
checkpoints in strict order 8/16/24/32. Each run completed the same frozen 64-problem
validation manifest once, with one candidate per problem; pass@4 is therefore
`null/unavailable`, never inferred from pass@1. The four runs total 256 validation
completions and 30,541 validation tokens, which remain outside the 512-completion and
131,072-token training budgets.

The original training run `ppo_formal_1p5b_seed42_20260719T131800Z`, its failure
summary, checkpoints, and checksums remain immutable with status
`engineering_failure_after_training / validation_pending`. The derived composite is
`scientifically_complete_with_recovered_validation`: training was neither rerun nor
resumed, final test was not run, and validation did not change checkpoint selection.
The recovered report also proves the runtime `valid_answer_rate` used a stale nested
component lookup and wrote misleading zeros despite complete flat reward evidence.
This report-truthfulness defect must receive a bounded CPU-only field-mapping repair
before formal GRPO seed 42 can be authorized; never start GRPO automatically.

### Stage H.4 formal valid-answer telemetry mapping

Formal PPO and GRPO now share `formal_domain_valid_answer_component_v1` for native
`valid_answer_rate`: the numerator counts flat `valid_answer_component > 0` rows and
the denominator is all completion rows in the update. This is extracted-answer probe
validity, not canonical parseability; `INVALID_NUMBER_USAGE` is excluded. Zero or
missing denominators/evidence remain null/unavailable with reasons, never false zero.
The metric is reporting-only and cannot affect reward, loss, optimizer, selection, or
stopping. Historical PPO artifacts and their misleading native zeros remain immutable;
the scientifically complete recovered composite is not downgraded or rerun. Formal
GRPO seed 42 still requires separate explicit authorization via `docs/NEXT_TASK.md`.

### Stage I formal GRPO seed-42 result

Run `grpo_formal_1p5b_seed42_20260720T031006Z` is a scientific success from commit
`548e2d371cbc09d5527aed3ed9dbf0ac1ad94a1d`: 32 updates/optimizer/global steps,
512 training completions, 50,773 rollout tokens, 128 reward groups, safe checkpoints
and frozen 64-problem validations at 8/16/24/32. Validation pass@1 was 3.125%,
4.6875%, 6.25%, and 7.8125%; pass@4 is null/unavailable because there is one
candidate per problem. Training had 101/128 nonzero-variance groups and 27
zero-advantage groups, so it was not a no-learning-signal run.

The verified backup SHA256 is
`b584363595f99c1d3b61a7b6cc088cdda7ac38a29169058df7b30cd38bea5023`; GPU
returned to 0 MiB/no process. No final test ran. The same-seed PPO/GRPO analysis is
descriptive only: loss and native entropy definitions differ, one seed is not a
significance claim, and test baseline is not a validation delta. The sole next queue
position is formal GRPO seed 123 and still requires separate explicit GPU authorization.


### Stage J formal GRPO seed-123 result

Run `grpo_formal_1p5b_seed123_20260720T035927Z` is a scientific success from commit
`e54d84d9795ad74da855e6fdf6e8a15700d36d1d`: 32 update/optimizer/global steps,
128 microsteps, 512 training completions, 52,284 rollout tokens, safe checkpoints and
frozen 64-problem validations at 8/16/24/32. Validation pass@1 was 4.6875%,
7.8125%, 9.375%, and 9.375%; pass@4 is null/unavailable because there is one
candidate per problem. Training retained 100/128 nonzero-variance reward groups.

The verified backup SHA256 is `e78eb0719bc93c1076bd06e50037cc453cbaa5103cf1e1fbfc9e8151212e521a` and the GPU returned to 0 MiB/no
process. No final test ran and no seed-42-driven tuning occurred. The two-seed GRPO
review is descriptive only and does not establish statistical significance. The sole
next queue position is formal PPO seed 123, which requires separate explicit GPU
authorization.


### Stage K formal PPO seed-123 result

Run `ppo_formal_1p5b_seed123_20260720T043732Z` is a scientific success from commit
`f6a62eeb8ce59c438f355b675db493552044de18`: one attempt completed 32 update /
optimizer / global steps, 512 training completions, 51,969 rollout tokens, safe PPO
policy/value adapters plus scalar-head checkpoints, and frozen validation at
8/16/24/32. Validation pass@1 was 3.1250%, 4.6875%, 4.6875%, and 4.6875%;
pass@4 is null/unavailable because there is one candidate per validation problem.

The verified backup SHA256 is
`689924eaa4392a4806f9d1adaa2bbf890b76d6813a6edfeafc2ca50213bc63c0`; parent
GPU release was 0 MiB/no process. All four active formal training runs are complete.
The four-run comparison is descriptive only: two seeds do not establish statistical
significance, PPO/GRPO loss and native entropy definitions are not directly comparable,
and formal test has not run. The only next queue item is the separately authorized
fixed step-32 PPO seed-42 final evaluation; never start it automatically.

### Stage L1 PPO seed-42 held-out final evaluation

The fixed checkpoint-32 evaluation `ppo_final_formal_1p5b_seed42_20260721T022152Z`
is a scientific success: 800/800 completions, 98,018 exact tokens, sampled pass@1
3.75%, and independent-pool pass@4 9.0%. The matching base seed-42 values are 4.0%
and 10.0%; paired deltas are descriptive and were not used for tuning or checkpoint
selection. Greedy remains null/unavailable because the frozen protocol has no greedy
pool. The verified backup SHA256 is
`04fcb03b22ab74e865e2627c0e02460b62c6c731e2245d054aefe5ff6b562fc1`; GPU
returned to 0 MiB/no process.

The earlier process `ppo_final_formal_1p5b_seed42_20260720T052931Z` was externally
terminated by a host power/network outage at 429 rows/41,144 tokens. It is immutable,
excluded, and never resumed or mixed with the successful run. The next queue item is
GRPO seed-42 checkpoint-32 final evaluation and requires separate authorization; never
start it or either seed-123 final evaluation automatically.

### Stage L2 GRPO seed-42 held-out final evaluation

The fixed checkpoint-32 run `grpo_final_formal_1p5b_seed42_20260721T034104Z` is a
scientific success: 800/800 completions, 94,288 exact tokens, sampled pass@1 7.0%, and
independent-pool pass@4 14.0%. Its Base/PPO/GRPO candidate identities match exactly;
paired pass@1 deltas favor GRPO on this single frozen seed, while pass@4 intervals span
zero. Do not generalize this into cross-seed algorithm superiority or use it for tuning.

Checkpoint manifest SHA
`c0c1a3dc04b28d8d42463009b728b3ffd144e677c35e5d3ace2749fc925fa65a`, all runtime
checksums, and backup SHA
`97be0c2f1931690fb631dec557eb17201df12b177810c0b270192c46e6920e48`
verified; GPU returned to 0 MiB/no process. The only next queue item is separately
authorized GRPO seed-123 checkpoint-32 final evaluation. Never start it, PPO123, final
aggregation, seed2026, baseline, or training automatically.

### GRPO-v2 Stage N frozen contract

Portfolio v1 remains immutable on `main`/`v0.1.0-formal-rlvr`. The versioned `improve/grpo-v2` experiment uses seed 42 only: a 256-example, one-epoch format/solution warm-start followed by 128 GRPO updates, 2,048 training completions, and a 524,288-token cap. It preserves the v1 model revision, prompt, reward, parser/verifier, sampling, completion cap, and policy LoRA. Checkpoints/dev occur at 32/64/96/128; only the frozen 128-problem `dev_v2` lexicographic rule may select a checkpoint.

The new core manifests are strictly hash- and source-identity-disjoint from all v1 data. MATH500 hidden-test capacity after exclusion is 3/50/65/88/94, so the preregistered test allocation is 3/33/43/59/62 and nested subset allocation is 3/8/10/14/15. Level 1 is `diagnostic_only_small_n`. Candidate 0 is shared between 400-problem pass@1 and the 100-problem nested pass@4 subset; each evaluated model has 700 completions. Hidden test is frozen before training and must never influence prompt, reward, curriculum, hyperparameters, checkpoint selection, or another training attempt. Public execution manifests contain no gold/solution; trusted verifier data stays outside Git. No Stage N artifact authorizes tokenizer/model loading, CUDA, generation, warm-start, GRPO, dev, or hidden-test execution.

### GRPO-v2 warm-start capacity and execution gate

Stage O.1 preserved a failed CPU tokenizer audit: 48/256 assistant targets exceeded the old 256-token target cap and one prompt exceeded the old 832-token cap. The authorized `post_freeze_capacity_amendment` changes only the independent prompt/target gates to 928/640; actual combined length remains capped at 1,088. No target, solution, prompt, sample, order, epoch, batch, GA, step, LoRA, reward, parser/verifier, or GRPO budget changed. The amended 256/256 audit passed with no truncation.

Future warm-start execution uses `math_rlvr.training.warmstart`, exact config/SHA, dual confirmation, clean `improve/grpo-v2`, offline local snapshot, completion-only labels, 256 unique samples, 64 microsteps, 16 optimizer/global steps, and one epoch. Checkpoints contain the policy adapter plus trusted training/RNG state, never base weights. GRPO-v2 imports only the adapter and its SHA and initializes a fresh optimizer. Secondary nested pass@10 is frozen on 50 problems inside pass@4; it is exploratory and cannot select checkpoints or trigger retraining. Stage O.2 itself authorizes no CUDA/model load/training/evaluation.

### GRPO-v2 shared unbiased pass@k gate

Stage O.3 supersedes the never-executed O.2 50-problem pass@10 design without rewriting history. The active hidden-evaluation subset is the byte-identical 100-problem Stage N pass@4 manifest, sampled once per problem with ten exchangeable candidates. Compute k=1/4/10 using exact `1-C(10-c,k)/C(10,k)` problem estimates, then average across problems. Keep `candidate0_accuracy_all_400` distinct from `unbiased_pass_at_1_subset_100`. Each model has exactly 1,300 rows and the four-model ledger has 5,200. Missing candidate evidence fails closed; test results never select checkpoints or trigger retraining. No Stage O.3 artifact authorizes GPU work.

### GRPO-v2 Stage P warm-start result

Run `warmstart_grpo_v2_seed42_20260722T051218Z` is the immutable seed-42 warm-start `scientific_training_success`: 256 unique samples, one epoch, 64 microsteps, 16 optimizer/global/scheduler steps, and 46,058 active supervised tokens. Checkpoint-16 is policy-adapter-only plus trusted optimizer/scheduler/RNG/runtime/cursor state; its artifact SHA is `507749d393f38690915a76228b4c53a8b5c8927d40aada9f2768a90334d892f0`. Never rerun this training to repair reporting or evaluation.

Base and warm-start dev-v2 are `not_executed_evaluator_unavailable`, not zero: the authorized HEAD had no frozen model-bound dev-v2 evaluator. The next task is CPU-only evaluator implementation/freeze, followed by separate GPU authorization for exactly two matched dev runs. Do not start GRPO-v2 or hidden test before matched dev evidence exists.

### Matched GRPO-v2 dev evaluator

The frozen shared evaluator `math_rlvr.evaluation.grpo_v2_dev` is the only authorized model-bound path for `dev_v2`. Base and warm-start modes must share the exact 128-problem order, prompt/parser/verifier/reward/sampling identity, candidate index 0, and per-problem seeds. Base forbids adapters; warm-start accepts only checkpoint-16 artifact `507749d3...92f0` and policy adapter `44066dd1...72b9`. Each run is inference-only with 128 completions and independent token/resource accounting. Pass@4/pass@10 are unavailable on this single-candidate dev protocol. Warm-start training is immutable and must never be rerun to repair evaluation.

### GRPO-v2 matched dev result

Base dev run `base_dev_grpo_v2_seed42_20260722T060500Z` and warm-start dev run `warmstart_dev_grpo_v2_seed42_20260722T060500Z` are immutable scientific successes with 128/128 single-candidate completions each. Candidate-0 pass@1 was 6/128 for Base and 8/128 for warm-start; format was 17/128 versus 23/128. The paired delta is +1.5625 pp with bootstrap 95% CI [-2.34375, +6.25] pp and exact McNemar p=0.7265625. Interpret this as improved protocol adherence and a small uncertain dev gain, not hidden-test proof. Never rerun either dev evaluation or the warm-start to improve these outcomes.

### GRPO-v2 Stage Q model-bound runtime gate

The only frozen GRPO-v2 training entrypoint is `math_rlvr.training.grpo_v2` with exact config `configs/grpo_v2/grpo_v2_seed42.json` SHA `059553888fdc997a5b9f214fde526d4be8c309ca84abe212c243fd74305b1b66`, exact warm-start checkpoint artifact `507749d3...92f0`, policy adapter `44066dd1...72b9`, and dual confirmation. It loads only the warm-start policy adapter and initializes a fresh GRPO optimizer/scheduler; SFT optimizer state is forbidden. Training is 512 unique curriculum prompts once, 128 updates/512 microsteps/2,048 completions, with a 524,288-token cap and checkpoint/dev cadence 32/64/96/128. Same-run resume is limited to 32/64/96. Hidden test cannot enter training or dev-only checkpoint selection. Stage Q was CPU-only and does not authorize execution.
