# Repository Guidelines

## Project Mission

This repository is the artifact-first Math RLVR project. Its goal is a reproducible PPO-versus-GRPO comparison for few-shot mathematical reasoning, with matched prompts, reward contracts, completion/token budgets, and auditable run artifacts. Formal experiments target `Qwen/Qwen2.5-1.5B-Instruct`; `Qwen/Qwen2.5-0.5B-Instruct` is the bounded smoke-test model. Countdown is verifier/smoke data, while GSM8K and MATH are training/evaluation data and MATH500 is held out from training.

Never advance to a new paid/GPU stage implicitly. Model download, CUDA initialization, model loading, generation, and PPO/GRPO updates each require the user's explicit scope. A successful stage does not authorize the next stage.

## Current Baseline and Milestones

The active branch is `pivot/math-rlvr`. Important milestones are:

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
