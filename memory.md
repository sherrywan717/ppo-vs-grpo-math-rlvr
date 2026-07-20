# Math RLVR Project Memory

This file records operational history and pitfalls that should survive context changes. It is not an authorization to rerun anything. Always follow the user's newest explicit scope and `AGENTS.md`.

## Completed Results

### Fixed Qwen 0.5B download and tokenizer audit

- Model: `Qwen/Qwen2.5-0.5B-Instruct`.
- Fixed revision: `7ae557604adf67be50417f59c2c2f167def9a775`.
- Cache: `/root/autodl-tmp/cache/huggingface` only.
- Three orphaned Python download processes were safely terminated while retaining the partial cache.
- The controlled resume completed on attempt 1 with `max_workers=1` in 284 seconds.
- Final cache size at that stage: 1,023,263,749 bytes; `model.safetensors` is 988,097,824 bytes.
- `snapshot_download(..., local_files_only=True)` passed for the fixed revision.
- CPU-only tokenizer audit passed Qwen chat boundaries, reasoning/answer envelope, EOS/PAD, left padding, variable-length batching, prompt/completion token boundaries, 512-token truncation, parser behavior, and two representative samples each for Countdown, GSM8K, and MATH.
- Reports: `reports/model_audit/qwen2.5-0.5b/`.

### Stage D static infrastructure

- Commit: `5a10cbae2abcb066b423b10ff9d327ad1483b75c`.
- CPU gates passed: compileall, Ruff, 40 tests, environment check, manifest validation, PPO dry-run, and GRPO dry-run.
- Dry-runs loaded no model, called no training method, and did not initialize CUDA.
- PPO and GRPO now expose the same prompt renderer identity.

### CUDA/model-load sanity A

- Run ID: `cuda_load_sanity_qwen25_05b_20260713T042511Z`.
- Commit: `6daca223bd17ddc9201e0b8dc7cdc3c677db9b39`.
- Full artifacts: `/root/autodl-tmp/runs/math_rlvr/cuda_load_sanity_qwen25_05b_20260713T042511Z/`.
- Git-safe artifacts: `reports/runs/cuda_load_sanity_qwen25_05b_20260713T042511Z/`.
- Backup: `/root/autodl-fs/math-rlvr-backups/cuda_load_sanity_qwen25_05b_20260713T042511Z.tar.gz`.
- Backup SHA256: `c18adab9da10b05b5befd8abab0d27152336e576ec77bcb0d9cdac6fa46a9ff3`.
- GPU: NVIDIA H800 PCIe; fixed local snapshot; BF16 on `cuda:0`; no meta parameters.
- Actual parameters: 494,032,768. Finite forward checks passed for two prompts with token lengths 65 and 61.
- Plan A explicitly specified no LoRA injection. The base was made read-only, so LoRA/trainable parameters and ratio were 0. Static matching found 24 each of q/k/v/o targets.
- Training updates, optimizer steps, generated completions, generated tokens, and checkpoints were all 0.
- PyTorch peak allocated/reserved: 1,001.73/1,046 MiB. The one-second `nvidia-smi` sampler observed 459 MiB.
- Measured load/check wall time: 1.896 seconds; 0.0005267 GPU-hours; cost ¥0.00468 at ¥8.88/GPU-hour.
- GPU returned to 0 MiB afterward and no compute process remained.
- One non-fatal warning occurred: `torch_dtype` is deprecated; use `dtype` instead.

## Pitfalls and Lessons

1. Hugging Face retry ownership: orphaned `python -` processes can retain blob locks and sockets. Inspect `/proc/<pid>/fd` before terminating only the confirmed download processes. Never delete `.incomplete` blobs or lock/cache trees to “fix” a resume.
2. Xet partial size behavior: the apparent `.incomplete` size can shrink or oscillate during resume because the downloader may recreate or sparsely populate it. Do not infer data loss from one `stat`; rely on the downloader's resume declaration and final hash/snapshot completion.
3. Offline enforcement: set `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, use the exact snapshot path, and pass `local_files_only=True`. A cache hit by model name alone is weaker than a fixed snapshot path and revision check.
4. TRL CPU tests: constructing `GRPOConfig`/`PPOConfig` with BF16 while CUDA is hidden can still probe GPU capability and fail. Injected fake-trainer tests must set `use_cpu=True` and `bf16=False`; production GPU builders retain BF16.
5. Shared renderer exports: importing a renderer only to expose it triggers Ruff F401 unless it is deliberately exported (for example through `__all__`) or otherwise referenced.
6. Generated report whitespace: Python CSV writers may produce CRLF, and Matplotlib SVG paths contain trailing spaces. Normalize Git-safe CSV/SVG before `git diff --check`, then regenerate `artifact_manifest.json` hashes.
7. Warning capture: `warnings.catch_warnings` does not necessarily catch Transformers logger messages written to stderr. Audit `stderr.log` and reconcile `warning_count`, `warnings.txt`, summaries, checksums, and the Git-safe manifest without rerunning the model.
8. Deprecated loader argument: Transformers 4.57.6 warns on `torch_dtype`; use `dtype=torch.bfloat16` for the next loader implementation after review.
9. Resource sampling: a one-second `nvidia-smi` interval can miss a short allocation peak. Treat PyTorch `max_memory_allocated`/`max_memory_reserved` as the authoritative process peak and increase sampler frequency for short runs.
10. Wall-time semantics: distinguish subprocess end-to-end duration, Python import/setup time, and the measured model-load/check interval. State which interval is used for GPU-hours and cost.
11. ArtifactManager creates an empty `checkpoints/` directory even for no-checkpoint runs. Exclude it from a backup whose contract says no checkpoints; archive validation should reject checkpoint paths, not merely checkpoint files.
12. Plan scope wins: plan A said “LoRA: none” even though generic authorization language mentioned an adapter. The run correctly did no injection and recorded zero trainable parameters. Do not silently borrow plan B's LoRA configuration.
13. Patch/sandbox limitation on this host: unprivileged namespaces are disabled, so the sandbox and dedicated patch helper may fail with a `bwrap` namespace error. Do not treat that as a repository failure. Use approved, narrowly scoped workspace operations and still perform full diff/secret audits.
14. Shell gate chaining: without `set -e` or explicit return-code checks, a failed audit such as `git diff --check` may not stop later commands. Use `set -e` for final audit/commit sequences; if a commit slips through, amend only the report metadata rather than rerunning the paid stage.

## Before Any GRPO Single-Update Smoke

- Obtain separate explicit authorization; CUDA load sanity does not authorize GRPO.
- Replace deprecated `torch_dtype` with reviewed `dtype` usage.
- Inject and verify plan B policy LoRA (`r=16`, alpha 32, dropout 0, q/k/v/o); A only checked static target presence.
- Increase `nvidia-smi` sampling frequency and retain PyTorch allocator peaks.
- Reconfirm clean worktree, exact local revision, no other GPU process, hard wall/token/completion caps, and failure-without-retry behavior.
- Do not start PPO, save a full base model, or proceed to another stage automatically.

## Guarded GRPO Runner Implementation

- Default CLI remains dry-run; `--execute` alone is rejected. The frozen smoke config additionally requires `--confirm-single-update`, clean Git, fixed local snapshot, and exact resolved budgets before real imports.
- `training/trl_compat.py` is the only TRL 0.24.0 private hook. It validates `completion_ids`/binary `completion_mask` shapes for exact token counts and records microsteps; official callbacks guard optimizer/global steps.
- BudgetGuard refuses a ninth completion, token 1,025, microstep 5, optimizer step 2, global step 2, non-finite rewards, INFRA_ERROR, and the 15-minute deadline before success.
- Checkpoint inventory permits adapter weights/config and trainer state but rejects full-size or non-adapter weight files. Success requires artifacts plus verified backup; failures stop the monitor and remain failure.
- CPU fake tests do not authorize or execute the real runner, generation, CUDA, or PPO.

## GRPO First Real Smoke Failure and CPU Repair

- First real attempt: `grpo_single_update_qwen25_05b_20260713T050407Z`.
- Failure-report commit: `ebc926c432d6778c3b057a0b7b518f7f2eaea5ed`; the run, report, and backup remain preserved.
- The run stopped before trainer construction completed: 2 selected prompts, 0 completions/tokens/microsteps/optimizer steps/global steps, 0 MiB sampled VRAM, no checkpoint.
- Primary pitfall: after correct local snapshot resolution, runtime mutated `model.name_or_path` to the snapshot path and the generic validator compared that path with the repo-ID allowlist, raising `unexpected model checkpoint`.
- Secondary pitfall: `dataclasses.asdict(BudgetGuard)` included the injected `clock` callable, causing JSON artifact finalization to fail.
- Repair: keep the frozen config identity unchanged and pass a frozen `ValidatedModelSource` separately to the loader/builder. The validator accepts only the exact canonical 0.5B snapshot returned by local-only Hugging Face resolution, with required files and Qwen2 causal-LM config identity.
- Repair: all guarded counters use `BudgetGuard.snapshot()`, with recursive primitive-only validation and a minimal primitive fallback failure record.
- HF cache detail: the project cache root is `/root/autodl-tmp/cache/huggingface`, but explicit `snapshot_download(cache_dir=...)` must receive its `hub/` child. Passing the parent makes local-only lookup miss the existing snapshot.
- CPU/offline path preflight read metadata only and left CUDA uninitialized. This repair does not authorize or perform a real rerun.

### CPU repair verification

- Full CPU gates after the first-failure repair: compileall passed, Ruff passed, 81 tests passed, environment check passed, manifest validation passed, GRPO/PPO dry-runs passed, and fake guarded execute passed.
- The real local snapshot offline path preflight passed with Qwen2ForCausalLM identity; CUDA was false before and after.
- Real model loads, completions, train calls, and optimizer steps were all 0. The frozen GRPO and PPO smoke YAML hashes matched the baseline exactly.

## GRPO Second Real Smoke Failure and Evidence Repair

- Second real attempt: `grpo_single_update_qwen25_05b_20260713T053852Z`; failure-report commit `9db5bf945e123e1f31670939bccda7b6e31aae8a`.
- The snapshot builder and BudgetGuard finalization passed. The run produced 2 prompts, 8 completions, 687 generated tokens, 4 microsteps, and exactly 1 optimizer/global step.
- All eight rewards were format errors with reward zero; reward variance was zero and zero-advantage fraction was 1.0. This is integration evidence, not evidence of learning.
- Failure occurred after training when the old inventory rejected the 7,441-byte `training_args.bin`. Trainer auto-save and the runner's manual `save_model` also produced duplicate adapter trees.
- Repair: use only Trainer's top-level `checkpoint-1`; remove the post-train manual save. Permit only exact, canonical, regular, non-symlink `training_args.bin` at checkpoint root, capped at 1 MiB, hashed without deserialization.
- Repair: the sole TRL 0.24.0 shim now joins exact IDs/masks, mask-derived token counts, decoded Unicode text, exact verifier input, and ordered reward results. Missing or reordered evidence blocks success.
- Frozen GRPO beta resolves to 0.0. KL must be `null` with `kl_available=false` and an explicit reason, never fabricated as zero.
- PyTorch allocator peaks are reset/recorded only inside the authorized real path through an injectable backend; CPU dry-runs keep CUDA uninitialized.
- The two historical failure summaries, metrics, inventories, full runs, commits, and backups remain failures and are not rewritten.

### CPU checkpoint/evidence repair verification

- Full CPU/offline gates passed with 107 tests. Compileall, Ruff, environment check, manifest validation, GRPO/PPO dry-runs, fake guarded execute, and fake checkpoint inventory/finalization all passed.
- CUDA remained uninitialized; real model loads, completions, train calls, and optimizer steps were all 0.
- Frozen GRPO/PPO YAML hashes remained unchanged, and protected summary/metrics/checkpoint-inventory hashes for both historical failure runs remained unchanged.

## GRPO Third Real Smoke Failure and Allocator Repair

- Third real attempt: `grpo_single_update_qwen25_05b_20260713T061248Z`; failure-report commit `438569d97a8636ea6ad13394920663016e01282e`. Preserve its full run, Git-safe report, failure status, and backup unchanged.
- The run stopped before model loading, completion generation, or updates with `RuntimeError: Invalid device argument`. Counters were 2 configured prompts and zero completions/tokens/microsteps/optimizer/global steps; no checkpoint was created.
- Exact input at the failing allocator call was the built-in string `"cuda:0"`, originating from `CudaAllocatorEvidence`'s constructor default. It was not callable, a function, a `torch.device`, an integer, or a GPU display name. The resolved GRPO config contained no device value.
- PyTorch 2.8 allocator APIs document `torch.device` or integer inputs. The string was passed directly to `reset_peak_memory_stats`, producing the failure.
- Repair: `normalize_cuda_device_index` accepts only reviewed CUDA forms and returns a range-checked, non-boolean integer. Implicit `cuda`/`None` resolution calls `current_device()` only inside the authorized available-CUDA path. All reset/current/peak allocator calls reuse one index; display label/name are never API inputs.
- Lifecycle states are `not_started`, `active`, `finalized`, `unavailable`, or `failed`. Normalize/reset/collection failures retain primitive-only phase/type/message and are re-raised.
- CPU verification passed 131 tests plus the explicit fake guarded execute and allocator lifecycle gates. CUDA remained uninitialized and frozen GRPO/PPO YAML hashes were unchanged. This CPU repair alone does not authorize a GRPO rerun.

### Minimal CUDA allocator probe

- Run ID: `cuda_allocator_probe_20260713T063028Z`; executed exactly once after the CPU fix commit with a clean worktree.
- Normalized index was integer 0 on NVIDIA H800 PCIe. `reset_peak_memory_stats(0)` succeeded.
- A 1,048,576-byte uint8 tensor yielded current/peak allocated 1 MiB and current/peak reserved 2 MiB. After deletion and `empty_cache`, current allocated and reserved were both 0 MiB.
- Probe wall time was 0.425709065 seconds, GPU-hours 0.0001182525, and cost CNY 0.00105008 at CNY 8.88/GPU-hour.
- No model, tokenizer, dataset, completion, trainer, optimizer, checkpoint, or network access was involved. No compute process remained after exit. This success validates allocator evidence only and does not authorize GRPO/PPO.

## Successful GRPO Smoke Prompt Forensics

- Successful evidence run: `grpo_single_update_qwen25_05b_20260713T063829Z`, commit `85776a8290f736b0469f377b0a3d3c4b86cdc7a1`. Execution counters were 2 prompts, 8 completions, 687 tokens, 4 microsteps, and one optimizer/global step.
- Offline replay with the unchanged parser/verifier reproduced all 8 runtime `format_error` statuses. Reasoning tags appeared in 0/8, complete answer pairs in 3/8, and complete envelopes in 0/8. Four outputs hit the 128-token cap. There is no parser misclassification evidence.
- Real v0 prompts are 85/83 tokens; the saved prompt hashes are normalized dataset user-text hashes and match exactly. The Qwen template is applied once, roles are system/user followed by an open assistant turn, and no prompt truncation occurs.
- v0 contains both literal tag pairs, but the system-message format instruction ends 43/41 tokens before the generation boundary and lacks explicit no-outside-text, closure, expression-only, no-equals/target, and concise-reasoning constraints.
- At the forensic-audit stage, the then-unactivated `prompt_v1_strict_concise` candidate was 157/155 tokens and placed the final format instruction 5 tokens before the boundary. It retains the problem but includes no gold construction. PPO and GRPO export the same renderer; the later smoke-only activation is recorded below.
- BOS is absent; EOS is `<|im_end|>` 151645; PAD is `<|endoftext|>` 151643; padding is left and truncation is right. Runtime alignment warnings did not alter rendered text or the assistant boundary.
- Any v0/v1 generation diagnostic requires independent `--generate-only --confirm-prompt-diagnostic` authorization, must not train, and is capped at 16 completions/2,048 generated tokens. This forensic audit does not authorize it or PPO.

## Guarded Prompt A/B Runner Implementation

- CPU-only implementation entry: `python -m math_rlvr.evaluation.prompt_ab --config configs/diagnostics/prompt_ab.yaml`. At implementation time no real A/B generation had been executed; the later successful diagnostic is recorded below.
- Dry-run needs no execution flag. Real generation requires both `--generate-only` and `--confirm-prompt-diagnostic`; `--confirm-single-update` is rejected.
- Static gates precede delayed runtime imports: clean `pivot/math-rlvr`, both offline variables equal to 1, exact canonical local 0.5B snapshot, and exact diagnostic config/budget.
- Matched seeds 42–49 map by problem and generation index and are reset separately for Python, PyTorch CPU, and PyTorch CUDA in each condition. v0 RNG consumption cannot alter v1.
- Completion IDs are sliced after padded input width; attention-mask sum records unpadded prompt length. EOS is retained and post-EOS padding is excluded. Decode/re-tokenize is not used for the token budget.
- Hard limits: v0/v1, two fixed train prompts each, four completions per prompt, 128 tokens each, 16 total completions, 2,048 total tokens, 120 seconds, and 3.5 GiB nvidia-smi stop gate.
- Real code is base BF16 eval/inference only with all parameters frozen. Zero Trainer/train/backward/optimizer/training-step/checkpoint/model-write counters are required.
- At runner implementation v1 remained unactivated. Candidate review required better envelope rate, at least one complete envelope, no increased truncation, and at least one nonzero within-problem reward-variance group. All-WRONG_ANSWER can still mean no advantage signal.
- `apply_patch` remains unavailable because this host disables unprivileged namespaces; standard narrowly scoped patches plus complete diff/tests are the approved local recovery pattern.

### Prompt A/B cleanup false positive

- Historical run `prompt_ab_qwen25_05b_20260713T101918Z` generated 16 completions/813 tokens but remains immutable failure.
- `pytorch_allocator.json` is `{}` because worker close raised before returning allocator evidence; exact nonzero bytes are unrecoverable and must not be invented.
- Parent evidence: PID 109901 exited, no worker compute process, GPU memory 0 MiB before/after, parent CUDA uninitialized; manual nvidia-smi was also 0 MiB/no process.
- In a spawned worker, current allocator memory before process exit is diagnostic. Persist it and warn; parent post-exit PID/process/memory-baseline verification is the authoritative release gate.
- Offline analysis of the immutable failed A/B run found v0 0% complete envelope and two zero-advantage groups; v1 25% complete envelope, two INVALID_EXPRESSION rewards, and two nonzero-variance groups. That failed run alone did not activate v1.

## Prompt v1 Smoke-Only Activation

- Successful matched diagnostic: `prompt_ab_qwen25_05b_20260713T105428Z`, report commit
  `fae8394c52771e16c774e4cf409d27394c166afe`.
- v0 produced 8/8 FORMAT_ERROR and zero nonzero-variance problem groups. v1 produced
  six FORMAT_ERROR and two INVALID_EXPRESSION outputs, a 25% complete-envelope rate,
  and nonzero reward variance in both problem groups.
- v1 valid-expression rate, number-usage accuracy, pass@1, and pass@4 were all zero.
  The evidence supports integration-smoke use only, not a final/production prompt or a
  claim of learning quality.
- `prompt_v1_strict_concise` is frozen byte-for-byte as `approved_for_smoke` and
  `not_approved` for production. Qwen 0.5B PPO and GRPO smoke configs share its
  version/hash/renderer identity; main/formal 1.5B configs remain unchanged.
- `prompt_v0_grpo_smoke`, its hash, and all historical A/B/GRPO artifacts remain
  immutable for replay.
- The next GPU stage is one newly and separately authorized GRPO single-update smoke.
  PPO remains unauthorized; never enter PPO automatically.

## Prompt v1 GRPO Single-Update Smoke

- Run `grpo_single_update_qwen25_05b_20260713T112100Z` executed exactly once with the
  frozen local Qwen 0.5B revision and v1 prompt identity. It completed 2 prompts, 8
  completions, 276 generated tokens, 4 microsteps, and 1 optimizer/global step.
- Infrastructure result: pass. Completion IDs/text/counts are complete, exposed
  metrics are finite, only the authoritative `checkpoint-1` exists, the adapter-only
  inventory passed, artifacts and persistent backup were verified, and post-process
  `nvidia-smi` returned to 0 MiB with no compute process.
- Learning-signal result: fail. Both problem groups had rewards `[0, 0, 0, 0]`, zero
  within-group variance, and zero advantage; all eight statuses were `FORMAT_ERROR`.
  Loss and grad norm were 0, entropy was 0.5070152432, and frozen beta=0 means KL is
  correctly unavailable/null rather than zero.
- PyTorch pre-exit allocator residue was 64 MiB allocated/108 MiB reserved; this is a
  warning because the process-exit GPU release gate passed. Runtime nvidia-smi peak is
  unavailable because the runner CSV is empty and must not be invented.
- The verified full-run backup is
  `/root/autodl-fs/math-rlvr-backups/grpo_single_update_qwen25_05b_20260713T112100Z.tar.gz`,
  SHA256 `583ae58d892fe7f743531cddbd1eaf6685d809c9607f6532f2471f875a44180c`.
- Do not interpret infrastructure success as learning. v1 remains smoke-only; PPO is
  still unauthorized and blocked until the user reviews both conclusions.

## Staged Shaped Reward v2 CPU Intervention

- Root cause of the v1 smoke's zero update was the old scalar mapping collapsing every
  partial-protocol `FORMAT_ERROR` to 0. The strict parser, verifier, Trainer, LoRA,
  and GPU path were not changed.
- `shaped_v2_staged` adds deterministic components: answer block 0.05, strict
  protocol 0.05, safe expression 0.05, exact number use 0.05, and correctness 0.80.
  Only the unchanged original canonical `VERIFIED_PASS` reaches 1.0.
- Partial analysis synthesizes a strict envelope around the uniquely extracted answer
  and calls the same canonical Countdown verifier. The probe can establish expression
  and number-use validity but can never grant the correctness component.
- Sparse reward remains 1 only for `VERIFIED_PASS`; canonical status and formal
  metrics remain unchanged. Resource limits get zero partial reward and infrastructure
  errors abort.
- Immutable run `grpo_single_update_qwen25_05b_20260713T112100Z` remains under the
  old policy. CPU replay produced group rewards `[0.10, 0.10, 0.15, 0.00]` with
  variance 0.00296875 and `[0.10, 0.05, 0.10, 0.05]` with variance 0.000625.
  Both groups now have potential advantage while all canonical statuses remain
  `FORMAT_ERROR`.
- This is a publicly recorded post-smoke intervention. Future fair PPO/GRPO smoke
  comparisons must both use the frozen new version/hash. It does not authorize GPU
  GRPO, and PPO remains unauthorized.


## Staged-Reward GRPO Single-Update Smoke

- Run: `grpo_single_update_qwen25_05b_20260713T122258Z`; invoked once with the fixed offline Qwen 0.5B snapshot, `prompt_v1_strict_concise`, and `shaped_v2_staged`. No retry or PPO occurred.
- Infrastructure passed: 8 completions, 276 exact generated tokens, 4 microsteps, and 1 optimizer/global step. The unique adapter-only `checkpoint-1`, counters, artifacts, and post-exit GPU release all passed.
- Reward integration passed. Every completion records the unchanged canonical status, scalar reward, policy version/hash, five shaped components, and verifier detail from the online text.
- Learning signal passed. Group rewards were `[0.10, 0.10, 0.15, 0.00]` (population variance `0.00296875`) and `[0.10, 0.05, 0.10, 0.05]` (variance `0.000625`); both groups were nonzero-variance. Loss `0.6453`, grad norm `4.648100852966309`, entropy `0.5070152431726456`, and learning rate `1e-5` were finite. Frozen beta 0 means KL is correctly unavailable/null.
- All eight canonical statuses remained `FORMAT_ERROR`; strict parser/verifier and formal metrics did not change. This confirms integration and a one-update gradient path, not task learning or algorithm superiority.
- PyTorch worker/pre-exit current memory was 64 MiB allocated / 108 MiB reserved; post-process `nvidia-smi` was 0 MiB with no compute process, so record only `worker_allocator_nonzero_before_process_exit`.
- PPO is still unauthorized. Stop after evidence, backup, and Git-safe reporting.

## Guarded PPO Single-Update Runner CPU Gate

- The recovered `guarded_ppo.py` was syntactically complete but stopped after its
  injected core lifecycle; it had no CLI, delayed real runtime, or tests. The correct
  existing edits were retained and only missing/incorrect wiring was repaired.
- TRL 0.24.0 source inspection confirmed `total_episodes=4` and single-device batch 4
  yield one rollout update with one response for each of four dataset rows.
  `generation.num_generations=4` is not a PPOConfig field and is explicitly recorded
  as ignored; total completions remain 4, never 16.
- TRL PPO internally hard-codes top-p 1.0. The sole compatibility shim now validates
  response length/temperature and applies the frozen YAML top-p 0.95 before generation.
- Policy and value are distinct local-only loads of the same validated Qwen 0.5B
  snapshot. Reference evaluation disables the policy adapter; reward is
  parameter-free. The optimizer is accepted only when its parameter objects exactly
  equal policy LoRA plus value LoRA/scalar-head trainables.
- The authoritative `checkpoint-1` contains separate policy adapter, value adapter,
  and scalar-head safetensors plus JSON trainer/resume metadata. Base-model and
  optimizer weights, unexpected files/directories, symlinks, duplicate adapters, and
  full-model-sized files fail closed.
- CPU gates passed: compileall, Ruff, the full test suite, environment and manifest
  checks, GRPO/PPO dry-runs, and an explicit fake guarded PPO execute. Environment
  evidence reported CUDA uninitialized and no model/tokenizer loaded. No real
  completion, CUDA operation, or PPO/GRPO optimizer update occurred.
- A future real PPO smoke requires both flags, clean Git, both offline variables, the
  exact local snapshot, and a new explicit authorization. This implementation gate is
  not that authorization.


## Accepted PPO Single-Update Smoke and Telemetry Closeout

- The user accepted immutable run `ppo_single_update_qwen25_05b_20260714T051538Z` as
  `execution_success/nonessential_telemetry_warning`; it must never be rerun or have
  its original failure report, launcher output, completions, checkpoint inventory, or
  other historical evidence rewritten.
- The training path completed four unique prompts, four responses/completions, 141
  generated tokens, and exactly one PPO epoch, minibatch, update, optimizer step, and
  global step. Rewards `[0.05, 0.10, 0.10, 0.10]` had population variance
  `0.00046875`. The role-separated policy/value adapter and scalar-head checkpoint
  contained no full base or optimizer weights.
- Launcher exit code 1 was post-training artifact finalization only: TRL 0.24.0 emitted
  `val/ratio_var=NaN`, and the literal trainer history reached strict JSON safety.
  The run remains accepted rather than retroactively rewritten.
- The CPU-only repair allowlists only `val/ratio_var` as nullable nonessential
  telemetry. It persists `value=null`, `available=false`, the raw key,
  `classification=non_finite`, the NaN/Inf subtype, and an explicit reason. It never
  fabricates zero and does not mutate original in-memory history. Required metrics,
  rewards, losses, counters, budgets, and unreviewed metric keys still fail closed on
  every NaN/Inf.
- Regression tests cover successful finalization with this warning, required/unreviewed
  non-finite failures, strict finite JSON, and unchanged SHA256 hashes for the four
  protected historical PPO evidence files. No CUDA, model load, generation, or real
  PPO/GRPO execution occurred during the repair.

## Stage D Smoke Readiness Decision

- The current matching GRPO technical smoke already exists:
  `grpo_single_update_qwen25_05b_20260713T122258Z`. It used the fixed Qwen 0.5B
  revision, `prompt_v1_strict_concise`, `shaped_v2_staged`, the shared canonical
  parser/verifier semantics, completed a real optimizer/global step with credible
  artifacts, and had nonzero reward variance in both groups.
- Stage D technical smoke is complete. The old v0 and zero-reward v1 GRPO smokes are
  not comparators for the accepted PPO run.
- The accepted PPO and GRPO runs do not support an algorithm-effect claim: PPO used
  four prompts × one response and a 4/512 completion/token budget; GRPO used two
  prompts × four responses and an 8/1,024 budget. Their actual totals and variance
  aggregation units also differ. A single update validates execution, not task
  learning or PPO superiority.
- The historical reports share the exact reward version/SHA and canonical status
  semantics but do not serialize a standalone parser/verifier version or SHA. The
  next matched 0.5B pilot should add that evidence, freeze one prompt allocation,
  match actual completion/token budgets, and predefine at least three seeds. The plan
  in `reports/stage_d/smoke_readiness_matrix.{json,md}` is not GPU authorization.


## Matched 0.5B Pilot CPU Freeze

- Baseline was clean `pivot/math-rlvr` at
  `973a3ce1ad3439f19bc99f9764cce8dd47ac8bb3`. No historical run/report/checkpoint was
  modified and no CUDA, model/tokenizer load, generation, Trainer, backward, optimizer
  step, download, Stage D rerun, or 1.5B action occurred.
- Stage D stays complete but algorithm-effect comparison is not established. Immutable
  counterparts are PPO `ppo_single_update_qwen25_05b_20260714T051538Z` with nullable
  ratio-variance warning and GRPO `grpo_single_update_qwen25_05b_20260713T122258Z`.
- Pilot manifest SHA256 is
  `0235210e038bc27ebf2e7218691f36f09c8e11f0bbc743f46a5318a279f6bc1f`; it selects
  frozen Countdown train IDs 0–3 in source order. Problem contracts exclude
  gold/construction, and rendered-prompt hashes cover canonical tokenizer-free chat
  payloads plus the open assistant turn.
- Parser contract is `strict_completion_parser_v1`, SHA256
  `655c30f20c677ead5728b402a1b6d5a4d4cefe54e4c1b34abebdafe41f3ba0ad`; Countdown
  verifier is `countdown_ast_fraction_v1`, SHA256
  `593fa4f1f12702411248b77d8059b4df84a182334a8f9923a2283d04a3fb0c74`. Hashes use
  canonical semantic JSON and do not depend on Markdown/comments.
- Six resolved configs cover PPO/GRPO × seeds 42, 123, 2026 and are accepted only by
  exact repository path plus raw-file SHA256. Per run: four prompts × four responses,
  16 completions, 2,048 tokens, one update/optimizer/global step, zero retries, and one
  isolated checkpoint/full backup.
- Real TRL 0.24.0 CPU config derivation confirms PPO local rollout batch 16 from batch4
  × GA4, one epoch/minibatch, forward batch4, and no `num_generations`; GRPO confirms
  generation batch16, four generations, batch4, GA4, and `steps_per_generation=4`.
- Fixed order alternates: 42 PPO/GRPO, 123 GRPO/PPO, 2026 PPO/GRPO. Expected suite
  cost is 0.04917 GPU-hours / ¥0.4366; 2× planning ceiling is 0.09833 GPU-hours /
  ¥0.8732. Expected/hard peak VRAM is 7/14 GiB for PPO and 3.5/7 GiB for GRPO.
- The original pilot freeze correctly left GPU execution disabled because TRL PPO
  shuffled its DataLoader and guarded evidence was fixed at historical 4/8 shapes.
  The later CPU execution-contract repair below resolves both items; it does not
  retroactively authorize any GPU run.

## Matched Pilot Execution-Contract Repair

- Baseline was clean `pivot/math-rlvr` at
  `28b8d586766811f71de8c1a2b1f8779bd68bcbdf`. Frozen pilot manifest, six resolved
  configs, Stage D configs and protected historical artifacts remained unchanged.
- TRL 0.24.0 source confirms PPO constructs `DataLoader(shuffle=True)`, prepares it,
  then `train()` consumes `self.dataloader`. For matched PPO only, the guarded subclass
  now replaces only that loader after `PPOTrainer.__init__`: explicit
  `SequentialSampler`, batch 16, `drop_last=True`, `num_workers=0`, world size 1, and
  the existing Accelerator `prepare_data_loader`. It never re-prepares model or
  optimizer.
- Every PPO dataset row carries position, problem/generation identity, problem and
  rendered-prompt hashes, seed and algorithm. The Accelerator-prepared first batch and
  every consumed iterator validate the exact prompt-major 16-key order. PPO reward
  evidence joins by this audited order and tokenized prompt identity, never by guessing
  from completion text.
- Immutable exact-path/SHA `ExpectedRunContract` profiles protect Stage D PPO=4/512,
  Stage D GRPO=8/1,024, pilot PPO=16/2,048 and pilot GRPO=16/2,048. All require one
  update/optimizer/global step. Profiles also bind local-only Qwen revision, BF16,
  policy LoRA, sampling, prompt/reward and parser/verifier identities. Arbitrary CLI
  widening and main/1.5B configs have no profile.
- Online overflow fails before update; finalization requires exact completion count and
  mask-derived token totals. Guard-derived counters are persisted into metrics,
  summary and run manifest. GRPO maps its real prompt-major four-generation output to
  the same 16 comparison keys exactly once without changing batching semantics.
- CPU gates passed: 120 targeted tests, 321 full tests, Ruff, compileall, environment
  check, manifest validation, six pilot dry-runs, two Stage D dry-runs, and explicit
  fake PPO/GRPO 16-completion execute/finalization. CUDA stayed uninitialized; model
  and tokenizer loads, real generation/Trainer/backward/optimizer calls were zero.
- The two known correctness blockers are closed. The six GPU jobs still require a new
  explicit user authorization, clean/offline/GPU preflight and the frozen run order;
  never start them, retry automatically, or enter 1.5B from this CPU result.

## Matched Pilot First GPU Attempt and Suite Stop

- The user explicitly authorized the frozen six-run matched suite from clean
  `pivot/math-rlvr` at `10fcc2173e45b5eab438b0712b9aa9562abdf214`. Initial Git,
  config/manifest identity, offline/local snapshot, storage and idle H800 preflights
  all passed; the preflight process reported CUDA uninitialized and zero model loads.
- Only Run 1 was executed, exactly once:
  `ppo_matched_0p5b_seed42_20260714T073357Z`. The frozen config hash was
  `1daeba7e6cd5e0af43c7f7cb9db87b46d44608adf9fdf432dc7b2c34ea059fdd`.
- The delayed execution path initialized the PPO value scalar head, then dataset
  rendering called `render_training_prompt`. `prompt_version_from_config` recognizes
  only `smoke-*` experiment names as allowed to select `prompt_v1_strict_concise`, so
  the valid `pilot-*` experiment fell into the main/formal rejection branch and raised
  `ValueError: main/formal configs must not activate a smoke prompt`.
- This was a failure before generation/training: 0 completions, 0 generated tokens,
  0 update/optimizer/global steps, no reward or loss evidence, and no checkpoint. The
  expected 16 prompt-major comparison keys and every frozen identity were recorded,
  but no actual completions exist to validate or aggregate.
- The resource window was 4.899377426 seconds, `nvidia-smi` peak was 4 MiB, GPU-hours
  were 0.001360938173925711 and estimated cost was CNY 0.012085130984460315. The
  worker exited; afterward GPU memory was 0 MiB with no compute process.
- Full artifacts and Git-safe reports were saved. The verified failure backup is
  `/root/autodl-fs/math-rlvr-backups/ppo_matched_0p5b_seed42_20260714T073357Z.failure.tar.gz`,
  SHA256 `21a64fb02f8522901eea92f4f027ba143b8b04f2a8c08292b75d0b6e9ec8f7a2`.
- The blocker requires a separate CPU-only pilot-aware prompt routing repair. No retry
  was attempted and Runs 2–6 were not executed. This suite authorization is exhausted;
  a repaired GPU run requires new explicit authorization. Do not enter 1.5B.

## Pilot-aware Prompt Routing CPU Repair

- Baseline was clean `pivot/math-rlvr` at
  `fbce9506f0b824bb26cc70a89d5c7d7a33b2b057`. The prior PPO seed-42 failure,
  Git-safe evidence and backup were fingerprinted before work and remained unchanged.
- Root cause affected both delayed runtimes: CPU pilot resolution understood the pilot
  family, but `prompt_version_from_config` later inferred eligibility solely from
  `experiment.name.startswith("smoke-")`. PPO called it after policy/value loading;
  GRPO had the same latent path after policy loading.
- `ValidatedExperimentScope` now binds scope to exact repository path/raw SHA.
  Stage D and matched pilot are derived from `ExpectedRunContract`; exact main configs
  resolve to `MAIN_FORMAL` with no execution profile. Names, arbitrary strings,
  unknown paths, hash drift and serialized scope drift cannot grant v1 access.
- Resolver, dry-run, ExpectedRunContract, pre-model prompt preflight, delayed dataset
  builder and prompt selector all verify one scope. Before model handling, PPO renders
  16 protected Python message rows and GRPO renders four, checking frozen rendered
  hashes and the same 16 comparison keys. Future runs persist
  `prompt_scope_preflight.json`.
- CPU regressions covered PPO/GRPO seeds 42/123/2026, the exact seed-42 failed path,
  main `pilot-*` name spoofing, unknown path/hash, Stage D, historical v0,
  prompt-major/evidence profiles and immutable failure hashes. Results: 168 targeted
  and 337 full tests passed, plus Ruff, compileall, check_env, manifest validation,
  six pilot and two Stage D dry-runs, and fake PPO/GRPO execute/finalization.
- CUDA remained uninitialized; real model/tokenizer load, generation, Trainer,
  backward and optimizer calls were zero. Frozen pilot/config/prompt/reward/parser/
  verifier identities did not change. No GPU command, failed-run retry or 1.5B action
  occurred.
- Routing is technically repaired, but GPU execution remains unauthorized. Any new
  matched suite must receive explicit authorization and start from a new PPO seed-42
  run; the immutable failed attempt stays excluded from scientific aggregation.


## PPO Pilot Collator Mapping CPU Repair

- New immutable failure: `ppo_matched_0p5b_seed42_20260714T082003Z`, executed once
  from `3df98042f53045116b99493b4a63826eb4fad46c`. Frozen identities and four prompt
  scope layers passed, but the run stopped with
  `TRLContractError: PPO data collator must return a mapping` before generation: 0
  completions/tokens, 0 update/optimizer/global steps and no checkpoint. Runs 2--6
  were not executed. Its verified failure backup SHA256 is
  `8fc1800417dc79ee22b6b8880986de8b4ea92efa48e48d869f2ee69ee6e34118`.
- Exact CPU reproduction used the frozen PPO seed-42 config, real 16-row dataset,
  fixed local tokenizer, `DataCollatorWithPadding`, CPU DataLoader and CPU
  Accelerator. The base returned `BatchEncoding`, a `MutableMapping` but not a
  concrete `dict`, with `input_ids`/`attention_mask` shaped `[16, 161]` and int64.
- Root cause was solely the ordered metadata wrapper's concrete-dict check. The
  compatibility boundary now accepts `MutableMapping` for attachment and `Mapping`
  for prepared-batch validation, preserving the same `BatchEncoding`. It does not
  retokenize, change padding/dtypes/response boundaries or pass metadata to model
  kwargs.
- Verification: 47 targeted and 369 full tests passed, plus Ruff, compileall,
  check_env, manifest validation, six pilot and two Stage D dry-runs, and fake
  16-completion finalization. CUDA remained uninitialized; model load, generation,
  Trainer.train, backward and optimizer calls were zero. Both historical PPO pilot
  failures and all frozen hashes remain unchanged and excluded from aggregation.

## PPO Pilot Loop Budget CPU Repair

- Baseline: clean `pivot/math-rlvr` at
  `ad21511416ea7cb8384dd42087cd782aee5ea167`. Third immutable failure
  `ppo_matched_0p5b_seed42_20260714T085240Z` remains excluded with its original 16
  completions, 574 tokens and partial-update evidence unchanged.
- Four-layer audit proved seed-42 resolves and constructs TRL 0.24.0 with 16 episodes,
  per-device batch 4, GA4, one PPO epoch and one minibatch. TRL derives one outer batch,
  one 16-example minibatch, four microbatches and one synchronized optimizer/global step.
- Root cause was guard duplication: the old optimizer wrapper counted every microbatch
  call. The second call failed as `2/2` while `sync_gradients=false`; TRL had not entered
  epoch or minibatch 2.
- The guard now uses idempotent real loop keys and the compatibility shim records the
  loop/optimizer event only at the fourth, synchronized microbatch. True second
  epoch/minibatch/outer and optimizer step remain fail-closed.
- Verification passed: targeted tests, 350 full tests, Ruff, compileall, check_env,
  manifest validation, PPO pilot dry-runs for all three seeds, Stage D PPO dry-run and
  fake 16-completion finalization using an actually constructed CPU `PPOConfig`. CUDA,
  real model loading, generation, Trainer.train, backward and real optimizer counts
  remained zero. See `reports/pilot_0p5b/ppo_loop_budget_fix.{md,json}`.

## PPO Pilot Sync-Boundary Failure

- New run `ppo_matched_0p5b_seed42_20260716T111934Z` executed exactly once from Phase A commit
  `3641ee74bb2fdc4145cee63bcaf2849496a03c3c`. Frozen identity/scope, sequential loader,
  16 pair keys and model/optimizer roles passed.
- It generated 16 completions / 574 tokens and 16 rewards, then failed with
  `TRLContractError: unexpected PPO microbatch count at optimizer boundary: 1 != 4`.
  Counters remained update/optimizer/global `0/0/0`; no completion rows, metrics or
  checkpoint finalized. It is excluded from scientific aggregation.
- Real Accelerate semantics contradicted the CPU hook assumption: `sync_gradients` was
  true on the first optimizer-wrapper call. The run was not retried; remaining five
  jobs were not executed; no further code repair occurred. GPU returned to 0 MiB/no
  process. Verified failure backup SHA256:
  `29a6e478a6692782700c23900b2c0836af5dd132961f835867834461490014e1`.

## Accelerate Backward-event PPO Guard

- Real Accelerate 1.14.0 CPU GA4 evidence showed four accumulate/backward and wrapper
  step calls but one bottom optimizer update; its first and only call saw sync=true.
- An exact consumed-one-batch reproduction found the missing TRL interaction: default
  `sync_with_dataloader` forced all four inner contexts to sync and would update four
  times. Setting it false yielded sync `[false,false,false,true]`, four batch-4
  backwards, 16 samples and one bottom update.
- The PPO shim now uses backward events and actual gradient-enabled forward batch sizes
  as microbatch authority. The bottom optimizer hook only enforces the real-step cap;
  it never infers microbatch count. The validated evidence is attached to the existing
  loader contract after success.
- 358 full tests and all CPU gates/dry-runs passed. Tiny reproduction backward/optimizer
  work is reported separately; Qwen/tokenizer/CUDA/generation and the real TRL PPO train
  loop remained unused. Frozen identities and four historical failure trees did not
  change. See `reports/pilot_0p5b/accelerate_microbatch_semantics.{md,json}`.

## Matched PPO Pilot Seed 42 Success

- `ppo_matched_0p5b_seed42_20260716T114710Z` succeeded once from `3c88fb1`: 16 completions/evidence, 574 tokens, backward
  4 × batch4, sync `[false,false,false,true]`, and epoch/minibatch/bottom optimizer/
  update/global `1/1/1/1/1`.
- Reward mean/std were `0.078125/0.03940475`; all four group variances were nonzero,
  but statuses were 14 format errors and 2 invalid expressions with pass@1/pass@4 zero.
- Finite policy/value losses were `0.04062834/6.80511189`. The adapter/head-only
  checkpoint passed. Peak nvidia-smi was 11,189 MiB; GPU-hours/cost were
  `0.00497375/CNY 0.04417`; parent release returned to 0 MiB/no process.
- Verified backup SHA256: `dd1833ea6fa75a6a8af1d7fba366b05e498b0524ee726a526eb7fa294f89b7f6`.
  The four older failures remain immutable/excluded. Next authorized job is GRPO seed42.

## Matched GRPO Pilot Seed 42 Success

- `grpo_matched_0p5b_seed42_20260716T115219Z` ran exactly once and succeeded with
  16 completion/evidence rows, 545 generated tokens, four microsteps and optimizer /
  update / global counters `1/1/1`.
- Reward mean/population standard deviation were `0.078125 / 0.03940475066537028`;
  all four groups had nonzero variance and zero-advantage groups were zero. Canonical
  results remained 13 format errors, two invalid expressions and one invalid number
  usage, with zero pass accuracy.
- Loss, grad norm and entropy were finite. Frozen beta was zero, so KL is correctly
  unavailable/null rather than fabricated as zero.
- The 16 comparison keys were complete and unique; the sole adapter-only checkpoint
  passed. Peak nvidia-smi memory was 3301 MiB, GPU-hours were 0.00315071, and cost was
  CNY 0.02798. The verified backup SHA256 is
  `6c31e369554cc4272235981722c96ff65de69614eb435b580673b061003322fb`.
- Parent GPU release passed. Continue only with frozen GRPO seed 123 after committing
  this report and restoring a clean worktree.

## Matched GRPO Pilot Seed 123 Success

- Run `grpo_matched_0p5b_seed123_20260716T115714Z` executed once: 16 completions,
  724 generated tokens, four microsteps and optimizer/update/global `1/1/1`.
- Reward mean/population std were `0.084375 / 0.03840064289826409`; four groups had
  nonzero variance, while the 12 format errors and four invalid expressions yielded
  zero pass accuracy. Loss/grad/entropy were finite; KL is null because beta is zero.
- The 16 comparison keys, frozen identities, adapter-only checkpoint and GPU release
  passed. Wall/GPU-hours/cost were `11.0505 s / 0.00306958 / CNY 0.02726`. Backup
  SHA256: `63d4e598169ec655a5d2c52e023606e1fb8b6b6915345d8a948d3273b45bf6f3`.
- Proceed only to frozen PPO seed 123 after committing and returning clean.

## Matched PPO Pilot Seed 123 Success

- `ppo_matched_0p5b_seed123_20260716T120000Z` ran once: 16 completions/evidence rows,
  565 tokens, four batch-4 backward events, sync false/false/false/true and exact
  epoch/minibatch/optimizer/update/global `1/1/1/1/1`.
- Reward mean/std were `0.09375 / 0.03903123748998998`; one of four groups had zero
  variance. Canonical accuracy was zero; policy/value losses and required metrics were
  finite. The role-separated adapter/head checkpoint and 16 keys passed.
- Peak VRAM/wall/GPU-hours/cost were `10015 MiB / 15.2371 s / 0.00423253 / CNY 0.03758`.
  Backup SHA256: `2ac5a29a453e8e71bda7aea2d498e798b0f6d29393c20286bd0e585fe7f326f9`.
- Parent GPU release passed. Continue only to frozen PPO seed 2026 after commit/clean.

## Matched Suite Recovery Reconciliation

- Raw run evidence, checkpoint hashes, persistent archives and Git-safe checksums
  establish `ppo_matched_0p5b_seed42_20260716T114710Z` as position-1 success:
  16 completions, 574 tokens, four batch-4 backward events and exact one-step
  counters. The derived recovery row was corrected; no original run changed.
- Historical PPO seed-42 failures remain `073357Z`, `082003Z`, `085240Z`, and
  `111934Z`.

## Matched PPO Pilot Seed 2026 Success

- `ppo_matched_0p5b_seed2026_20260716T122924Z`: 16 completions, 512 tokens,
  four batch-4 backward events, sync false/false/false/true, and exact
  epoch/minibatch/optimizer/update/global `1/1/1/1/1`.
- Reward mean/std `0.09375 / 0.04284784125250652`; all four groups had nonzero
  variance, while all canonical statuses were FORMAT_ERROR.
- Backup SHA256: `2fb70aacae8bcf8cfdb5bba359b2ce8b95e62dd1505f15f7d0af31d28e1ee03e`.

## Matched GRPO Pilot Seed 2026 Success

- `grpo_matched_0p5b_seed2026_20260716T123716Z`: 16 completions/rewards,
  703 tokens, four microsteps and optimizer/update/global `1/1/1`.
- Reward mean/std `0.078125 / 0.04990224819584785`; all four group variances
  were nonzero. Loss/grad norm/entropy were `0.4634579420 / 2.6247079372 /
  0.4789166301`; KL is unavailable because beta=0.
- Backup SHA256: `124818a2f9951b4927f3c0411cf5f2fdb37c1eaf3714f54b87c1d856db52eed0`.

## Matched 0.5B Pilot Aggregate

- Six valid runs completed: PPO/GRPO for seeds 42, 123, and 2026; each had 16
  completions and one optimizer/global step.
- Total measured resource window was 83.274789 seconds, 0.023131886 GPU-hours,
  and CNY 0.205411. All pass@1/pass@4 values were zero.
- The pilot supports execution/artifact comparability only. It does not establish
  learning, statistical significance, or PPO/GRPO superiority.
- Historical engineering failures `073357Z`, `082003Z`, `085240Z`, and
  `111934Z` are excluded. Recommended next step: CPU-only 1.5B GSM8K+MATH
  configuration freezing.

## Stage E: Formal 1.5B CPU-only freeze

- Exact model: `Qwen/Qwen2.5-1.5B-Instruct` revision
  `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`. Official API/config metadata only was
  queried; no weights or tokenizer were downloaded or loaded.
- Formal prompt: `prompt_v2_formal_math`, SHA
  `89e459da827474d9bcc66e4407b06b5f8a968ce10d0be92e830c59fd9830a994`.
- Formal reward: `shaped_v3_domain`, SHA
  `b9eda9520bb0271e28f6c209db85a408cdc0a65c2d403871b2b0fcc06e06a463`;
  answer block/strict protocol/domain-valid/canonical-correct weights are
  0.05/0.05/0.10/0.80. Countdown number usage is absent.
- Parser SHA remains `655c30f2...ba0ad`; formal verifier router SHA is
  `ac360315...886fd`, with GSM8K `91f9de47...4b50` and MATH `0a4fb547...efa7`.
- Formal data registry identity is `d7c53f61...e7393`; 128 train, 64 validation, 400
  final-test records and fixed 100-problem pass@4 subsets have zero prohibited overlap.
  Schedule SHA `a4b3745e...8b6ee` freezes 32 updates of 2 GSM8K + 2 MATH.
- Historical validation manifest provenance is disclosed, not rewritten:
  `source_split=validation`, physical source `train`, derived selection split
  `validation`.
- Per algorithm/seed: seeds 42/123/2026, 32 updates, 512 completions, 131,072-token
  cap, 32 optimizer/global steps, checkpoints 8/16/24/32. PPO rollout16, batch4/GA4,
  epoch/minibatch1/1; GRPO generation16, num_generations4, batch4/GA4, no shuffle.
- Static parameter contract from pinned metadata: policy LoRA 4,358,144; PPO value
  LoRA 1,089,536; scalar head 1,537; PPO optimizer union 5,449,217; GRPO optimizer
  4,358,144. Policy/value overlap, reference trainables, and reward trainables are zero.
- Test is baseline/final-only; step32 is fixed before execution. Evaluation stores all
  completions, per-problem metrics, seed raw/mean/sample SD, and paired problem bootstrap
  95% intervals. Three seeds do not support exaggerated significance claims.
- Planned future stages require separate authorization: model download; CUDA/load
  sanity; baseline; PPO42; GRPO42; seed42 review; remaining four balanced runs; final
  test; CPU aggregation. Stage E authorizes none of them.

## Formal 1.5B four-run amendment

- The original six-run Stage E plan remains preserved at `499fea9f`; the active suite
  now contains PPO/GRPO seeds 42 and 123 only.
- Active order: PPO42, GRPO42, seed-42 review, GRPO123, PPO123, four step-32 final
  evaluations, then CPU aggregation. Active-suite SHA256:
  `f6de8c555a70837d08c1e34e13a738a21ce247b3a09531df6244a2f1d3ef53bd`.
- Seed-2026 descriptors were not deleted or changed; their status is
  `reserved_not_scheduled`, outside the active registry, costs, queue, and statistics.
- Four training runs retain 32 updates, 512 completions, 131,072-token cap, and
  checkpoints/validation at 8/16/24/32. Two seeds do not support statistical-
  significance or general-superiority claims.

## Formal 1.5B multi-update runtime freeze

- The CPU-safe formal guard accepts only four active config path/SHA identities and
  enforces 32 updates, 512 exact comparison keys, 131,072 generated tokens, and
  checkpoints/validation at steps 8/16/24/32.
- Same-run resume binds run/config/suite identity and exact counter/key prefixes;
  cross-run, non-checkpoint, and completed-step resume fail closed.
- Fake PPO/GRPO 32-step execution, overflow failure backup, and baseline/final
  evaluation finalization pass without model/tokenizer/CUDA/Trainer activity.
- Formal evaluation quantity is two baseline seeds (1,600 completions), 16 validation
  passes (1,024), and four final checkpoint evaluations (3,200).
- Expected/ceiling full Stage E execution is 6.8767/14.0 GPU-hours and CNY
  61.06/124.32. The next stage still requires explicit 1.5B download authorization and
  then an independently authorized CUDA/model-load sanity.

## Stage E handoff pause

- Stage E.1 implementation is paused at verified code/runtime baseline
  `8ab031e567f877d48af75adb0ea5a6fba9e8bf55` on `pivot/math-rlvr`.
- `PROJECT_HANDOFF.md` is the concise current-state entry point.
  `docs/NEXT_TASK.md` preserves the complete unexecuted CPU-only formal 1.5B
  model-bound CLI wiring acceptance contract.
- The sole current blocker is real-model formal PPO/GRPO/evaluation CLI wiring. The
  generic 32-step CPU contract exists; no 1.5B download/load, CUDA, generation,
  evaluation, or training is authorized.
- Authority order is Git/configs/manifests/original artifacts, then
  `PROJECT_HANDOFF.md`, `docs/NEXT_TASK.md`, `AGENTS.md`/`memory.md`, and historical
  chat. Never alter historical evidence to match a derived handoff.


## Stage E.1 Formal Model-Bound CLI and Exact Resume Completion

- Documentation/evidence contract commit: `4107fca`; model-bound CLI commit:
  `c082ec8`; trusted same-run resume commit: `e3cf482`.
- PPO, GRPO, and evaluation require separate dual confirmations and accept only exact
  active path/SHA identities. Seed 2026 remains rejected as reserved.
- Recovery checkpoints bind same-run identity, steps 8/16/24, counter/key prefixes,
  optimizer/scheduler/trainer state and RNG state; full Qwen weights are forbidden.
- CPU gates completed with no CUDA/model/generation/training. This stage did not
  authorize the later model download or baseline.

## Stage F: Pinned Qwen 1.5B Snapshot and CUDA Sanity

- Download commit `5bbf913358f018e413ea70ef3ce34fa38afcfa1d`; CUDA sanity commit
  `2a86af7572d4f6b1419b1012b9b19f50cf9cbade`.
- Exact revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` downloaded once to the
  canonical Hugging Face cache. Snapshot size was 3,098,973,447 bytes; download took
  797.2488 seconds with zero retries and passed offline/local-only re-resolution.
- CUDA sanity run `cuda_load_sanity_qwen25_1p5b_20260718T113620Z` loaded the BF16
  1,543,714,304-parameter model and tokenizer locally, rendered two frozen prompts,
  and produced finite logits. It performed no generation, LoRA, Trainer, backward,
  optimizer, checkpoint, baseline, or training.
- Sanity wall/GPU-hours/cost were 6.630834 s / 0.001841898 / CNY 0.016356; peak
  nvidia-smi memory was 3,915 MiB. The worker allocator's 32 MiB residue was a warning;
  post-process GPU state was 0 MiB/no compute process.
- Static Stage F backup:
  `/root/autodl-fs/math-rlvr-backups/code-rlvr_stage-f_2a86af7.tar.gz`, SHA256
  `aba02b5508b12d86c8a4f2ece50a00bf62e9551b082aff99acbcc688ff0a1195`.

## Stage G: Baseline Serialization Failure and Repair

- Immutable attempt `baseline_formal_1p5b_seed42_20260718T114907Z` failed while
  serializing the first reward evidence because `RewardResult.to_dict()` is flat and
  the evaluator accessed a nonexistent `["components"]` field. Persisted evidence was
  0/800; generated-token count is unavailable because no completion record was safely
  appended. It is excluded from scientific statistics.
- Resource wall time was 6.312120 s; derived GPU-hours/cost were 0.001753367 / CNY
  0.015570. Failure backup SHA256:
  `b32174ddea42dd458a86a20aa53948b8b56fcf838f996fced22cd2648a0bd6d4`.
- Commit `b47966e` preserves the failed attempt. Repair commit
  `cce3a212f7f5a60edbf43ffd6eef4794850173f6` reuses the existing flat reward evidence
  mapping and regression-tests GSM8K/MATH JSON append/readback without changing reward,
  prompt, parser, verifier, data, sampling, or config identity.

## Stage G.1: Prompt-Length Failure

- New immutable attempt `baseline_formal_1p5b_seed42_20260718T120909Z` used the repaired
  serializer but failed at `math:HuggingFaceH4/MATH-500:test:219`: exact rendered
  prompt length 800 exceeded the frozen 512 cap.
- The run persisted 642/800 evidence rows and 74,968 exact generated tokens before the
  failure. Wall/GPU-hours/cost were 1,620.691772 s / 0.450192159 / CNY 3.997706.
- Failure backup SHA256:
  `20f639e6432921ff8008607af2612f4412cfacd88ce1192f719763cf038418e1`.
- The 642 rows are immutable engineering evidence, excluded from baseline statistics,
  and never copied, resumed, or spliced into a later config identity.

## Stage G.2: Full Prompt Audit and Capacity Amendment

- The pinned tokenizer audit covered 1,192 actual-mode rows / 592 unique problems.
  Maxima were train 713, validation 339, test/overall 800, GSM8K 262, MATH 800, and
  MATH500 Level 1–5 279/257/415/800/767. No truncation occurred. Three unique problem
  IDs exceeded 512.
- Commit `edecfcf503cff8ee8aef3c7ef2136dae04e192b7` records the public
  `post-freeze prompt-length capacity amendment`: shared evaluation/PPO/GRPO cap
  512 -> 832; completion cap stays 256; 1,088 remains below context 32,768.
- Evaluation raw/canonical SHA are now `85100dd0...f4a35` /
  `d8ba5ab8...fe44`; active-suite raw/canonical SHA are `11869c63...2017` /
  `1d7c29f7...9d600`. Active config SHAs are PPO42 `1093e87a...ec43`, GRPO42
  `3371d231...9199`, GRPO123 `cc95138f...75e`, PPO123 `3d6cc1f3...c15e`.
- No scientific variable except prompt capacity changed. Previously fitting token IDs
  are unchanged and PPO/GRPO remain matched. Static amendment backup:
  `/root/autodl-fs/math-rlvr-backups/code-rlvr_stage-g2-prompt-cap_edecfcf.tar.gz`,
  SHA256 `3836d7379b47aa4a934e009386d570b1a4a455a32497f2f2c6b45286174fcdbb`.

## Stage G.2: Successful Frozen Baselines and Next Decision

- Scientific seed-42 run `baseline_formal_1p5b_seed42_20260718T125833Z`: 800/800,
  96,150 tokens, sampled pass@1 0.040, pass@4 0.100, 0.584779932 GPU-hours,
  CNY 5.192846. Backup SHA256 `77105f38c67ecb773edc54cedb63fe489df1298155d2eabbe0cf07e7b7cd5a13`.
- Scientific seed-123 run `baseline_formal_1p5b_seed123_20260718T133624Z`: 800/800,
  91,651 tokens, sampled pass@1 0.025, pass@4 0.060, 0.569322944 GPU-hours,
  CNY 5.055588. Backup SHA256 `e473075db8123664c13a3d77e8c9960be108fe881aa5deef88c04660bff2edf0`.
- Greedy accuracy is null/unavailable because the frozen protocol has no separate
  greedy completion. Both success runs have complete checksums and are the only runs
  included in `reports/formal_1p5b/01_baseline_results.md`.
- Result commit `287f7d313c5ad8ac1500eb416eeacd605c3298f3`; final Git backup
  `/root/autodl-fs/math-rlvr-backups/code-rlvr_stage-g2-baseline_287f7d3.tar.gz`,
  SHA256 `f102571c8c1d3909080abc1c5ff2fdff8d04236b753c553438d9d47e9fee6cdc`.
- Scientific inclusion decision: only the two 800/800 post-amendment runs are included;
  the 0/800 serialization and 642/800 prompt-length attempts remain excluded forever.
- With baseline and GPU release verified, there is no technical training blocker. The
  unique next task is Stage H formal PPO seed 42, requiring a new explicit GPU
  authorization. PPO success will not authorize GRPO automatically.

## Stage H: First Formal PPO Seed-42 Attempt Failed

- Authorized execution HEAD was `176f738cf949c06b305e2c80a7c84d0082e6eed1`; the
  difference from baseline commit `287f7d3` was documentation-only. Frozen PPO42 config
  raw SHA `1093e87a...ec43` and suite raw/canonical SHA `11869c63...2017` /
  `1d7c29f7...9d600` passed preflight. The local pinned snapshot, baseline checksums,
  idle H800, offline mode, and writable storage also passed.
- The sole authorized command created `ppo_formal_1p5b_seed42_20260718T150510Z` and was not retried. Live TRL
  stdout reached the displayed 8/32 checkpoint boundary with `episode=128`, then
  `_normal_metrics` raised `FormalRuntimeError: formal Trainer did not expose required
  grad norm`. This violates the Stage H policy that missing optional grad norm is
  null/unavailable and nonblocking.
- Finalized `completions.jsonl`, `metrics.jsonl`, and `verification_results.jsonl` are
  empty and formal counters are zero. The transient display is not primary evidence;
  generated-token count and scientific reward/loss/entropy/completion metrics are
  unavailable. No checkpoint validation or final test ran. The attempt is an immutable
  engineering failure, excluded from scientific aggregation.
- Partial `checkpoint-8` holds policy/value adapters and scalar head only. It lacks
  optimizer, scheduler, RNG, runtime/counter, comparison-key prefix, and trusted
  inventory state; it is not resume-capable and contains no full base weights.
- Measured wall/GPU-hours/cost: 181.951470 s / 0.050542075 / CNY 0.448814. Peak
  nvidia-smi VRAM was 28,655 MiB and mean utilization 36.0585%. After worker exit the
  GPU was 0 MiB with no compute process.
- Verified failure backup:
  `/root/autodl-fs/math-rlvr-backups/ppo_formal_1p5b_seed42_20260718T150510Z.failure.tar.gz`, SHA256
  `76896c5b3db3ee4439566b8b68c0cad798af5b5610f393138aa23eba6c40debb`. Git-safe evidence is under `reports/runs/ppo_formal_1p5b_seed42_20260718T150510Z/`.
- Next decision: bounded CPU-only repair of optional grad-norm availability and
  per-update evidence persistence. No new PPO attempt, GRPO, seed 123, validation, or
  final test is authorized.

## Stage H.1: Optional Grad Norm and Incremental Evidence Repair

- Starting HEAD `108aa260481710ceb90080200af348f7a0ec0765` was clean. Scope stayed
  CPU-only and limited to two defects; no model/tokenizer/CUDA/generation/real Trainer/
  backward/optimizer/evaluation ran.
- Root cause 1: the failed TRL rows had no `grad_norm` or `train/grad_norm`; required
  parsing and the formal required-metric set caused the checkpoint-8 exception. Missing
  aggregate/policy/value grad norms now persist null/unavailable/reason/raw-key-null;
  finite values retain their raw key and NaN/Inf still fails closed.
- Root cause 2: `CompletedTrainerBackend` did not replay evidence into the observer
  until all 32 steps returned. The guarded PPO log callback now updates the observer
  after each logged update and before checkpoint callbacks; the observer atomically
  rewrites the existing completion/metric JSONL prefixes.
- Fake step-8 checkpoint failure preserved exactly 128 ordered completion rows, eight
  metric rows and counters 8/8/8. Existing 32-step fake and same-run resume passed.
- Verification: 23 related pytest passed; affected Ruff and compileall passed; formal
  PPO dry-run reported no training started. Full pytest was intentionally omitted.
- Historical failed run/checksums and all frozen identities remain unchanged. The next
  real PPO42 attempt needs a new explicit authorization and new run ID; GRPO remains
  unauthorized.

## Stage H: Second Formal PPO Seed-42 Attempt Failed

- From clean authorized HEAD `1d31f56386857909c881bba1a7c5302166cf9682`, the sole
  command created `ppo_formal_1p5b_seed42_20260719T131800Z` and executed exactly once.
  Frozen config `1093e87a...ec43`, suite raw/canonical `11869c63...2017` /
  `1d7c29f7...9d600`, pinned model revision, offline snapshot, baseline checksum,
  storage, and idle H800 passed preflight.
- Stage H.1 incremental evidence worked: 32 metric rows and 512 ordered completion rows
  persist 32 update/optimizer/global steps and 51,369 exact training rollout tokens.
  Training stayed below the 131,072 cap. No automatic retry, GRPO, seed 123, baseline,
  validation, or final test ran.
- Checkpoint directories 8/16/24/32 contain policy/value adapters, scalar head,
  optimizer, scheduler, RNG, trainer/runtime prefixes, identity, and SHA256 artifact
  manifests with `base_weights_included=false`. They are not authorized for resume or
  evaluation because the attempt failed before validation and run-level checkpoint
  finalization.
- Failure: after training reached update 32, `CompletedTrainerBackend` replayed the
  scheduled checkpoint sequence from step 8. The incremental observer guard was
  already at update 32, so `record_checkpoint(8)` raised `formal checkpoint cadence
  mismatch`. Validation rows/tokens are 0/0; the existing seed-42 test baseline remains
  an independent reference and is not a validation delta.
- Diagnostic training means across the excluded run: reward 0.2365234, canonical pass
  0.16796875, format 0.486328125, policy/value/total loss
  0.0099564/4.4741664/0.4573730, approximate KL 0.0006067, ratio 0.994854,
  clip fraction 0.0046075, and TRL native entropy 1.2805083. Grad norms, unified
  response-token entropy, entropy std, advantage, and return remain null/unavailable
  with reasons. These are failure diagnostics, not a scientific PPO result.
- Resource evidence: 751.268899 seconds, 0.208685805 GPU-hours, CNY 1.853130,
  53,151 MiB peak nvidia-smi VRAM and 34.8984% mean utilization. Post-process GPU state
  was 0 MiB/no compute process.
- Verified immutable failure backup:
  `/root/autodl-fs/math-rlvr-backups/ppo_formal_1p5b_seed42_20260719T131800Z.failure.tar.gz`,
  SHA256 `f63812afed44cdc9f0fcafdf0931454548da1a4ce145840ebf91bb6fa5a6d7c5`.
  Git-safe evidence is under
  `reports/runs/ppo_formal_1p5b_seed42_20260719T131800Z/`; the run is excluded from
  scientific aggregation.
- Next decision: one bounded CPU-only repair aligning checkpoint/validation cadence
  with incremental observer updates. No repair or further GPU execution is authorized.

## Stage H.2: Independent Validation Cadence and Recovery Eligibility

- Starting HEAD `135ca10e6d002ce6d3e29e5d0cde56b4e6ec29eb` was clean. Scope remained
  CPU-only; no CUDA, model/tokenizer, generation, Trainer, backward, optimizer, PPO,
  GRPO, baseline, validation, or final test ran.
- Root cause: the PPO callback had already advanced the training observer through
  update 32. `CompletedTrainerBackend` then correctly entered deferred checkpoint/
  validation replay at step 8, but checkpoint and validation guards incorrectly
  required each replayed step to equal the current training update.
- Repair: training stays a monotonic 1..32 cursor. Checkpoint and validation cursors are
  independent ordered sequences 8/16/24/32. Deferred steps may be below the completed
  training cursor, never ahead of it; duplicates, skipped/illegal steps and validation
  without its trusted checkpoint still fail closed. Checkpoint inventory/identity is
  validated before its cadence event is committed.
- Verification: 19 targeted CPU/fake pytest cases passed, including online and
  post-training deferred cadence, ordering/duplicate/missing/illegal rejection,
  training-counter/token-budget independence, trusted resume/identity rejection and
  incremental step-8 evidence. Affected Ruff/compileall, formal PPO dry-run and formal
  validation protocol dry-run passed.
- Read-only original evidence audit replayed 32 metric rows, 512 completion rows and
  51,369 tokens through the formal training guard. All comparison keys, completion
  IDs/masks/counts, rewards and verifier evidence passed.
- Checkpoints 8/16/24/32 passed actual file size/SHA256 inventory, prefix counts,
  model/config/suite/prompt/reward/parser/verifier identity and existing validation
  selection. No base weights exist. Original checksums file SHA256 remains
  `43295b905f4175a41de21cd41e71e1e42d687c80a411af0421f91ecc3133e372`;
  failure backup remains `f63812af...d7c5`.
- Eligibility: `training_contract=complete`, `training_evidence=complete`,
  `checkpoint_contract=complete`, `validation_contract=pending`,
  `validation_only_eligible=true`, `training_resume_authorized=false`, and
  `training_rerun_required=false`. The original run is still
  `engineering_failure_after_training` and excluded until four validations complete.
- The unique next task is a separately authorized Stage H.3 validation-only sequence
  for checkpoints 8,16,24,32. Expected/ceiling totals are about 20/40 minutes,
  0.3333/0.6667 GPU-hours and CNY 2.96/5.92. No command starts automatically.

## Stage H.3: PPO Seed-42 Validation-Only Recovery

- Execution base HEAD `2b4395171e72ef77032b567490960226d3b1bb1c` was clean. Frozen
  config/suite/model/prompt/reward/parser/verifier identities, original run checksums,
  four checkpoint inventories, local snapshot, offline mode, storage, and idle H800
  passed the bounded preflight.
- Four commands ran once each, in strict order, with zero retries:
  `ppo_validation_formal_1p5b_seed42_step8_20260720T020928Z`,
  `...step16_20260720T020928Z`, `...step24_20260720T020928Z`, and
  `...step32_20260720T020928Z`. Each persisted 64/64 completion rows.
- Generated validation tokens were 7,533 / 7,848 / 7,663 / 7,497 (30,541 total).
  Sampled pass@1 was 4.6875% / 3.125% / 3.125% / 3.125%. Pass@4 is
  null/unavailable because the protocol has one candidate per problem. Format/parseable
  was 14.0625% at step 8 and 10.9375% thereafter.
- Validation cost was 0.271514265 GPU-hours / CNY 2.411047, with 977.451 seconds
  summed wall time and 3,847 MiB maximum nvidia-smi VRAM. Backup SHAs are
  `bed8ab2d...caf2ef`, `eea05aae...e1efc`, `195dcc00...f7874`, and
  `6dfbae6c...cfb723`; all run checksums and backups verified. GPU ended at 0 MiB with
  no compute process.
- The native validation aggregate leaves pass@1 null because the rows use
  `sample_kind=validation`; the derived report uses complete per-problem
  `canonical_correct` evidence. The training runtime's stale nested-component
  `valid_answer_rate=0` is marked unreliable; parseable is derived from immutable
  canonical statuses rather than reporting a fabricated zero.
- Original training run `ppo_formal_1p5b_seed42_20260719T131800Z`, summary,
  checkpoints, and checksum remain unchanged and individually excluded. Composite
  `ppo_formal_1p5b_seed42_composite_20260720T020928Z` is
  `scientifically_complete_with_recovered_validation`; no training rerun/resume,
  baseline, final test, GRPO, or seed-123 execution occurred.
- Aggregation exposed a true report-truthfulness blocker for future GRPO: formal
  training still computes native `valid_answer_rate` from obsolete nested
  `components.valid_answer`, producing false zeros after RewardResult evidence became
  flat. The PPO composite corrects only its derived report from canonical statuses.
- The unique next task is a bounded CPU-only Stage H.4 field-mapping repair with
  targeted tests. GRPO seed 42 still requires separate authorization afterward.

## Stage H.4 valid-answer metric truth repair

- Starting HEAD `68f50a5fea9ec1183adabf04fc60d8daf1dc7d16` was clean; the stage was CPU-only.
- Root cause: formal training still read `components.valid_answer` after
  `RewardEvaluation.to_dict()` became flat. The resulting native zero was reporting
  telemetry only and never entered reward, loss, advantage/return, optimizer, scheduler,
  selection, early stopping, or budget logic.
- The shared PPO/GRPO mapping now aggregates flat `valid_answer_component > 0`, with
  definition version, numerator, denominator, raw source/status scope and explicit
  null/unavailable reasons. It is extracted-answer probe validity and is not equivalent
  to canonical parseable rate. Contradictory aggregate/evidence fails finalization.
- 44 targeted tests plus affected Ruff/compileall and both formal dry-runs passed. No
  CUDA, model/tokenizer, generation, Trainer, backward, optimizer, PPO/GRPO execution,
  baseline, validation, or final test ran.
- Historical PPO artifacts remain immutable; checksums-file SHA256 remains
  `43295b905f4175a41de21cd41e71e1e42d687c80a411af0421f91ecc3133e372`. The composite
  remains `scientifically_complete_with_recovered_validation`; no PPO rerun is needed.
- Next decision: separately authorize formal GRPO seed 42. It must not start
  automatically.

## Stage I formal GRPO seed-42 success

- Execution base was clean HEAD `548e2d371cbc09d5527aed3ed9dbf0ac1ad94a1d`. The
  frozen command ran exactly once as `grpo_formal_1p5b_seed42_20260720T031006Z`;
  automatic retries were zero.
- Training completed 32 update/optimizer/global steps, 512 completions, 50,773 exact
  rollout tokens under the 131,072 cap, and the expected 128 four-completion groups.
  Of these, 101 had nonzero variance, 27 were all-equal/zero-advantage, and 15 were
  all-zero. Mean reward/canonical pass/format/parseable was
  0.267188/19.1406%/54.6875%/46.875%.
- Checkpoints 8/16/24/32 are policy-adapter-only plus trusted optimizer/scheduler/RNG/
  counter/prefix state. They contain no value adapter/head or base weights. All run
  checksums and inventories verified.
- Frozen validation produced 64 completions per checkpoint and 29,113 tokens total,
  separate from training. Pass@1 was 3.125%/4.6875%/6.25%/7.8125%; pass@4 is
  null/unavailable under one candidate per problem. No final test or test-driven
  selection occurred.
- Measured usage was 1,189.819 seconds, 0.330505 GPU-hours, CNY 2.934886, and 11,209
  MiB peak nvidia-smi VRAM. The verified backup is
  `/root/autodl-fs/math-rlvr-backups/grpo_formal_1p5b_seed42_20260720T031006Z.tar.gz`,
  SHA256 `b584363595f99c1d3b61a7b6cc088cdda7ac38a29169058df7b30cd38bea5023`.
  GPU ended at 0 MiB/no compute process.
- Warnings: run-root incremental JSONL was filled after `train()` while trusted planned
  checkpoint prefixes remained complete; the model-role snapshot captured the lazy
  optimizer before creation, but checkpoint-32 has 224 unique optimizer IDs/state
  entries matching 224 LoRA trainable tensors; allocator and per-checkpoint resource
  summaries are unavailable. No source/config change or rerun was made.
- Same-seed PPO/GRPO review is in `05_seed42_ppo_vs_grpo.md`. It is descriptive, not a
  significance or superiority claim. The next decision is separately authorized formal
  GRPO seed 123; do not start it automatically.
