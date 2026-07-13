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
- The unactivated `prompt_v1_strict_concise` candidate is 157/155 tokens and places the final format instruction 5 tokens before the boundary. It retains the problem but includes no gold construction. PPO and GRPO export the same candidate renderer.
- BOS is absent; EOS is `<|im_end|>` 151645; PAD is `<|endoftext|>` 151643; padding is left and truncation is right. Runtime alignment warnings did not alter rendered text or the assistant boundary.
- Any v0/v1 generation diagnostic requires independent `--generate-only --confirm-prompt-diagnostic` authorization, must not train, and is capped at 16 completions/2,048 generated tokens. This forensic audit does not authorize it or PPO.

## Guarded Prompt A/B Runner Implementation

- CPU-only implementation entry: `python -m math_rlvr.evaluation.prompt_ab --config configs/diagnostics/prompt_ab.yaml`. No real A/B generation has been executed.
- Dry-run needs no execution flag. Real generation requires both `--generate-only` and `--confirm-prompt-diagnostic`; `--confirm-single-update` is rejected.
- Static gates precede delayed runtime imports: clean `pivot/math-rlvr`, both offline variables equal to 1, exact canonical local 0.5B snapshot, and exact diagnostic config/budget.
- Matched seeds 42–49 map by problem and generation index and are reset separately for Python, PyTorch CPU, and PyTorch CUDA in each condition. v0 RNG consumption cannot alter v1.
- Completion IDs are sliced after padded input width; attention-mask sum records unpadded prompt length. EOS is retained and post-EOS padding is excluded. Decode/re-tokenize is not used for the token budget.
- Hard limits: v0/v1, two fixed train prompts each, four completions per prompt, 128 tokens each, 16 total completions, 2,048 total tokens, 120 seconds, and 3.5 GiB nvidia-smi stop gate.
- Real code is base BF16 eval/inference only with all parameters frozen. Zero Trainer/train/backward/optimizer/training-step/checkpoint/model-write counters are required.
- v1 remains unactivated. Candidate review requires better envelope rate, at least one complete envelope, no increased truncation, and at least one nonzero within-problem reward-variance group. All-WRONG_ANSWER can still mean no advantage signal.
- `apply_patch` remains unavailable because this host disables unprivileged namespaces; standard narrowly scoped patches plus complete diff/tests are the approved local recovery pattern.
