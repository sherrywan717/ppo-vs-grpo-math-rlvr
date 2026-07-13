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
