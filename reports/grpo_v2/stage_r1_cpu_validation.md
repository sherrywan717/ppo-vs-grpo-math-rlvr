# Stage R.1 CPU validation

Status: **passed**. No CUDA, model-weight load, generation, Trainer construction, backward or optimizer step occurred.

- Targeted pytest: 38 passed, 0 failed. Scope: capacity preflight, GRPO-v2 runtime, matched dev contract and dev safety.
- Ruff: passed on affected files.
- Compileall: passed on affected modules and tests.
- Real pinned-tokenizer GRPO-v2 dry-run: passed 512 training + 128 dev prompts; 0 new-cap overflow, 0 truncation, `cuda_initialized=false`.
- GRPO-v2 contract validation and project manifest validation: passed.
- Runtime registry/config identity validation: passed.
- `check_env`: `cuda_initialized=false`, no model/tokenizer loaded in that independent check process, no generated-code execution.
- `git diff --check`: passed.
- Secret, forbidden model/training-state filename and >50 MiB scans: zero matches.
- Full pytest was intentionally not run because Stage R.1 authorizes directly affected tests only.

The immutable Stage R failure evidence checksums and all non-capacity scientific identities were asserted in the targeted suite. Hidden-test accesses were zero.
