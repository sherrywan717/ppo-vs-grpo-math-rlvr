# Stage O.2 CPU validation

Status: **passed**. This stage initialized no CUDA context, loaded no model weights,
generated no completion, and performed no Trainer, backward, or optimizer operation.

## Capacity amendment and target audit

- The immutable Stage O.1 evidence remains preserved. It recorded 256/256 audited
  targets, 48 targets over the old 256-token cap, one prompt over the old 832-token
  cap, and no actual truncation.
- `post_freeze_capacity_amendment` changes only `max_prompt_length` from 832 to 928
  and `max_target_length` from 256 to 640. The actual combined ceiling stays 1,088.
- The complete amended audit passed 256/256: zero prompt, target, or combined
  overflows; zero truncation; zero missing EOS; zero envelope, gold, label, overlap,
  hash, or order failures.
- Observed maxima are prompt 914, target including EOS 609, and actual combined
  1,019. Their margins are respectively 14, 31, and 69 tokens.
- Prompt p95/p99 are 225/317; target p95/p99 are 363/497; combined p95/p99 are
  539/733.

## Guarded runtime

- Exact contract: seed 42, 256 unique samples, one epoch, microbatch 4, gradient
  accumulation 4, effective batch 16, 64 microsteps, and 16 optimizer/global/
  scheduler steps.
- Static policy LoRA role: r16, alpha32, dropout0, q/k/v/o, with 4,358,144 expected
  trainable parameters. Base weights remain frozen and forbidden from checkpoints.
- The collator independently checks prompt <=928 and target including EOS <=640,
  then checks the actual combined sequence <=1,088 without tokenizer truncation.
  Prompt, system, user, and assistant-prefix labels are -100; assistant target and
- Each consumed batch atomically persists ordered sample IDs, exact counters, and
  cumulative active/supervised tokens before later checkpoint finalization.
  EOS are active; padding labels are -100.
- The CLI requires the exact config path/SHA, a new run directory, `--execute`, and
  `--confirm-grpo-v2-warmstart`. Model imports remain behind that boundary.
- The checkpoint role is policy-adapter-only plus optimizer, scheduler, RNG,
  trainer/runtime state, data cursor, frozen identities, and SHA inventory. GRPO-v2
  receives only the verified policy adapter and starts a fresh GRPO optimizer.
- Success and failure paths reuse the existing verified formal archive helper.

## Nested pass@10 contract

The secondary exploratory subset contains 50 problems and is a strict subset of the
100-problem nested pass@4 set: 25 GSM8K and 25 MATH (levels 2/4/5/7/7). Candidate 0
is shared by pass@1/pass@4/pass@10, candidates 1--3 are reused, and only candidates
4--9 are added. Each model therefore has 1,000 completions; four models have 4,000.
Pass@10 is forbidden for checkpoint selection or retraining. Historical throughput
implies 1,200 added completions cost about 1.1634 GPU-hours / CNY 10.33, with a
planning ceiling of CNY 12.20.

## Verification

- Targeted pytest: 16 passed.
- Ruff affected files: passed.
- Compileall affected modules/tests: passed.
- Guarded warm-start dry-run: passed; model/training not started.
- Existing manifest validation and GRPO-v2 contract dry-run: passed.
- `check_env`: `cuda_initialized=false`, `model_or_tokenizer_loaded=false`, and
  `generated_code_execution=false`.
- Secret/large-file audit: 1,255 files scanned; no credential pattern, forbidden
  model/checkpoint artifact, or file over 50 MiB found.
- `git diff --check`: passed.
