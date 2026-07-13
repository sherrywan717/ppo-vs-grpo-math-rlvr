# Prompt A/B cleanup semantic follow-up

This is a CPU-only follow-up to immutable failed run `prompt_ab_qwen25_05b_20260713T101918Z`. The historical run, summary, completions, checksums, backup and failure status were not modified.

## Root cause evidence

The worker generated all 16 completions (813 tokens), but `RealGenerationBackend.close()` raised when its pre-process-exit allocator diagnostic remained nonzero. The exception occurred before allocator evidence was returned; the failure path called the idempotent close helper again and persisted `{}`. Therefore the actual nonzero worker allocated/reserved bytes are **not present and cannot be recovered** from the historical artifacts. They must not be guessed.

Parent evidence is conclusive for post-process release: worker PID `109901` exited; it was absent from nvidia-smi compute processes; GPU memory returned from baseline `0 MiB` to `0 MiB`; parent CUDA initialization was false. Manual post-run nvidia-smi also showed `0 MiB` and no compute process. The original failure was therefore a cleanup-gate semantic false positive, while its recorded engineering status remains failure.

## Corrected semantics

Worker allocator current/peak evidence is a pre-exit diagnostic. Nonzero current allocated/reserved values are persisted verbatim with warning `worker_allocator_nonzero_before_process_exit`; they no longer independently fail the run. The authoritative release gate is the non-CUDA parent after worker exit: PID exited, worker absent, no newly introduced compute PID, memory restored to baseline, and parent CUDA uninitialized. Any parent gate failure remains fatal. Success is still written only after generation, evidence/consistency, parent release verification, finalization and verified backup.

## Offline analysis

| Metric | v0 | v1 |
|---|---:|---:|
| Complete envelope / format | 0/8 (0%) | 2/8 (25%) |
| Reasoning open/close | 0% / 0% | 25% / 25% |
| Answer open/close | 12.5% / 12.5% | 100% / 100% |
| Prose outside envelope | 100% | 0% |
| Truncated at 128 | 2/8 (25%) | 0/8 (0%) |
| RewardStatus | 8 FORMAT_ERROR | 6 FORMAT_ERROR, 2 INVALID_EXPRESSION |

Paired transitions: six `FORMAT_ERROR -> FORMAT_ERROR`; two `FORMAT_ERROR -> INVALID_EXPRESSION` (matched seeds 43 and 46). Both v0 truncated pairs became non-truncated under v1.

Per-problem rewards:

- `countdown:train:0`: v0 `[0,0,0,0]`, variance 0; v1 `[0,0.1,0,0]`, variance 0.001875.
- `countdown:train:1`: v0 `[0,0,0,0]`, variance 0; v1 `[0.1,0,0,0]`, variance 0.001875.
- v0 zero-advantage/nonzero-variance groups: 2/0. v1: 0/2.

The two v1 INVALID_EXPRESSION texts were:

1. `<reasoning>\n13 + 6 = 19\n9 - 5 = 4\n13 * 6 = 78\n15 / 26 = 0.6\n</reasoning>\n<answer>\n13 + 6 = 19\n9 - 5 = 4\n13 * 6 = 78\n15 / 26 = 0.6\n</answer>`
2. `<reasoning>Use each of [16, 4, 11, 13] exactly once to make -4.</reasoning>\n<answer>11 - (13 * 4) = -4.</answer>`

The envelope parser accepted both, but the current verifier requires one arithmetic expression. Both answers contain `=`; the first contains multiple lines/equations and the second omits 16 and ends with punctuation, so expression validation fails. Historical `verifier_detail` is empty; this explanation is a CPU replay/code-path diagnosis, not a fabricated historical field.

v1 satisfies all four review predicates: higher envelope rate, at least one complete envelope, no truncation regression, and at least one nonzero-variance problem group. This evidence is real and useful, but the original run remains failure and v1 is not automatically activated.
