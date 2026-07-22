# Engineering postmortem

The project preserves failed attempts rather than rewriting history. They are engineering evidence and are excluded from scientific aggregates.

## Failure classes

- **Reward evidence serialization:** the first formal baseline tried to read a removed nested `components` field. A minimal flat-schema mapping fix passed CPU regression tests; the 0/800 attempt remains immutable.
- **Prompt capacity:** a later baseline reached 642/800 before a frozen 512-token prompt cap rejected an 800-token MATH500 prompt. A full-tokenizer audit produced a disclosed post-freeze capacity amendment to 832 without changing prompt text or token IDs for shorter samples. The 642 rows were not reused.
- **Optional telemetry as a blocker:** the first PPO attempt treated absent grad norm as required and delayed primary-evidence persistence. The repair made optional metrics null/unavailable and atomically persisted each completed update before checkpoints.
- **Validation cadence:** PPO42 finished 32 updates/512 completions, but post-training validation replay compared step 8 with the training cursor at 32. Training and validation cursors were separated; four trusted checkpoints were evaluated later without rerunning training. The original run remains `engineering_failure_after_training`; the transparent composite is scientifically complete.
- **External interruption:** a PPO42 final-evaluation attempt ended after 429/800 due to host power/network loss. Its prefix remains immutable and excluded; a later separately authorized run started from zero.

## What the failures changed

Fixes were limited to training correctness, safe recovery, or report truthfulness. They did not change formal data, prompt, reward weights, parser/verifier, LoRA, sampling, update/completion/token budgets, or checkpoint-32 selection. Failed runs retain their original summaries and checksums.

## What did not become a gate

Optional telemetry, PTY behavior, CRLF/SVG formatting, and offline-rebuildable plots are warnings. The project deliberately reduces noncritical engineering gates without reducing scientific evidence.
