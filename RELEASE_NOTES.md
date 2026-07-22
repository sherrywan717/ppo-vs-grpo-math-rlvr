# Release notes

## v0.1.0-formal-rlvr — portfolio v1

This release freezes the public, Git-safe evidence for the Qwen2.5-1.5B PPO-versus-GRPO Math RLVR project.

### Included

- Four completed 32-update training/checkpoint-validation runs: PPO and GRPO at seeds 42 and 123.
- Matching Base, PPO42, and GRPO42 held-out final evaluations under one frozen 800-completion protocol.
- Machine-readable training, validation, final, resource, paired-comparison, error, and per-problem evidence.
- Figures rebuilt from committed CSV/JSON with a consistent Base/PPO/GRPO palette.
- Reproducibility, methodology, limitations, safety, postmortem, interview, and GRPO-v2 planning documents.
- A public-file manifest and checksums; full checkpoints, model cache, optimizer state, credentials, and large run archives remain external.

### Headline result

At seed 42, Base/PPO/GRPO sampled pass@1 was 4.0%/3.75%/7.0%. GRPO improved over Base by +3.0 pp (paired bootstrap 95% CI [+1.0,+5.0], McNemar p=0.00754) and over PPO by +3.25 pp (95% CI [+1.25,+5.5], p=0.00443). Positive pass@4 deltas are trends because their confidence intervals cross zero.

### Deferred work

GRPO seed-123 and PPO seed-123 checkpoint-32 final evaluations are `deferred_not_executed`. They are not represented as zeros or inferred results. The next mainline is a separately versioned, CPU-designed GRPO-v2 improvement phase that must not tune against the held-out test.
