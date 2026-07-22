# GRPO-v2 roadmap

GRPO-v2 is a versioned single-seed improvement experiment on `improve/grpo-v2`; it never rewrites portfolio v1. Stage N freezes data and science only and authorizes no GPU work.

## Frozen sequence

1. **Stage O:** seed-42 format/solution warm-start, 256 examples, one epoch, followed by one independent 128-problem dev evaluation.
2. **GRPO-v2:** initialize from warmstart-only adapter; 128 updates over 512 unique prompts, 2,048 completions, 524,288 generated-token cap; checkpoint/dev at 32/64/96/128.
3. **Selection:** dev-only lexicographic maximum canonical pass, parseable, format; minimum truncation; then earlier checkpoint.
4. **One sealed final stage:** compare Base, old GRPO-v1 seed42 checkpoint-32, warmstart-only, and selected v2. Each model produces 400 candidate-0 outputs plus 300 additional candidates for the fixed nested subset.

## Data and test protection

Core manifests have zero content/source overlap with each other and all v1 manifests. MATH500 unseen capacity is 3/50/65/88/94; hidden test uses 3/33/43/59/62 and nested MATH uses 3/8/10/14/15. Level 1 is diagnostic-only small-n. Public manifests omit gold/solution. Hidden-test results cannot trigger retraining, prompt/reward changes, curriculum edits, hyperparameter changes, or another attempt.

## Attribution

A warmstart-only gain belongs to supervised warm-start. Only a selected v2 gain over warmstart supports incremental RLVR benefit. Format gains without canonical gains are protocol-adherence gains, not math-reasoning gains. One seed cannot establish general superiority.
