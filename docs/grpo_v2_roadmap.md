# GRPO-v2 roadmap

GRPO-v2 is a versioned single-seed improvement experiment on `improve/grpo-v2`; it never rewrites portfolio v1. Stage N freezes data and science only and authorizes no GPU work.

## Frozen sequence

1. **Stage O:** seed-42 format/solution warm-start, 256 examples, one epoch, followed by one independent 128-problem dev evaluation.
2. **GRPO-v2:** initialize from warmstart-only adapter; 128 updates over 512 unique prompts, 2,048 completions, 524,288 generated-token cap; checkpoint/dev at 32/64/96/128.
3. **Selection:** dev-only lexicographic maximum canonical pass, parseable, format; minimum truncation; then earlier checkpoint.
4. **One sealed final stage:** compare Base, old GRPO-v1 seed42 checkpoint-32, warmstart-only, and selected v2. Each model produces 300 non-subset candidate-0 outputs plus 1,000 rows from one shared n=10 batch on the unchanged 100-problem subset, totaling 1,300 completions.

## Data and test protection

Core manifests have zero content/source overlap with each other and all v1 manifests. MATH500 unseen capacity is 3/50/65/88/94; hidden test uses 3/33/43/59/62 and nested MATH uses 3/8/10/14/15. Level 1 is diagnostic-only small-n. Public manifests omit gold/solution. Hidden-test results cannot trigger retraining, prompt/reward changes, curriculum edits, hyperparameter changes, or another attempt.

## Attribution

A warmstart-only gain belongs to supervised warm-start. Only a selected v2 gain over warmstart supports incremental RLVR benefit. Format gains without canonical gains are protocol-adherence gains, not math-reasoning gains. One seed cannot establish general superiority.

## Stage O.2 frozen execution boundary

The warm-start tokenizer audit is complete. Its original cap failure remains public, followed by the authorized 928/640/1,088 capacity amendment and 256/256 pass. The next separately authorized stage is exactly one guarded seed-42 warm-start; dev evaluation remains a later authorization. Secondary nested pass@10 adds inference-scale diagnosis only and cannot affect training or checkpoint selection.

## Stage O.3 shared unbiased pass@k amendment

The O.2 50-problem pass@10 design is historical and `superseded_before_any_evaluation`. The active unchanged 100-problem subset uses one n=10 exchangeable batch per problem and exact `1-C(n-c,k)/C(n,k)` estimates for k=1/4/10. Candidate-0 accuracy over all 400 problems remains a distinct binary metric. The ledger is 1,300 completions/model and 5,200/four models; test results still cannot select or retrain anything.
