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

## Stage P execution status

Warm-start `warmstart_grpo_v2_seed42_20260722T051218Z` is complete and immutable: 256 samples, one epoch, 64 microsteps, 16 optimizer/global/scheduler steps, checkpoint-16 artifact `507749d3...92f0`. Base and warm-start dev-v2 remain `not_executed_evaluator_unavailable` because no frozen model-bound dev evaluator existed; no result is represented as zero. The next stage is CPU-only evaluator implementation/freeze, then a new explicit authorization for the two matched dev runs. GRPO-v2 and hidden test remain blocked and unexecuted.

## Stage P.1 matched dev evaluator

The shared evaluator is frozen at `configs/grpo_v2/dev_evaluation_seed42.json` (raw SHA `8501bfb9...2528`). It evaluates 128 ordered dev-v2 problems with one candidate each, identical per-problem seeds, prompt/parser/verifier/reward and sampling for Base and warm-start. The only model-state difference is the exact immutable checkpoint-16 policy adapter. Pass@4/pass@10 are unavailable by protocol. Phase A CPU gates passed; Phase B runs Base then warm-start once each. GRPO-v2 and hidden test remain unauthorized.

## Stage P.1 result and next gate

Matched Base and warm-start dev-v2 runs both completed 128/128. Base pass@1 was 6/128 and warm-start 8/128; format improved 17/128 to 23/128. The paired +1.5625 pp pass@1 delta has 95% bootstrap CI [-2.34375,+6.25] pp and McNemar p=0.7265625, so the gain is uncertain and dev-only. The warm-start is eligible for explicit GRPO-v2 review because format and pass@1 did not decline, but the repository still lacks a frozen model-bound 128-update GRPO-v2 runner. The next stage is CPU-only runtime freeze; hidden test remains sealed.

## Stage Q guarded GRPO-v2 runtime

The model-bound runtime is frozen and CPU/fake validated. It initializes the v2 policy from only the immutable warm-start adapter, creates a fresh GRPO optimizer/scheduler, consumes the frozen 512-prompt curriculum once over 128 updates, persists primary evidence per update, and writes trusted same-run checkpoints at 32/64/96/128. The same frozen matched dev protocol runs independently at those steps and alone selects a checkpoint. Hidden test remains inaccessible. A real run still requires separate explicit GPU authorization.
