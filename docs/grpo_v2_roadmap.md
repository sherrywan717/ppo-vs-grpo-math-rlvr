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

## Stage R first attempt

`grpo_v2_seed42_20260726T030733Z` stopped before training when immutable curriculum position 83 rendered to 914 prompt tokens under the frozen GRPO cap 832. No update, completion, checkpoint, dev evaluation, or hidden-test access occurred, and the attempt was not retried. Warm-start and matched dev remain valid. A separately authorized CPU-only capacity reconciliation is required before any new GRPO-v2 GPU attempt.

## Stage R.1 capacity reconciliation

The pinned tokenizer and exact runtime renderer audited all 512 training and 128 dev prompts. The actual maximum is 918 tokens, so the preregistered rule freezes prompt cap 928; completion remains 256 and the explicit sequence ceiling is 1,184. Old-cap overflows were two; new-cap overflows and truncations are zero. The execute path now completes this exact audit before CUDA, model/adapter load, Trainer/optimizer construction or generation. Config SHA is `ce3883b0326492b9109963e8d95496936aa3b3b8670cb9d3b4e9346f65c8cc93` and runtime registry canonical SHA is `fad035928e6fdc285ec290d295f4d481700c04ac7f5639f41d3e3ac8a0451beb`. No data, curriculum, hidden test, warm-start state, model, prompt, reward/parser/verifier, LoRA, sampling or budget changed. A fresh GRPO-v2 run still requires separate GPU authorization.

## Stage R.2 initialization result

Fresh run `grpo_v2_seed42_20260726T034649Z` passed the reconciled 512+128 prompt
preflight, but failed before training when the post-constructor fresh-optimizer audit
accessed `.state` on an optimizer that had not yet been created. All training,
completion, checkpoint, dev and hidden-test counters are zero. The run is immutable
and excluded. The next gate is a narrow CPU-only optimizer-lifecycle repair; no GPU
retry or hidden evaluation is authorized.

## Stage R.4 result and Stage S.1 evaluator freeze

GRPO-v2 run `grpo_v2_seed42_20260726T044303Z` completed 128 updates and 2,048 training completions; frozen dev selection chose checkpoint-96 at 33/128. Training is immutable and hidden-test access remains zero. Stage S.1 froze one shared evaluator for Base, old GRPO-v1 checkpoint-32, warmstart checkpoint-16, and selected GRPO-v2 checkpoint-96. Each model uses 400 candidate-0 keys plus the unchanged shared 100-problem n=10 batch, or 1,300 completions. File-backed primary evidence replaces large IPC payloads. A separately authorized one-time four-model hidden evaluation is next; its results may not change training or checkpoint selection.

## Stage S.2 Base finalization failure

The frozen Base command produced the complete 1,300-row hidden ledger, but the evaluator reused a dev-only 128-row aggregate and stopped during metric finalization. Base evidence and its verified backup are immutable; Base must not run again. The other three model commands were not executed. The only next gate is CPU-only repair of this row-count assumption and report recovery from existing Base evidence, followed by separate authorization for the remaining roles.

## Stage S.3 and S.4 final hidden comparison

Stage S.3 recovered Base metrics from the immutable 1,300-row ledger without model
loading or regeneration. Stage S.4 then evaluated old GRPO-v1, warm-start-only, and
selected GRPO-v2 checkpoint-96 exactly once each. The final ledger is 5,200
completions with matched problem/candidate/seed keys.

Candidate-0 accuracy is Base 6/400, old GRPO-v1 17/400, warm-start-only 10/400, and
selected GRPO-v2 43/400. Shared-pool unbiased pass@1/pass@4/pass@10 for selected
GRPO-v2 is 14.40%/31.14%/42.00%. Selected GRPO-v2 improves over Base by +9.25 pp
(paired bootstrap 95% CI +6.50 to +12.25 pp) and over warm-start-only by +8.25 pp
(+5.25 to +11.25 pp), supporting incremental RLVR benefit only for this single
frozen seed/protocol. Hidden-test results are terminal and cannot trigger tuning,
checkpoint reselection, or another run.
## Stage T portfolio release

GRPO-v2 is scientifically closed. The final portfolio publishes the complete 5,200
completion comparison, training/dev evidence, deterministic error analysis, resource
ledger, figure sources, checksums, and remote archive index. No additional training or
test access belongs to this roadmap. Any successor must begin with a new preregistered
protocol and a new untouched hidden-test identity.
