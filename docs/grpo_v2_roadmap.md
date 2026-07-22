# GRPO-v2 roadmap

GRPO-v2 is a new, explicitly versioned improvement phase. It does not rewrite portfolio v1 and receives no implicit GPU authorization.

## Goal

Improve canonical mathematical correctness and output reliability while preserving the public comparison's artifact-first standards and GRPO's resource advantage.

## CPU-only design phase

1. Freeze a new experiment identity (`grpo_v2`) rather than editing v1 configs.
2. Use only training and validation evidence for hypotheses; the published held-out test is off-limits for tuning.
3. Pre-register allowed changes, candidate configurations, selection rule, budgets, and stopping rule.
4. Keep the v1 model revision, parser/verifier safety, per-candidate evidence, token accounting, and null/unavailable semantics unless a change is explicitly disclosed.
5. Focus on observed training/validation failure modes: format compliance, parseability, truncation, within-group reward variance, and zero-advantage groups.
6. Run CPU fake paths and tokenizer/prompt audits before requesting a bounded GPU pilot.

## Candidate research directions

- Curriculum or sampling changes based only on train/validation difficulty, never test outcomes.
- A versioned prompt/reward intervention that improves parseable outputs without rewarding incorrect mathematics.
- Length-aware generation controls within a pre-registered completion budget.
- Group construction that preserves within-group learning signal.
- More seeds or a larger validation set before stronger claims.

## Promotion criteria

A candidate must improve pre-registered validation metrics across seeds without identity drift, evidence loss, unsafe checkpoints, or unacceptable format/truncation trade-offs. Only then should a separately authorized fixed final evaluation be considered.
