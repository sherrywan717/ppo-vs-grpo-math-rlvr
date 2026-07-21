# PPO seed-42 final-evaluation error analysis

This analysis is derived from the successful 800-row evidence in
`ppo_final_formal_1p5b_seed42_20260721T022152Z`. The interrupted 429-row outage run is
excluded.

- Canonical statuses: 688 format errors, 14 parse errors, 71 parseable wrong answers,
  and 27 verified passes.
- All 80 truncated candidates are format errors. Among 720 non-truncated candidates,
  608 are format errors. Truncation therefore co-occurs with 11.6% of all format
  errors, but co-occurrence is not treated as proof that truncation alone caused every
  failure.
- GSM8K has no truncation and lower held-out pass@1 (2.5%) than MATH500 (5.0%).
  MATH500 has 20% truncation, concentrated most strongly in Levels 4–5, while its
  positive results are concentrated in Levels 1–2.
- The adapter changes individual outcomes in both directions: pass@1 has 8
  improvements and 9 regressions; pass@4 has 2 improvements and 3 regressions.
- Exact-text diversity does not show mode duplication within the frozen pass@4 groups:
  all 400 candidate texts are unique within their respective four-candidate group.

The complete stratified counts are in
`metrics/ppo_seed42_final_error_analysis.csv`; paired rows and the mechanical case
selection are in `metrics/ppo_seed42_final_paired_comparison.csv` and
`metrics/ppo_seed42_final_representative_cases.csv`. No test example was used to tune
the model, prompt, reward, checkpoint, or analysis protocol.
