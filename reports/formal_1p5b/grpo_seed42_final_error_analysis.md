# GRPO seed-42 final-evaluation error analysis

This analysis is derived from all 800 immutable completion rows in
`grpo_final_formal_1p5b_seed42_20260721T034104Z`.

- Canonical statuses: 604 format errors, 34 parse errors, 113 parseable wrong answers,
  and 49 verified passes.
- All 64 truncated candidates are format errors. The other 540 format errors are not
  truncated, so truncation is associated with only 10.6% of format errors and cannot
  explain the majority of formatting failures.
- GSM8K has one truncated candidate and pass@1 6.0%; MATH500 has 63 truncated
  candidates and pass@1 8.0%. MATH Level 5 remains 0/40 pass@1 despite a lower
  truncation rate than Level 4, so truncation alone does not explain difficulty.
- The flat domain-valid component is positive for 291 candidates, while only 162 are
  canonical-parseable. This gap is expected because the shaping probe extracts an
  answer separately from the strict full-completion parser.
- There are 113 partial-reward candidates at or above 0.20 that are canonically wrong.
  They are retained for reward-hacking review, but this count alone does not prove
  exploitation.
- All 400 pass@4 candidate texts are unique within their four-candidate groups.

The full strata are in `metrics/grpo_seed42_final_error_analysis.csv`; paired
Base/PPO/GRPO transitions and mechanically selected examples are in
`metrics/seed42_final_paired_comparison.csv` and
`metrics/grpo_seed42_final_representative_cases.csv`. No example was used for tuning.
