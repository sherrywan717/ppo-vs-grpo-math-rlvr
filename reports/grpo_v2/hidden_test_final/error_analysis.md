# Four-model hidden-test error analysis

All comparisons use the same frozen 400 problems, candidate index 0, prompt, seed schedule, parser, and verifier. Case examples are selected mechanically by transition category and lexicographic problem ID; no example was chosen for promotional value.

## Paired transitions

| Comparison | Improved | Regressed | Both correct | Both wrong | Delta | McNemar p |
|---|---:|---:|---:|---:|---:|---:|
| base → old_grpo_v1 | 11 | 0 | 6 | 383 | +2.75 pp | 0.000976562 |
| base → warmstart_only | 4 | 0 | 6 | 390 | +1.00 pp | 0.125 |
| base → selected_grpo_v2 | 38 | 1 | 5 | 356 | +9.25 pp | 1.45519e-10 |
| old_grpo_v1 → selected_grpo_v2 | 31 | 5 | 12 | 352 | +6.50 pp | 1.29135e-05 |
| warmstart_only → selected_grpo_v2 | 37 | 4 | 6 | 353 | +8.25 pp | 1.02584e-07 |


Selected GRPO-v2 improved 38 Base errors and regressed on 1 Base success. Relative to old GRPO-v1 it improved 31 and regressed on 5; relative to warm-start-only it improved 37 and regressed on 4.

## Canonical failure taxonomy on candidate 0

| Role | FORMAT_ERROR | PARSE_ERROR | Parseable wrong | VERIFIED_PASS | Truncated |
|---|---:|---:|---:|---:|---:|
| Base | 369/400 | 3/400 | 22/400 | 6/400 | 43/400 |
| Old GRPO-v1 | 315/400 | 17/400 | 51/400 | 17/400 | 44/400 |
| Warm-start-only | 352/400 | 7/400 | 31/400 | 10/400 | 46/400 |
| Selected GRPO-v2 | 165/400 | 41/400 | 151/400 | 43/400 | 34/400 |


GRPO-v2 reduced strict format failure to 165/400 from 369/400 for Base, while parseable-but-wrong answers became the dominant remaining failure (151/400). This is important: protocol compliance improved substantially, but many parseable outputs still contain incorrect mathematics. Candidate-0 truncation fell to 34/400; truncation therefore cannot explain most remaining wrong answers.

## Dataset and difficulty

Selected GRPO-v2 achieved 27/200 on GSM8K and 16/200 on MATH500. Old GRPO-v1 reached 11/200 and 6/200; the improvement appears in both domains, with the larger absolute count on GSM8K. Selected GRPO-v2 MATH Level correct counts are 1/3, 6/33, 4/43, 3/59, and 2/62 for Levels 1–5. Level 1 is `diagnostic_only_small_n`; three problems cannot characterize the population.

## Truncation and reward-status interpretation

MATH outputs are more truncation-prone than GSM8K because their completions are longer. The final comparison reports status and truncation independently: a truncated completion may also be a format failure, so those categories must not be added as mutually exclusive counts. High shaped reward without `VERIFIED_PASS` remains a diagnostic, never a canonical success.

## Case-study policy

[`case_studies.md`](case_studies.md) lists the lexicographically first five IDs in each preregistered transition category, including v2 regressions and all-wrong cases. This prevents cherry-picking. Full per-problem/candidate evidence remains in the verified runtime archives indexed by [`release/remote_artifacts.md`](../../../release/remote_artifacts.md).

This is a single-seed paired result. Test outcomes are final and cannot trigger tuning or retraining.
