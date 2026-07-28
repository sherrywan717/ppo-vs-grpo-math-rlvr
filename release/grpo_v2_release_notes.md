# GRPO-v2 Math RLVR Portfolio

Release tag: `v0.2.0-grpo-v2`

## Highlights

- Completed a 256-example completion-only warm-start and a 128-update GRPO-v2 run over 512 unique prompts.
- Dev-only selection chose checkpoint-96 at 33/128 canonical pass@1.
- Completed the frozen four-model hidden comparison: 5,200/5,200 completions and 606,487 generated tokens.
- Candidate-0 accuracy: Base 6/400 (1.50%), old GRPO-v1 17/400 (4.25%), warm-start-only 10/400 (2.50%), selected GRPO-v2 43/400 (10.75%).
- Selected GRPO-v2 shared-pool unbiased pass@1/@4/@10: 14.40% / 31.14% / 42.00%.
- Base → GRPO-v2: +9.25 pp, 38 improvements / 1 regression, paired bootstrap 95% CI [+6.50,+12.25] pp, McNemar p=1.46e-10.
- Old GRPO-v1 → GRPO-v2: +6.50 pp, CI [+3.75,+9.50] pp.
- Warm-start-only → GRPO-v2: +8.25 pp, CI [+5.25,+11.25] pp.

## Evidence

- [Final comparison](../reports/grpo_v2/hidden_test_final/final_comparison.md)
- [Training report](../reports/grpo_v2/grpo_v2_training/grpo_v2_seed42_20260726T044303Z/report.md)
- [Machine-readable aggregate](../reports/grpo_v2/hidden_test_final/aggregate_summary.json)
- [Error analysis](../reports/grpo_v2/hidden_test_final/error_analysis.md)
- [Cost ledger](../reports/grpo_v2/portfolio/cost_ledger.json)
- [Remote artifact index](remote_artifacts.md)

## Scientific boundaries

This is one seed on one 1.5B model. MATH Level 1 has n=3 and is diagnostic only. Candidate-0 accuracy and shared-pool pass@k use different problem universes. The hidden test was opened once after dev selection and cannot be used for more tuning, checkpoint selection, or retraining. The release does not claim universal GRPO superiority.

## Artifact policy

GitHub contains source, configs/manifests, reports, CSV/JSON, figures, small samples, and checksum inventories. Base weights, full checkpoints, optimizer/scheduler/RNG state, credentials, caches, and full run archives remain outside GitHub.
## Preservation patch

The disaster-recovery patch adds the audited
[200-question Chinese interview guide](../docs/INTERVIEW_200_QA_ZH.md),
preservation inventory, public adapter/evidence bundles, and independent download
verification. The original `v0.2.0-grpo-v2` tag remains immutable; archival assets
are published under `v0.2.1-grpo-v2-archive`.
