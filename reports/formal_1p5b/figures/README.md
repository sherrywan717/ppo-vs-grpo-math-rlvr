# Formal figure directory

Stage E generated no scientific figures because it performed no model execution.
Stage G.2 baseline figures are regenerated exclusively from the committed baseline
CSV artifacts with `scripts/aggregate_formal_baseline.py`; their captions and relative
links are in `../01_baseline_results.md`. Future training figures use
`scripts/plot_formal_results.py`. Planned outputs are reward/update, canonical
validation pass/update, format/valid-answer, loss, KL/entropy/grad norm, completion
length, verifier status, resources/cost, per-seed PPO/GRPO, baseline/post pass@k,
domain and MATH500 level results, and paired confidence intervals.
