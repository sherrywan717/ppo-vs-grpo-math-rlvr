# Artifact index

This index points to the smallest authoritative public artifact for each claim. Large runtime archives and checkpoints are listed in [release/remote_artifacts.md](release/remote_artifacts.md), not uploaded to GitHub.

## Formal experiment reports

- [Baseline results](reports/formal_1p5b/01_baseline_results.md)
- [PPO seed-42 training and recovered validation](reports/formal_1p5b/03_ppo_training.md)
- [GRPO seed-42 training](reports/formal_1p5b/04_grpo_training.md)
- [Seed-42 training/validation comparison](reports/formal_1p5b/05_seed42_ppo_vs_grpo.md)
- [GRPO seed-123 training](reports/formal_1p5b/06_grpo_seed123_training.md)
- [GRPO seed stability](reports/formal_1p5b/07_grpo_seed_stability.md)
- [PPO seed-123 training](reports/formal_1p5b/08_ppo_seed123_training.md)
- [PPO seed stability](reports/formal_1p5b/09_ppo_seed_stability.md)
- [Four-run training/validation aggregate](reports/formal_1p5b/10_four_run_training_validation_aggregate.md)
- [PPO seed-42 final evaluation](reports/formal_1p5b/11_ppo_seed42_final_evaluation.md)
- [GRPO seed-42 final evaluation](reports/formal_1p5b/12_grpo_seed42_final_evaluation.md)
- [Seed-42 final comparison](reports/formal_1p5b/13_seed42_final_comparison.md)

## Machine-readable evidence

- [Run registry](reports/formal_1p5b/run_registry.csv)
- [Four-run training metrics](reports/formal_1p5b/metrics/four_run_training_metrics.csv)
- [Four-run validation metrics](reports/formal_1p5b/metrics/four_run_validation_metrics.csv)
- [Four-run resource metrics](reports/formal_1p5b/metrics/four_run_resources.csv)
- [Final comparison metrics](reports/formal_1p5b/metrics/seed42_final_comparison_metrics.json)
- [Paired comparison and intervals](reports/formal_1p5b/metrics/seed42_final_paired_summary.json)
- [PPO42 candidate evidence](reports/formal_1p5b/metrics/ppo_seed42_final_per_candidate.csv)
- [GRPO42 candidate evidence](reports/formal_1p5b/metrics/grpo_seed42_final_per_candidate.csv)
- [Baseline per-problem evidence](reports/formal_1p5b/metrics/baseline_per_problem.csv)
- [Reward-group statistics](reports/formal_1p5b/metrics/grpo_reward_group_statistics.csv)
- [Release payload manifest](release/portfolio_v1_manifest.json)
- [Release checksums](release/checksums.sha256)

## Contracts and provenance

- [Formal experiment plan](reports/formal_1p5b/experiment_plan.md)
- [Pass metric contract](reports/formal_1p5b/pass_metric_contract.json)
- [Metric definitions](reports/formal_1p5b/metric_definitions.json)
- [Sample ledger](reports/formal_1p5b/sample_ledger.csv)
- [Checkpoint safety](docs/checkpoint-safety.md)
- [Artifact schema](docs/artifact-schema.md)

## Figures

The portfolio figures are in [reports/formal_1p5b/figures](reports/formal_1p5b/figures/README.md). Run `python scripts/build_portfolio_v1.py` to rebuild the ten `portfolio_*.png` files solely from committed CSV/JSON evidence.
