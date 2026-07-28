# GRPO-v2 reproducibility guide

This guide documents provenance; it does not authorize model download, CUDA, training, evaluation, or another hidden-test access.

## Frozen identity

- Model: `Qwen/Qwen2.5-1.5B-Instruct`
- Revision: `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`
- GRPO-v2 config: `configs/grpo_v2/grpo_v2_seed42.json`
- Hidden evaluator config: `configs/grpo_v2/hidden_test_evaluation.json`
- Hidden evaluator raw SHA256: `ff588378a5a6bf1331d08ad95d7311648373eb6e28cae763447d9d67941b7d22`
- Warm-start checkpoint artifact SHA256: `507749d393f38690915a76228b4c53a8b5c8927d40aada9f2768a90334d892f0`
- Selected GRPO-v2 checkpoint: step 96
- Selected checkpoint manifest SHA256: `73bb15a32911f490216be2a80eb0d112be0f79236a6d461fd81fbd0579639246`
- Selected adapter SHA256: `0ebfe5752fb066273692512bd8c3ef23bda4d58786bdfc017aa6aca75fa57080`

## Public CPU-only verification

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

PYTHONPATH=src python scripts/check_env.py
PYTHONPATH=src python scripts/validate_manifests.py
sha256sum -c reports/grpo_v2/hidden_test_final/checksums.sha256
```

The last command validates the Git-safe final comparison artifacts. Full runtime and checkpoint archives live outside Git and are indexed in [`release/remote_artifacts.md`](../release/remote_artifacts.md).

## Data identities

- Train: 512 unique prompts.
- Warm-start: 256-record declared train subset.
- Dev: 128 unique prompts, used only for checkpoint selection.
- Hidden test: 400 unique problems, opened once after selection.
- Shared pass@k set: 100 hidden problems, n=10 candidates per model.

[`data_leakage_audit.json`](../reports/grpo_v2/data_leakage_audit.json) records zero cross-split and v1 overlap. Public execution manifests exclude trusted gold/solution fields.

## Provenance commands

These are the commands that created the recorded artifacts. They are documentation, not permission to rerun them.

```bash
# Warm-start provenance
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src python -m math_rlvr.training.warmstart   --config configs/grpo_v2/warmstart_seed42.json   --run-dir <NEW_RUN_DIR>   --execute --confirm-grpo-v2-warmstart

# GRPO-v2 provenance
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src python -m math_rlvr.training.grpo_v2   --config configs/grpo_v2/grpo_v2_seed42.json   --warmstart-checkpoint <TRUSTED_WARMSTART_CHECKPOINT_16>   --run-dir <NEW_RUN_DIR>   --execute --confirm-grpo-v2

# Hidden evaluator provenance; role is one of the four frozen roles.
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src python -m math_rlvr.evaluation.grpo_v2_hidden   --config configs/grpo_v2/hidden_test_evaluation.json   --role <base|old_grpo_v1|warmstart_only|selected_grpo_v2>   --checkpoint <EXACT_ROLE_CHECKPOINT_IF_REQUIRED>   --run-dir <NEW_RUN_DIR>   --execute --confirm-grpo-v2-hidden
```

The project has already consumed its hidden-test authorization. Do not execute these commands against the current hidden test again.

## Metric reconstruction

Candidate-0 accuracy is binary accuracy over all 400 hidden problems. Shared-pool pass@k is computed independently per problem from n=10 and correct count c:

```text
pass_hat(k) = 1 - C(10-c, k) / C(10, k),  k in {1,4,10}
```

Aggregate pass@k is the arithmetic mean of problem-level estimates. Source files:

- [`four_model_summary.csv`](../reports/grpo_v2/hidden_test_final/four_model_summary.csv)
- [`paired_candidate0_comparisons.csv`](../reports/grpo_v2/hidden_test_final/paired_candidate0_comparisons.csv)
- [`paired_pass_k_comparisons.csv`](../reports/grpo_v2/hidden_test_final/paired_pass_k_comparisons.csv)
- [`math_level_results.csv`](../reports/grpo_v2/hidden_test_final/math_level_results.csv)

## Figure reconstruction contract

[`figure_sources.json`](../reports/grpo_v2/portfolio/figure_sources.json) maps each published image to committed CSV/JSON. Training figures are additionally bound by the run's [`figure_sources.json`](../reports/grpo_v2/grpo_v2_training/grpo_v2_seed42_20260726T044303Z/figure_sources.json). Published images are derived artifacts; CSV/JSON is authoritative.

## Checkpoint and resume safety

Only project-created adapter-only checkpoints with matching run/config/model/data identities are trusted. The checkpoint inventory includes optimizer, scheduler, RNG, counters, curriculum cursor, prefix evidence, and SHA256. Evaluation loads only the policy adapter—not optimizer, scheduler, or RNG state. Base weights and full checkpoints are never committed.

## Environment and unavailable telemetry

Runs use BF16, local-only snapshots, `HF_HUB_OFFLINE=1`, and `TRANSFORMERS_OFFLINE=1`. Optional metrics unavailable from the frozen TRL runtime remain JSON `null` with `available=false` and a reason. Checkpoint-dev-only GPU cost and isolated IPC-idle cost were not separately measurable; they are unavailable rather than zero.
