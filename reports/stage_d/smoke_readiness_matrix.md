# Stage D smoke readiness matrix

Assessment date: 2026-07-14 UTC. This audit uses existing Git-safe reports and run manifests only; it does not replay rewards or execute either trainer.

| Field | PPO `ppo_single_update_qwen25_05b_20260714T051538Z` | GRPO `grpo_single_update_qwen25_05b_20260713T122258Z` | Readiness |
|---|---|---|---|
| Model | Qwen 0.5B, revision `7ae557604adf67be50417f59c2c2f167def9a775`, BF16, local-only | Same | Matched |
| Prompt | `prompt_v1_strict_concise`, SHA `6842002…ecd7`, renderer v1 | Same | Matched |
| Reward | `shaped_v2_staged`, SHA `90af0614…e14` | Same | Matched |
| Parser/verifier | Canonical strict contract through the shared reward adapter | Same | Contract matched; neither historical report has a standalone parser/verifier version/SHA field |
| Data | Countdown; IDs 0, 2, 3, 1; four unique prompts × one response | Countdown; IDs 0, 1; two unique prompts × four responses | Task matched, sampling allocation unmatched |
| Sampling | temperature 0.8, top-p 0.95, max completion length 128 | Same | Matched |
| Policy LoRA | r16, alpha 32, dropout 0, q/k/v/o; 2,162,688 trainables | Same | Matched |
| Completion/token budget | 4 / 512; actual 4 / 141 | 8 / 1,024; actual 8 / 276 | Unmatched |
| Steps | one PPO epoch, minibatch, update, optimizer step, global step | one iteration, four microsteps, one optimizer/global step | Outer update matched; inner work differs |
| Seed | 42 | 42 | Matched |
| Reward variance | population variance 0.00046875 | group variances 0.00296875 and 0.000625; zero-advantage groups 0 | Both nonzero; aggregation units differ |
| Completion status | 2 format errors, 2 invalid expressions | 8 format errors | Observed outcomes unmatched |
| Checkpoint | role-separated adapter/head only; no base/optimizer weights | one safe adapter checkpoint; no base/optimizer weights | Both pass |
| Real update | completed one optimizer/global/update step; accepted as `execution_success/nonessential_telemetry_warning` | completed one optimizer/global step with credible finalized artifacts | Both pass |

## Decision

A fully current GRPO technical-smoke counterpart already exists: `grpo_single_update_qwen25_05b_20260713T122258Z`. It uses `prompt_v1_strict_concise`, `shaped_v2_staged`, the fixed Qwen 0.5B revision, the shared canonical parser/verifier semantics, completed a real update with credible artifacts, and produced nonzero within-group reward variance.

Stage D technical smoke is therefore complete. This conclusion means both guarded execution paths, reward integration, counters, and checkpoint safety have been exercised. It does **not** establish task learning or PPO superiority over GRPO.

The two runs are not an algorithm-effect comparison: PPO sampled four prompts once each, whereas GRPO sampled two prompts four times each; completion/token budgets and actual totals differ; reward variance is aggregated at different units. The old v0 and all-zero-reward GRPO runs are historical diagnostics and are not used as the current PPO comparator.

## Next stage: plan only

Before any further GPU work, freeze a paired Qwen 0.5B pilot around one ordered prompt manifest, the exact current model/prompt/reward identities, identical sampling and policy LoRA, explicit parser/verifier hashes, and predefined seeds. A proposed unit is four shared prompts × four responses per prompt × 16 completions per algorithm per seed, with a 2,048 generated-token cap and three seeds. PPO prompt repetition must be specified in advance so actual completions—not trainer steps—match GRPO.

Linear scaling from the observed smoke GPU-hours gives a rough total of 0.0491–0.0982 GPU-hours, or ¥0.44–¥0.88 at ¥8.88/GPU-hour, including a 2× planning ceiling. This is approximate because model setup dominates these very short runs.

Planned config paths are `configs/pilot/ppo_0p5b_matched.yaml` and `configs/pilot/grpo_0p5b_matched.yaml`. They do not exist yet, so the following are command templates, not executable authorization:

```bash
PYTHONPATH=src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m math_rlvr.training.ppo --config configs/pilot/ppo_0p5b_matched.yaml --execute --confirm-single-update
PYTHONPATH=src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m math_rlvr.training.grpo --config configs/pilot/grpo_0p5b_matched.yaml --execute --confirm-single-update
```

No GPU run, model load, completion generation, or trainer execution was performed for this audit.
