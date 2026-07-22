# Interview guide

## Thirty-second summary

I built an artifact-first RLVR system to compare PPO and GRPO fairly on Qwen2.5-1.5B. I matched policy-side inputs and budgets, separated algorithm-intrinsic value-model costs, persisted per-update/per-candidate evidence, and evaluated fixed checkpoint-32 on held-out GSM8K/MATH500. At seed 42, GRPO reached 7.0% pass@1 versus 4.0% Base and 3.75% PPO, while using roughly one-fifth PPO's peak VRAM during training/validation.

## Design questions

**Why not compare trainer steps?** PPO and GRPO have different inner loops. I matched 32 optimizer/global updates, 512 completions, and a token cap, then plotted against actual generated tokens.

**How was leakage prevented?** Train, validation, and test have separate manifests and ledgers. Checkpoint-32 was fixed in advance; test never selected a checkpoint or changed a hyperparameter.

**Why use shaped reward and canonical metrics?** Shaping supplies learnable intermediate signal. Canonical verifier status remains the scientific correctness measure, preventing reward improvement from being misrepresented as mathematical success.

**Why is PPO memory so high?** This PPO implementation loads a distinct value backbone/adapter/head in addition to the policy/reference roles. GRPO avoids the value model and uses within-group relative advantage.

**What failed?** Early runs exposed serialization, prompt-capacity, optional telemetry, evidence-flush, and deferred-validation cadence bugs. Failed attempts remain immutable and excluded. The fixes targeted correctness/evidence boundaries, not the scientific result.

**What can you claim?** A paired seed-42 held-out improvement for GRPO and consistent two-seed validation direction under this small contract. I cannot claim universal superiority, statistical significance across tasks/models, or seed-123 final performance.

## Deep-dive anchors

- Reward/parser/verifier: [reward_and_verifier.md](reward_and_verifier.md)
- Checkpoint/resume safety: [checkpoint-safety.md](checkpoint-safety.md)
- Engineering failures: [engineering_postmortem.md](engineering_postmortem.md)
- Machine-readable result: [seed42_final_paired_summary.json](../reports/formal_1p5b/metrics/seed42_final_paired_summary.json)
