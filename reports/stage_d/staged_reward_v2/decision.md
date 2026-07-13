# Staged shaped reward v2 decision

CPU-only offline replay; no model, generation, trainer, optimizer, GRPO, or PPO was run.

- Policy: `shaped_v2_staged`
- Policy SHA256: `90af0614676279eb8a47636acfdbeaded6d92237d3b16f027d79557057ca0e14`
- Canonical strict parser/verifier status remains FORMAT_ERROR for all 8 historical outputs.
- New nonzero-variance groups: 2/2
- New zero-advantage groups: 0/2

## Group results

| Problem | Old rewards | Staged rewards | Staged variance | Zero advantage |
|---|---|---|---:|---|
| `countdown:train:0` | [0.0, 0.0, 0.0, 0.0] | [0.1, 0.1, 0.15, 0.0] | 0.00296875 | False |
| `countdown:train:1` | [0.0, 0.0, 0.0, 0.0] | [0.1, 0.05, 0.1, 0.05] | 0.00062500 | False |

## Post-smoke intervention disclosure

This reward change occurred after the first v1 GRPO smoke and is a publicly recorded post-smoke intervention. The old run retains its original reward semantics; subsequent PPO/GRPO fair comparisons must both use the frozen new reward version.

该 reward 修改发生在首次 v1 GRPO smoke 之后，是公开记录的 post-smoke intervention。旧 run 保留原 reward 语义，后续 PPO/GRPO 公平比较必须共同使用冻结后的新 reward 版本。

This change does not alter sparse reward, strict format metrics, canonical correctness, the historical run, or any formal/main 1.5B configuration. It only makes the staged training scalar expose deterministic partial protocol progress in both smoke configs.

PPO remains unauthorized. A new real GRPO single-update requires separate authorization.
