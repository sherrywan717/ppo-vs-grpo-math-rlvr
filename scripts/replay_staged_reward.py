"""CPU-only replay of the immutable v1 GRPO completions under staged reward v2."""

from __future__ import annotations

import csv
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from math_rlvr.config import load_config, resolve_training_config
from math_rlvr.dataset import load_manifest
from math_rlvr.rewards.staged import STAGED_REWARD_POLICY
from math_rlvr.verifier import MathVerifier

INPUT = Path(
    "reports/runs/grpo_single_update_qwen25_05b_20260713T112100Z/completions.jsonl"
)
OUTPUT = Path("reports/stage_d/staged_reward_v2")
MANIFEST = Path("/root/autodl-tmp/datasets/math_rlvr/manifests/countdown_train.json")
EXPECTED_INPUT_SHA256 = "1b4e213df6d69aa7cc1663deeb92ca27850346a8bde552360a78448ce9ff8d02"
OLD_GRPO_YAML_SHA256 = "5df5d72f71ada14a6ce903990b1b21bbd9d682ba8a05b1f77a91bc974c3872e0"
OLD_PPO_YAML_SHA256 = "b888b12fb56fe356633b2d04f2c9713bb8d02c13be66fe349f60b5d40cbc1ee3"
INTERVENTION = (
    "This reward change occurred after the first v1 GRPO smoke and is a publicly "
    "recorded post-smoke intervention. The old run retains its original reward "
    "semantics; subsequent PPO/GRPO fair comparisons must both use the frozen new "
    "reward version."
)
INTERVENTION_ZH = (
    "该 reward 修改发生在首次 v1 GRPO smoke 之后，是公开记录的 post-smoke intervention。"
    "旧 run 保留原 reward 语义，后续 PPO/GRPO 公平比较必须共同使用冻结后的新 reward 版本。"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def replay() -> dict:
    if sha256(INPUT) != EXPECTED_INPUT_SHA256:
        raise RuntimeError("immutable historical completions hash mismatch")
    records = [json.loads(line) for line in INPUT.read_text().splitlines()]
    if len(records) != 8:
        raise RuntimeError("offline replay requires exactly eight historical completions")

    problems = {problem.problem_id: problem for problem in load_manifest(MANIFEST)}
    verifier = MathVerifier()
    replay_rows = []
    for record in records:
        problem = problems[record["problem_id"]]
        text = record["raw_completion"]
        def bound(candidate, item=problem):
            return verifier(item, candidate)

        evaluation = STAGED_REWARD_POLICY.evaluate(text, bound)
        evidence = evaluation.to_dict()
        if evidence["canonical_status"] != record["reward_status"]:
            raise RuntimeError("offline canonical status differs from immutable runtime evidence")
        replay_rows.append(
            {
                "problem_id": record["problem_id"],
                "generation_index": record["generation_index"],
                "completion_index": record["completion_index"],
                "raw_completion": text,
                "old_canonical_status": record["reward_status"],
                "new_canonical_status": evidence["canonical_status"],
                "old_scalar_reward": float(record["scalar_reward"]),
                **evidence,
            }
        )

    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in replay_rows:
        grouped[row["problem_id"]].append(row)

    group_rows = []
    for problem_id, items in sorted(grouped.items()):
        ordered = sorted(items, key=lambda row: row["generation_index"])
        old_rewards = [row["old_scalar_reward"] for row in ordered]
        new_rewards = [row["scalar_reward"] for row in ordered]
        group_rows.append(
            {
                "problem_id": problem_id,
                "old_rewards": old_rewards,
                "new_rewards": new_rewards,
                "old_mean": round(statistics.fmean(old_rewards), 10),
                "new_mean": round(statistics.fmean(new_rewards), 10),
                "old_variance": round(statistics.pvariance(old_rewards), 10),
                "new_variance": round(statistics.pvariance(new_rewards), 10),
                "old_zero_advantage": len(set(old_rewards)) == 1,
                "new_zero_advantage": len(set(new_rewards)) == 1,
            }
        )

    nonzero_groups = sum(row["new_variance"] > 0 for row in group_rows)
    if nonzero_groups == 0:
        raise RuntimeError("staged reward produced no within-problem reward variance")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    figures = OUTPUT / "figures"
    figures.mkdir(exist_ok=True)
    (OUTPUT / "offline_replay.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in replay_rows)
    )
    write_json(OUTPUT / "paired_reward_comparison.json", replay_rows)
    paired_fields = [
        "problem_id",
        "generation_index",
        "completion_index",
        "old_canonical_status",
        "new_canonical_status",
        "old_scalar_reward",
        "scalar_reward",
        "reward_policy_version",
        "reward_policy_sha256",
        "answer_block_component",
        "strict_protocol_component",
        "valid_expression_component",
        "exact_number_usage_component",
        "correctness_component",
        "verifier_detail",
        "raw_completion",
    ]
    write_csv(
        OUTPUT / "paired_reward_comparison.csv",
        [{key: row[key] for key in paired_fields} for row in replay_rows],
        paired_fields,
    )
    group_fields = [
        "problem_id",
        "old_rewards",
        "new_rewards",
        "old_mean",
        "new_mean",
        "old_variance",
        "new_variance",
        "old_zero_advantage",
        "new_zero_advantage",
    ]
    write_csv(OUTPUT / "group_statistics.csv", group_rows, group_fields)
    write_json(OUTPUT / "group_statistics.json", group_rows)

    labels = [str(row["completion_index"]) for row in replay_rows]
    old = [row["old_scalar_reward"] for row in replay_rows]
    new = [row["scalar_reward"] for row in replay_rows]
    x = list(range(len(labels)))
    plt.figure(figsize=(9, 4.8))
    plt.bar([value - 0.18 for value in x], old, width=0.36, label="old shaped v1")
    plt.bar([value + 0.18 for value in x], new, width=0.36, label="staged v2")
    plt.xticks(x, labels)
    plt.xlabel("Completion index")
    plt.ylabel("Scalar reward")
    plt.title("Immutable v1 GRPO completions: old vs staged reward")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures / "paired_reward_comparison.png", dpi=150)
    plt.close()

    plt.figure(figsize=(7, 4.5))
    group_labels = [row["problem_id"].rsplit(":", 1)[-1] for row in group_rows]
    plt.bar(
        [value - 0.18 for value in range(len(group_rows))],
        [row["old_variance"] for row in group_rows],
        width=0.36,
        label="old",
    )
    plt.bar(
        [value + 0.18 for value in range(len(group_rows))],
        [row["new_variance"] for row in group_rows],
        width=0.36,
        label="staged v2",
    )
    plt.xticks(range(len(group_rows)), group_labels)
    plt.ylabel("Population reward variance")
    plt.title("Within-problem reward variance")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures / "group_reward_variance.png", dpi=150)
    plt.close()

    components = [
        "answer_block_component",
        "strict_protocol_component",
        "valid_expression_component",
        "exact_number_usage_component",
        "correctness_component",
    ]
    matrix = [[row[name] for name in components] for row in replay_rows]
    plt.figure(figsize=(10, 5))
    plt.imshow(matrix, aspect="auto", cmap="YlGn", vmin=0, vmax=0.8)
    plt.yticks(range(8), labels)
    plt.xticks(
        range(len(components)),
        [name.removesuffix("_component") for name in components],
        rotation=25,
    )
    plt.xlabel("Reward component")
    plt.ylabel("Completion index")
    plt.title("Staged reward component audit")
    plt.colorbar(label="Component value")
    plt.tight_layout()
    plt.savefig(figures / "reward_components.png", dpi=150)
    plt.close()

    grpo = resolve_training_config(load_config("configs/smoke/grpo.yaml"))
    ppo = resolve_training_config(load_config("configs/smoke/ppo.yaml"))
    if (
        grpo["reward_policy_sha256"] != ppo["reward_policy_sha256"]
        or grpo["reward_component_weights"] != ppo["reward_component_weights"]
    ):
        raise RuntimeError("PPO/GRPO staged reward identity mismatch")

    decision = {
        "schema_version": "math_rlvr.staged_reward_v2_decision.v1",
        "status": "approved_for_next_smoke_review",
        "training_executed": False,
        "historical_run_modified": False,
        "historical_run_id": "grpo_single_update_qwen25_05b_20260713T112100Z",
        "historical_reward_policy": "shaped_v1_legacy",
        "reward_policy_version": grpo["reward_policy_version"],
        "reward_component_weights": grpo["reward_component_weights"],
        "reward_policy_sha256": grpo["reward_policy_sha256"],
        "prompt_version": grpo["prompt_version"],
        "prompt_sha256": grpo["prompt_sha256"],
        "old_yaml_sha256": {
            "grpo": OLD_GRPO_YAML_SHA256,
            "ppo": OLD_PPO_YAML_SHA256,
        },
        "new_yaml_sha256": {
            "grpo": sha256(Path("configs/smoke/grpo.yaml")),
            "ppo": sha256(Path("configs/smoke/ppo.yaml")),
        },
        "groups": group_rows,
        "new_nonzero_variance_group_count": nonzero_groups,
        "new_zero_advantage_group_count": sum(
            row["new_zero_advantage"] for row in group_rows
        ),
        "canonical_status_counts": {"format_error": 8},
        "protected_hashes": {
            "main_grpo_yaml": sha256(Path("configs/main/grpo.yaml")),
            "main_ppo_yaml": sha256(Path("configs/main/ppo.yaml")),
            "prompt_module": sha256(Path("src/math_rlvr/prompt.py")),
            "historical_completions": sha256(INPUT),
            "historical_summary": sha256(
                Path(
                    "reports/runs/grpo_single_update_qwen25_05b_20260713T112100Z/"
                    "summary.json"
                )
            ),
        },
        "intervention_disclosure": INTERVENTION,
        "intervention_disclosure_zh": INTERVENTION_ZH,
        "ppo_authorized": False,
        "gpu_grpo_authorized": False,
    }
    write_json(OUTPUT / "decision.json", decision)

    lines = [
        "# Staged shaped reward v2 decision",
        "",
        "CPU-only offline replay; no model, generation, trainer, optimizer, GRPO, or PPO was run.",
        "",
        f"- Policy: `{decision['reward_policy_version']}`",
        f"- Policy SHA256: `{decision['reward_policy_sha256']}`",
        "- Canonical strict parser/verifier status remains FORMAT_ERROR for all 8 "
        "historical outputs.",
        f"- New nonzero-variance groups: {nonzero_groups}/2",
        f"- New zero-advantage groups: {decision['new_zero_advantage_group_count']}/2",
        "",
        "## Group results",
        "",
        "| Problem | Old rewards | Staged rewards | Staged variance | Zero advantage |",
        "|---|---|---|---:|---|",
    ]
    for row in group_rows:
        lines.append(
            f"| `{row['problem_id']}` | {row['old_rewards']} | {row['new_rewards']} | "
            f"{row['new_variance']:.8f} | {row['new_zero_advantage']} |"
        )
    lines += [
        "",
        "## Post-smoke intervention disclosure",
        "",
        INTERVENTION,
        "",
        INTERVENTION_ZH,
        "",
        "This change does not alter sparse reward, strict format metrics, canonical correctness, "
        "the historical run, or any formal/main 1.5B configuration. It only makes the staged "
        "training scalar expose deterministic partial protocol progress in both smoke configs.",
        "",
        "PPO remains unauthorized. A new real GRPO single-update requires separate authorization.",
        "",
    ]
    (OUTPUT / "decision.md").write_text("\n".join(lines))
    return decision


if __name__ == "__main__":
    print(json.dumps(replay(), indent=2))
