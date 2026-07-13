"""Evidence contracts for the generation-only prompt diagnostic.

This module is deliberately CPU-only: it contains no torch import and all inputs
and outputs are primitive JSON values.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

CAPABILITY_SCHEMA_VERSION = "prompt-ab-evidence-contract-v1"
CAPABILITY_FIELDS = (
    "paired_artifacts_supported",
    "group_rewards_supported",
    "zero_advantage_groups_supported",
    "allocator_evidence_supported",
    "failure_backup_supported",
    "post_worker_gpu_verification_supported",
    "cross_file_consistency_supported",
)


class EvidenceContractError(RuntimeError):
    """Raised when diagnostic evidence cannot prove the frozen contract."""


def validate_capability_manifest(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("schema_version") != CAPABILITY_SCHEMA_VERSION:
        raise EvidenceContractError("unsupported prompt diagnostic capability schema")
    capabilities = payload.get("capabilities")
    if not isinstance(capabilities, dict) or set(capabilities) != set(CAPABILITY_FIELDS):
        raise EvidenceContractError("prompt diagnostic capability fields mismatch")
    missing = [name for name in CAPABILITY_FIELDS if capabilities.get(name) is not True]
    if missing:
        raise EvidenceContractError(f"required prompt diagnostic capability unavailable: {missing}")
    return {"schema_version": CAPABILITY_SCHEMA_VERSION, "capabilities": dict(capabilities)}


def load_capability_manifest(path: Path) -> dict[str, Any]:
    return validate_capability_manifest(json.loads(path.read_text(encoding="utf-8")))


def _pair_key(row: dict[str, Any]) -> tuple[str, int, int]:
    return (str(row["problem_id"]), int(row["generation_index"]), int(row["seed"]))


def build_paired_comparison(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Join v0/v1 on the stable problem/generation/matched-seed key."""
    indexed: dict[str, dict[tuple[str, int, int], dict[str, Any]]] = {
        "v0": {},
        "v1": {},
    }
    for row in rows:
        condition = row.get("condition")
        if condition not in indexed:
            raise EvidenceContractError("unexpected paired condition")
        key = _pair_key(row)
        if key in indexed[condition]:
            raise EvidenceContractError(f"duplicate paired row: {condition}:{key}")
        if int(row.get("matched_seed", row["seed"])) != key[2]:
            raise EvidenceContractError("matched seed differs from generation seed")
        indexed[condition][key] = row
    if len(indexed["v0"]) != 8 or len(indexed["v1"]) != 8:
        raise EvidenceContractError("paired comparison requires 8 rows per condition")
    if set(indexed["v0"]) != set(indexed["v1"]):
        raise EvidenceContractError("missing or mismatched v0/v1 pair")

    pairs = []
    for pair_index, key in enumerate(sorted(indexed["v0"])):
        v0, v1 = indexed["v0"][key], indexed["v1"][key]
        problem_id, generation_index, seed = key
        pair = {
            "pair_index": pair_index,
            "problem_id": problem_id,
            "generation_index": generation_index,
            "matched_seed": seed,
        }
        for condition, row in (("v0", v0), ("v1", v1)):
            pair.update(
                {
                    f"{condition}_prompt_hash": row["prompt_hash"],
                    f"{condition}_rendered_prompt_sha256": row["rendered_prompt_sha256"],
                    f"{condition}_completion_index": row["completion_index"],
                    f"{condition}_token_count": row["exact_completion_token_count"],
                    f"{condition}_raw_text": row["decoded_raw_text"],
                    f"{condition}_reward_status": row["reward_status"],
                    f"{condition}_scalar_reward": row["scalar_reward"],
                    f"{condition}_format_valid": row["format_valid"],
                    f"{condition}_truncated": row["truncated_at_128"],
                    f"{condition}_expression_valid": row["expression_valid"],
                    f"{condition}_number_usage_valid": row["number_usage_valid"],
                    f"{condition}_final_correct": row["final_answer_correct"],
                }
            )
        pair.update(
            {
                "status_transition": f"{v0['reward_status']} -> {v1['reward_status']}",
                "reward_delta": v1["scalar_reward"] - v0["scalar_reward"],
                "token_count_delta": v1["exact_completion_token_count"]
                - v0["exact_completion_token_count"],
                "format_delta": int(v1["format_valid"]) - int(v0["format_valid"]),
            }
        )
        pairs.append(pair)
    return pairs


def build_group_reward_evidence(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["condition"], row["problem_id"])].append(row)
    evidence: dict[str, Any] = {"conditions": {}}
    for condition in ("v0", "v1"):
        problem_payload: dict[str, Any] = {}
        selected_groups = {key: value for key, value in groups.items() if key[0] == condition}
        if len(selected_groups) != 2:
            raise EvidenceContractError(f"{condition} requires exactly two reward groups")
        for (_, problem_id), group in sorted(selected_groups.items()):
            ordered = sorted(group, key=lambda item: int(item["generation_index"]))
            if [int(item["generation_index"]) for item in ordered] != [0, 1, 2, 3]:
                raise EvidenceContractError("reward group generation order mismatch")
            rewards = [float(item["scalar_reward"]) for item in ordered]
            if not all(math.isfinite(value) for value in rewards):
                raise EvidenceContractError("non-finite group reward")
            variance = statistics.pvariance(rewards)
            zero = len(set(rewards)) == 1
            problem_payload[problem_id] = {
                "generation_indices": [0, 1, 2, 3],
                "scalar_rewards": rewards,
                "reward_statuses": [item["reward_status"] for item in ordered],
                "mean": statistics.mean(rewards),
                "min": min(rewards),
                "max": max(rewards),
                "std": statistics.pstdev(rewards),
                "variance": variance,
                "unique_reward_count": len(set(rewards)),
                "zero_variance": zero,
                "zero_advantage_group": zero,
            }
        zero_count = sum(value["zero_advantage_group"] for value in problem_payload.values())
        evidence["conditions"][condition] = {
            "problems": problem_payload,
            "zero_advantage_group_count": zero_count,
            "nonzero_variance_group_count": len(problem_payload) - zero_count,
        }
    return evidence


def write_paired_csv(path: Path, pairs: list[dict[str, Any]]) -> None:
    if not pairs:
        raise EvidenceContractError("cannot write empty paired comparison")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pairs[0]))
        writer.writeheader()
        writer.writerows(pairs)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _checksum_paths(path: Path) -> set[str]:
    covered = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        digest, separator, name = line.partition("  ")
        if separator != "  " or len(digest) != 64:
            raise EvidenceContractError("malformed checksums entry")
        target = path.parent / name
        if not target.is_file() or hashlib.sha256(target.read_bytes()).hexdigest() != digest:
            raise EvidenceContractError(f"checksum mismatch: {name}")
        covered.add(name)
    return covered


def validate_cross_file_consistency(run_dir: Path, *, require_backup: bool) -> dict[str, Any]:
    """Re-read artifacts and prove that all redundant counters agree."""
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    completions = read_jsonl(run_dir / "completions.jsonl")
    metrics = json.loads((run_dir / "per_condition_metrics.json").read_text(encoding="utf-8"))
    pairs = json.loads((run_dir / "paired_comparison.json").read_text(encoding="utf-8"))
    pair_csv = _read_csv(run_dir / "paired_comparison.csv")
    seeds = json.loads((run_dir / "seed_map.json").read_text(encoding="utf-8"))
    groups = json.loads((run_dir / "per_problem_rewards.json").read_text(encoding="utf-8"))
    if summary["completion_count"] != 16 or len(completions) != 16:
        raise EvidenceContractError("completion total mismatch")
    counts = Counter(row["condition"] for row in completions)
    if counts != {"v0": 8, "v1": 8} or len(pairs) != 8 or len(pair_csv) != 8:
        raise EvidenceContractError("condition or paired total mismatch")
    rebuilt_pairs = build_paired_comparison(completions)
    if pairs != rebuilt_pairs:
        raise EvidenceContractError("paired JSON differs from completion evidence")
    rebuilt_groups = build_group_reward_evidence(completions)
    if groups != rebuilt_groups:
        raise EvidenceContractError("per-group rewards differ from completion evidence")
    token_total = sum(int(row["exact_completion_token_count"]) for row in completions)
    if token_total != summary["budget"]["total_generated_tokens"]:
        raise EvidenceContractError("generated-token total mismatch")
    for condition in ("v0", "v1"):
        selected = [row for row in completions if row["condition"] == condition]
        statuses = dict(Counter(row["reward_status"] for row in selected))
        if statuses != metrics[condition]["reward_status_counts"]:
            raise EvidenceContractError("RewardStatus counts mismatch")
        group = groups["conditions"][condition]
        if group["zero_advantage_group_count"] != metrics[condition]["zero_advantage_group_count"]:
            raise EvidenceContractError("zero-advantage group count mismatch")
        for problem_id, values in group["problems"].items():
            if len(values["scalar_rewards"]) != 4:
                raise EvidenceContractError(f"reward group size mismatch: {problem_id}")
    seed_keys = {
        (row["condition"], row["problem_id"], row["generation_index"], row["seed"]) for row in seeds
    }
    completion_keys = {
        (row["condition"], row["problem_id"], row["generation_index"], row["seed"])
        for row in completions
    }
    if seed_keys != completion_keys:
        raise EvidenceContractError("seed map differs from completion evidence")
    prompt_hashes = json.loads((run_dir / "prompt_hashes.json").read_text(encoding="utf-8"))
    expected_hashes = {row["problem_id"]: row["prompt_hash"] for row in prompt_hashes}
    if any(expected_hashes.get(row["problem_id"]) != row["prompt_hash"] for row in completions):
        raise EvidenceContractError("problem prompt hash mismatch")
    limits = summary["budget"]["limits"]
    if limits["total_completions"] != 16 or limits["total_generated_tokens"] != 2048:
        raise EvidenceContractError("planned budget mismatch")
    if any(summary["safety_counters"].values()):
        raise EvidenceContractError("training/checkpoint side-effect counter is nonzero")
    if manifest.get("completion_count") not in (None, 16):
        raise EvidenceContractError("manifest completion counter mismatch")
    required = {
        "summary.json",
        "run_manifest.json",
        "completions.jsonl",
        "per_condition_metrics.json",
        "per_condition_metrics.csv",
        "paired_comparison.json",
        "paired_comparison.csv",
        "per_problem_rewards.json",
        "seed_map.json",
        "prompt_hashes.json",
        "resource_cost.json",
        "pytorch_allocator.json",
        "post_worker_gpu_verification.json",
    }
    covered = _checksum_paths(run_dir / "checksums.sha256")
    if not required <= covered:
        raise EvidenceContractError(
            f"checksums omit required artifacts: {sorted(required - covered)}"
        )
    if require_backup:
        backup = json.loads((run_dir / "backup_manifest.json").read_text(encoding="utf-8"))
        if backup.get("verified") is not True:
            raise EvidenceContractError("verified backup manifest missing")
    return {
        "valid": True,
        "completion_count": 16,
        "paired_row_count": 8,
        "total_generated_tokens": token_total,
        "reward_status_counts": dict(Counter(row["reward_status"] for row in completions)),
    }


def minimal_failure_record(exc: BaseException, *, phase: str, run_id: str | None) -> dict[str, Any]:
    return {
        "status": "failure",
        "failure_kind": "runtime_failure" if run_id else "preflight_rejected",
        "failure_phase": phase,
        "exception_type": type(exc).__name__,
        "reason": str(exc),
        "run_id": run_id,
    }
