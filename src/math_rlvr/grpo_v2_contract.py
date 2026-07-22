"""CPU-only contracts for the pre-registered GRPO-v2 experiment."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

CHECKPOINT_STEPS = (32, 64, 96, 128)


def canonical_json_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def selection_key(
    *,
    dataset_revision: str,
    source_split: str,
    source_problem_id: str,
    namespace: str,
    seed: int = 42,
) -> str:
    """Gold-independent stable selection key."""
    return canonical_json_sha256(
        {
            "dataset_revision": dataset_revision,
            "source_split": source_split,
            "source_problem_id": source_problem_id,
            "selection_namespace": namespace,
            "selection_seed": seed,
        }
    )


def select_checkpoint(rows: list[dict]) -> int:
    """Apply the frozen dev-only lexicographic selection rule."""
    expected = set(CHECKPOINT_STEPS)
    if {int(row["checkpoint_step"]) for row in rows} != expected:
        raise ValueError("checkpoint candidates must be exactly 32/64/96/128")
    for row in rows:
        if row.get("evaluation_split") != "dev_v2":
            raise ValueError("checkpoint selection may use dev_v2 only")
    best = max(
        rows,
        key=lambda row: (
            float(row["canonical_pass_at_1"]),
            float(row["parseable_rate"]),
            float(row["format_rate"]),
            -float(row["truncation_rate"]),
            -int(row["checkpoint_step"]),
        ),
    )
    return int(best["checkpoint_step"])


def wilson_interval(
    numerator: int, denominator: int, z: float = 1.959963984540054
) -> tuple[float, float]:
    if denominator <= 0:
        raise ValueError("denominator must be positive")
    p = numerator / denominator
    scale = 1 + z * z / denominator
    centre = (p + z * z / (2 * denominator)) / scale
    radius = z * math.sqrt(p * (1 - p) / denominator + z * z / (4 * denominator**2)) / scale
    return centre - radius, centre + radius


def validate_nested_success(candidate_correctness: list[bool]) -> dict[str, bool]:
    """Validate one problem's shared candidate-0 nested k=1/4/10 pool."""
    if len(candidate_correctness) != 10 or any(
        type(value) is not bool for value in candidate_correctness
    ):
        raise ValueError("nested pass@10 requires exactly ten boolean candidates")
    success = {
        "success_at_1": candidate_correctness[0],
        "success_at_4": any(candidate_correctness[:4]),
        "success_at_10": any(candidate_correctness),
    }
    if not (success["success_at_1"] <= success["success_at_4"] <= success["success_at_10"]):
        raise ValueError("nested pass@k monotonicity failed")
    return success


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def validate_public_manifest(rows: list[dict]) -> None:
    forbidden = {"gold", "gold_answer", "solution", "answer", "target", "target_text"}
    ids: set[str] = set()
    hashes: set[str] = set()
    for row in rows:
        if forbidden & set(row):
            raise ValueError("public execution manifest leaks trusted target fields")
        if row["problem_id"] in ids or row["content_hash"] in hashes:
            raise ValueError("duplicate problem identity")
        ids.add(row["problem_id"])
        hashes.add(row["content_hash"])


def validate_contract_tree(root: Path) -> dict:
    manifest_dir = root / "configs/grpo_v2/manifests"
    manifests = {
        name: read_jsonl(manifest_dir / f"{name}.jsonl")
        for name in ("train_v2", "warmstart_v2", "dev_v2", "test_v2_hidden")
    }
    for rows in manifests.values():
        validate_public_manifest(rows)
    train_ids = {row["problem_id"] for row in manifests["train_v2"]}
    warm_ids = {row["problem_id"] for row in manifests["warmstart_v2"]}
    dev_ids = {row["problem_id"] for row in manifests["dev_v2"]}
    test_ids = {row["problem_id"] for row in manifests["test_v2_hidden"]}
    if len(train_ids) != 512 or len(warm_ids) != 256 or len(dev_ids) != 128 or len(test_ids) != 400:
        raise ValueError("manifest count contract failed")
    if (
        not warm_ids < train_ids
        or train_ids & dev_ids
        or train_ids & test_ids
        or dev_ids & test_ids
    ):
        raise ValueError("split/subset contract failed")
    nested = json.loads((manifest_dir / "pass4_nested_subset.json").read_text())
    nested_ids = {row["problem_id"] for row in nested["problems"]}
    if len(nested_ids) != 100 or not nested_ids < test_ids:
        raise ValueError("nested pass@4 subset contract failed")
    curriculum = json.loads((root / "configs/grpo_v2/curriculum.json").read_text())
    positions = curriculum["positions"]
    if len(positions) != 512 or {p["problem_id"] for p in positions} != train_ids:
        raise ValueError("curriculum coverage failed")
    if [p["position"] for p in positions] != list(range(1, 513)):
        raise ValueError("curriculum order failed")
    return {"status": "passed", "counts": {k: len(v) for k, v in manifests.items()}, "nested": 100}
