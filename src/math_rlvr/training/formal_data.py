"""Frozen, offline-verifiable data contract for the formal 1.5B experiment."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from math_rlvr.dataset import MathProblem, content_hash, load_manifest

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REGISTRY_PATH = REPOSITORY_ROOT / "configs/formal_1p5b/data_registry.json"


class FormalDataContractError(RuntimeError):
    """Raised when frozen formal-data evidence drifts or is incomplete."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ordered_values_sha256(values: list[str]) -> str:
    return canonical_sha256(values)


def registry_sha256(registry: dict[str, Any]) -> str:
    body = dict(registry)
    body.pop("registry_sha256", None)
    return canonical_sha256(body)


def _distribution(problems: list[MathProblem], attribute: str) -> dict[str, int]:
    return dict(sorted(Counter(str(getattr(problem, attribute)) for problem in problems).items()))


def _source_distribution(problems: list[MathProblem]) -> dict[str, int]:
    return dict(sorted(Counter(problem.source for problem in problems).items()))


def load_formal_data_registry(path: Path = DEFAULT_REGISTRY_PATH) -> dict[str, Any]:
    registry = json.loads(path.read_text())
    expected = registry.get("registry_sha256")
    actual = registry_sha256(registry)
    if expected != actual:
        raise FormalDataContractError(
            f"formal data registry SHA256 mismatch: expected {expected}, got {actual}"
        )
    return registry


def derive_training_schedule(problems: list[MathProblem]) -> list[list[str]]:
    gsm8k = [problem.problem_id for problem in problems if problem.source == "gsm8k"]
    math = [problem.problem_id for problem in problems if problem.source == "math"]
    if len(gsm8k) != 64 or len(math) != 64:
        raise FormalDataContractError("formal train manifest must contain 64 GSM8K and 64 MATH")
    return [
        gsm8k[2 * index : 2 * index + 2] + math[2 * index : 2 * index + 2] for index in range(32)
    ]


def _validate_manifest(name: str, specification: dict[str, Any]) -> tuple[Path, list[MathProblem]]:
    path = Path(specification["path"])
    if not path.is_file():
        raise FormalDataContractError(f"missing formal manifest: {path}")
    actual_file_sha = file_sha256(path)
    if actual_file_sha != specification["file_sha256"]:
        raise FormalDataContractError(f"formal manifest file SHA256 mismatch: {name}")
    problems = load_manifest(path)
    if len(problems) != specification["count"]:
        raise FormalDataContractError(f"formal manifest count mismatch: {name}")
    if len({problem.problem_id for problem in problems}) != len(problems):
        raise FormalDataContractError(f"duplicate formal problem ID: {name}")
    if len({problem.content_hash for problem in problems}) != len(problems):
        raise FormalDataContractError(f"duplicate formal problem content: {name}")
    problem_ids = [problem.problem_id for problem in problems]
    content_hashes = [problem.content_hash for problem in problems]
    if ordered_values_sha256(problem_ids) != specification["ordered_problem_ids_sha256"]:
        raise FormalDataContractError(f"ordered formal problem IDs drifted: {name}")
    if ordered_values_sha256(content_hashes) != specification["ordered_content_hashes_sha256"]:
        raise FormalDataContractError(f"ordered formal content hashes drifted: {name}")
    expected_distributions = specification["distributions"]
    actual_distributions = {
        "source": _source_distribution(problems),
        "category": _distribution(problems, "category"),
        "difficulty": _distribution(problems, "difficulty"),
    }
    if actual_distributions != expected_distributions:
        raise FormalDataContractError(f"formal manifest distribution drifted: {name}")
    for problem in problems:
        source = specification["source_contracts"][problem.source]
        if problem.metadata.get("dataset_id") != source["dataset_id"]:
            raise FormalDataContractError(f"dataset identity drifted: {problem.problem_id}")
        if problem.metadata.get("revision") != source["revision"]:
            raise FormalDataContractError(f"dataset revision drifted: {problem.problem_id}")
    return path, problems


def validate_formal_data_registry(
    path: Path = DEFAULT_REGISTRY_PATH,
) -> dict[str, Any]:
    """Validate frozen external manifests without loading datasets, models, or tokenizers."""

    registry = load_formal_data_registry(path)
    loaded = {
        name: _validate_manifest(name, specification)[1]
        for name, specification in registry["manifests"].items()
    }
    core_names = ("train", "validation", "gsm8k_test", "math500_test")
    for index, left_name in enumerate(core_names):
        left = {problem.content_hash for problem in loaded[left_name]}
        for right_name in core_names[index + 1 :]:
            right = {problem.content_hash for problem in loaded[right_name]}
            if left & right:
                raise FormalDataContractError(
                    f"cross-split content leak: {left_name} vs {right_name}"
                )

    for subset_name, parent_name in (
        ("gsm8k_pass4", "gsm8k_test"),
        ("math500_pass4", "math500_test"),
    ):
        subset = loaded[subset_name]
        parent = loaded[parent_name]
        parent_ids = {problem.problem_id for problem in parent}
        if len(subset) != 50 or not {problem.problem_id for problem in subset} <= parent_ids:
            raise FormalDataContractError(f"invalid pass@4 subset: {subset_name}")

    correction = registry["provenance_corrections"]["validation"]
    validation = loaded["validation"]
    if correction != {
        "historical_metadata_source_split": "validation",
        "physical_source_split": "train",
        "row_count": 64,
        "selection_split": "validation",
    }:
        raise FormalDataContractError("unexpected validation provenance correction")
    if any(
        problem.metadata.get("source_split") != correction["historical_metadata_source_split"]
        for problem in validation
    ):
        raise FormalDataContractError("historical validation provenance evidence drifted")

    train = loaded["train"]
    derived_schedule = derive_training_schedule(train)
    schedule = registry["training_schedule"]
    if schedule["schedule_sha256"] != canonical_sha256(derived_schedule):
        raise FormalDataContractError("formal training schedule SHA256 mismatch")
    flattened = [problem_id for group in derived_schedule for problem_id in group]
    if schedule["ordered_problem_ids_sha256"] != ordered_values_sha256(flattened):
        raise FormalDataContractError("formal training schedule ordered IDs drifted")
    if len(flattened) != 128 or len(set(flattened)) != 128:
        raise FormalDataContractError("formal training schedule must cover 128 unique problems")
    if set(flattened) != {problem.problem_id for problem in train}:
        raise FormalDataContractError("formal training schedule does not cover the train manifest")

    return {
        "registry_sha256": registry["registry_sha256"],
        "manifest_count": len(loaded),
        "train_problem_count": len(train),
        "validation_problem_count": len(validation),
        "test_problem_count": len(loaded["gsm8k_test"]) + len(loaded["math500_test"]),
        "training_updates": len(derived_schedule),
        "problems_per_update": len(derived_schedule[0]),
        "validation_provenance_correction_count": correction["row_count"],
    }


def validate_local_source_artifacts(
    path: Path = DEFAULT_REGISTRY_PATH,
) -> dict[str, Any]:
    """Verify cached Arrow provenance and all-MATH500 leakage entirely offline."""

    from datasets import Dataset

    registry = load_formal_data_registry(path)
    loaded = {
        name: _validate_manifest(name, specification)[1]
        for name, specification in registry["manifests"].items()
    }
    datasets: dict[tuple[str, str], Any] = {}
    for source in registry["sources"].values():
        for split, artifact in source["local_arrow_artifacts"].items():
            artifact_path = Path(artifact["path"])
            if not artifact_path.is_file() or file_sha256(artifact_path) != artifact["sha256"]:
                raise FormalDataContractError(f"local source artifact drifted: {artifact_path}")
            dataset = Dataset.from_file(str(artifact_path))
            if len(dataset) != artifact["row_count"]:
                raise FormalDataContractError(f"local source row count drifted: {artifact_path}")
            datasets[(source["dataset_id"], split)] = dataset

    correction = registry["provenance_corrections"]["validation"]
    verified_rows = 0
    for manifest_name, problems in loaded.items():
        if manifest_name.endswith("pass4"):
            continue
        for problem in problems:
            physical_split = (
                correction["physical_source_split"]
                if manifest_name == "validation"
                else str(problem.metadata["source_split"])
            )
            source_rows = datasets[(str(problem.metadata["dataset_id"]), physical_split)]
            row = source_rows[int(problem.metadata["source_index"])]
            prompt = row["question"] if problem.source == "gsm8k" else row["problem"]
            if problem.prompt != prompt or problem.content_hash != content_hash(prompt):
                raise FormalDataContractError(f"source row mismatch: {problem.problem_id}")
            verified_rows += 1

    math500_source = registry["sources"]["math500"]
    math500_rows = datasets[(math500_source["dataset_id"], "test")]
    all_math500_hashes = {content_hash(row["problem"]) for row in math500_rows}
    train_and_validation_hashes = {
        problem.content_hash for name in ("train", "validation") for problem in loaded[name]
    }
    if all_math500_hashes & train_and_validation_hashes:
        raise FormalDataContractError("full MATH500 leakage into train or validation")
    return {
        "source_artifact_count": sum(
            len(source["local_arrow_artifacts"]) for source in registry["sources"].values()
        ),
        "verified_manifest_source_rows": verified_rows,
        "all_math500_problem_count": len(all_math500_hashes),
        "math500_train_validation_overlap": 0,
    }
