import json
from collections import Counter
from pathlib import Path

from math_rlvr.dataset import load_manifest
from math_rlvr.training.formal_data import (
    DEFAULT_REGISTRY_PATH,
    canonical_sha256,
    derive_training_schedule,
    load_formal_data_registry,
    ordered_values_sha256,
    validate_formal_data_registry,
    validate_local_source_artifacts,
)


def test_formal_data_registry_and_frozen_identities():
    registry = load_formal_data_registry()
    assert registry["schema_version"] == "formal_1p5b_data_registry_v1"
    assert registry["selection_seed"] == 20260712
    assert registry["registry_sha256"] == (
        "d7c53f6180187711da780a3a1f81f8b45e6164ddc9f115eac2fb6ae3e1fe7393"
    )
    assert {name: source["revision"] for name, source in registry["sources"].items()} == {
        "gsm8k": "740312add88f781978c0658806c59bc2815b9866",
        "math": "0530c78699ea5e8eb5530600900e1f328b48acad",
        "math500": "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be",
    }
    assert {name: spec["file_sha256"] for name, spec in registry["manifests"].items()} == {
        "train": "553939ce40ef20af86f5eabe987bff42814f07e9d40ddf1c4cde1208dcc96dd0",
        "validation": "83eee5f6191f003c3c5d8f273adb2a5631c848d1f196bfe34891efeca658e70d",
        "gsm8k_test": "2b84fe8130a93310dfe3cda206a09daa012defd3fab2d9f12dc86d592788e5b8",
        "math500_test": "34a407fad90ef229f5ce5e1c24d1c964b140ec1a14c9fa5f1ecf4559435b5c4f",
        "gsm8k_pass4": "51bc4189d4b039787310ad4c200aeb5f8b7f8dd7179c75d8d58eecb362ae4c8c",
        "math500_pass4": "2b2afd7e862231bbb5b2906e4280c8ffefca741683a60d70cb05d99c2065f99c",
    }
    assert "gold" not in DEFAULT_REGISTRY_PATH.read_text().lower()
    assert validate_formal_data_registry() == {
        "registry_sha256": registry["registry_sha256"],
        "manifest_count": 6,
        "train_problem_count": 128,
        "validation_problem_count": 64,
        "test_problem_count": 400,
        "training_updates": 32,
        "problems_per_update": 4,
        "validation_provenance_correction_count": 64,
    }


def test_formal_schedule_is_exact_domain_balanced_cover():
    registry = load_formal_data_registry()
    train = load_manifest(Path(registry["manifests"]["train"]["path"]))
    by_id = {problem.problem_id: problem for problem in train}
    schedule = derive_training_schedule(train)
    assert len(schedule) == 32
    assert all(len(group) == 4 for group in schedule)
    assert all(
        Counter(by_id[problem_id].source for problem_id in group) == {"gsm8k": 2, "math": 2}
        for group in schedule
    )
    flattened = [problem_id for group in schedule for problem_id in group]
    assert len(flattened) == len(set(flattened)) == 128
    assert set(flattened) == set(by_id)
    assert canonical_sha256(schedule) == registry["training_schedule"]["schedule_sha256"]
    assert (
        ordered_values_sha256(flattened)
        == registry["training_schedule"]["ordered_problem_ids_sha256"]
    )


def test_validation_provenance_is_corrected_without_rewriting_history():
    registry = load_formal_data_registry()
    validation_path = Path(registry["manifests"]["validation"]["path"])
    validation = load_manifest(validation_path)
    correction = registry["provenance_corrections"]["validation"]
    assert len(validation) == correction["row_count"] == 64
    assert {problem.metadata["source_split"] for problem in validation} == {"validation"}
    assert correction == {
        "historical_metadata_source_split": "validation",
        "physical_source_split": "train",
        "row_count": 64,
        "selection_split": "validation",
    }


def test_pass4_subsets_and_math500_level_strata():
    registry = load_formal_data_registry()
    manifests = {
        name: load_manifest(Path(spec["path"])) for name, spec in registry["manifests"].items()
    }
    for subset_name, parent_name in (
        ("gsm8k_pass4", "gsm8k_test"),
        ("math500_pass4", "math500_test"),
    ):
        subset_ids = {problem.problem_id for problem in manifests[subset_name]}
        parent_ids = {problem.problem_id for problem in manifests[parent_name]}
        assert len(subset_ids) == 50
        assert subset_ids <= parent_ids
    assert Counter(problem.difficulty for problem in manifests["math500_test"]) == {
        str(level): 40 for level in range(1, 6)
    }
    assert Counter(problem.difficulty for problem in manifests["math500_pass4"]) == {
        str(level): 10 for level in range(1, 6)
    }


def test_local_source_rows_revisions_and_full_math500_leakage():
    evidence = validate_local_source_artifacts()
    assert evidence == {
        "source_artifact_count": 4,
        "verified_manifest_source_rows": 592,
        "all_math500_problem_count": 500,
        "math500_train_validation_overlap": 0,
    }


def test_registry_contains_only_external_references_not_problem_payloads():
    registry = json.loads(DEFAULT_REGISTRY_PATH.read_text())
    forbidden = {"prompt", "gold_answer", "solution", "answer"}

    def visit(value):
        if isinstance(value, dict):
            assert not forbidden & set(value)
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(registry)
