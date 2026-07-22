import json
from pathlib import Path

import pytest

from math_rlvr.grpo_v2_contract import (
    select_checkpoint,
    selection_key,
    validate_contract_tree,
    wilson_interval,
)

ROOT = Path(__file__).resolve().parents[1]


def test_frozen_contract_counts_and_overlap():
    result = validate_contract_tree(ROOT)
    assert result["counts"] == {
        "train_v2": 512,
        "warmstart_v2": 256,
        "dev_v2": 128,
        "test_v2_hidden": 400,
    }
    assert result["nested"] == 100


def test_math_stratification_and_nested_contract():
    rows = [
        json.loads(x)
        for x in (ROOT / "configs/grpo_v2/manifests/test_v2_hidden.jsonl").read_text().splitlines()
    ]
    nested = json.loads((ROOT / "configs/grpo_v2/manifests/pass4_nested_subset.json").read_text())[
        "problems"
    ]
    levels = {
        level: sum(r["source"] == "math" and r["difficulty"] == str(level) for r in rows)
        for level in range(1, 6)
    }
    nested_levels = {
        level: sum(r["source"] == "math" and r["difficulty"] == str(level) for r in nested)
        for level in range(1, 6)
    }
    assert levels == {1: 3, 2: 33, 3: 43, 4: 59, 5: 62}
    assert nested_levels == {1: 3, 2: 8, 3: 10, 4: 14, 5: 15}
    all_l1 = {r["problem_id"] for r in rows if r["source"] == "math" and r["difficulty"] == "1"}
    nested_l1 = {
        r["problem_id"] for r in nested if r["source"] == "math" and r["difficulty"] == "1"
    }
    assert len(all_l1) == 3 and nested_l1 == all_l1


def test_selection_key_ignores_gold_and_is_stable():
    args = dict(
        dataset_revision="abc",
        source_split="test",
        source_problem_id="id",
        namespace="test",
        seed=42,
    )
    assert selection_key(**args) == selection_key(**args)
    assert selection_key(**args) != selection_key(**{**args, "namespace": "nested"})


def test_hidden_evaluation_ledger_and_small_n_contract():
    contract = json.loads((ROOT / "configs/grpo_v2/evaluation.json").read_text())
    assert contract["candidate0_problem_count"] == 400
    assert contract["nested_pass4_problem_count"] == 100
    assert contract["extra_nested_candidates"] == 300
    assert contract["nested_candidate_rows"] == 400
    assert contract["completions_per_model"] == 1000
    assert contract["total_four_model_completions"] == 4000
    assert contract["nested_pass10_problem_count"] == 50
    assert contract["pass10_candidate_rows"] == 500
    assert contract["math_level_reporting"]["1"]["denominator"] == 3
    assert contract["math_level_reporting"]["1"]["status"] == "diagnostic_only_small_n"
    lo, hi = wilson_interval(1, 3)
    assert lo < 1 / 3 < hi and hi - lo > 0.5


def test_checkpoint_selection_is_dev_only_and_lexicographic():
    rows = [
        {
            "checkpoint_step": s,
            "evaluation_split": "dev_v2",
            "canonical_pass_at_1": 0.1,
            "parseable_rate": 0.2,
            "format_rate": 0.3,
            "truncation_rate": 0.1,
        }
        for s in (32, 64, 96, 128)
    ]
    assert select_checkpoint(rows) == 32
    rows[2]["canonical_pass_at_1"] = 0.2
    assert select_checkpoint(rows) == 96
    rows[2]["evaluation_split"] = "test_v2_hidden"
    with pytest.raises(ValueError):
        select_checkpoint(rows)


def test_all_core_hash_and_source_overlaps_are_zero():
    audit = json.loads((ROOT / "reports/grpo_v2/data_leakage_audit.json").read_text())
    assert set(audit["core_cross_split_overlap_counts"].values()) == {0}
    assert set(audit["source_identity_overlap_counts"].values()) == {0}


def test_manifest_hashes_match_frozen_registry():
    import hashlib

    registry = json.loads((ROOT / "configs/grpo_v2/data_registry.json").read_text())
    for item in registry["manifests"].values():
        assert hashlib.sha256((ROOT / item["path"]).read_bytes()).hexdigest() == item["sha256"]
