import csv
import json
from pathlib import Path

import pytest

from math_rlvr.grpo_v2_contract import (
    aggregate_unbiased_pass_k,
    canonical_json_sha256,
    pass_k_batch_seed,
    shared_pass_k_estimates,
    unbiased_pass_at_k,
    validate_contract_tree,
    validate_model_evaluation_ledger,
    validate_shared_candidate_batch,
)

ROOT = Path(__file__).resolve().parents[1]
SAMPLING = {
    "temperature": 0.8,
    "top_p": 0.95,
    "top_k": None,
    "max_completion_length": 256,
    "num_return_sequences": 10,
}


def candidate(problem_id: str, index: int, *, correct: bool = False, sampling=None) -> dict:
    return {
        "problem_id": problem_id,
        "content_hash": f"hash-{problem_id}",
        "model_identity": "model",
        "checkpoint_identity": "checkpoint",
        "batch_seed": pass_k_batch_seed(problem_id=problem_id, content_hash=f"hash-{problem_id}"),
        "prompt_hash": f"prompt-{problem_id}",
        "generate_call_id": f"call-{problem_id}",
        "sampling_config": dict(SAMPLING if sampling is None else sampling),
        "candidate_index": index,
        "canonical_correct": correct,
        "verifier_status": "VERIFIED_PASS" if correct else "WRONG_ANSWER",
        "evidence_ref": f"evidence:{problem_id}:{index}",
    }


@pytest.mark.parametrize(
    ("c", "expected"),
    [
        (0, {1: ("0", 0.0), 4: ("0", 0.0), 10: ("0", 0.0)}),
        (1, {1: ("1/10", 0.1), 4: ("2/5", 0.4), 10: ("1", 1.0)}),
        (2, {1: ("1/5", 0.2), 4: ("2/3", 2 / 3), 10: ("1", 1.0)}),
        (10, {1: ("1", 1.0), 4: ("1", 1.0), 10: ("1", 1.0)}),
    ],
)
def test_exact_unbiased_formula(c, expected):
    for k, (rational, value) in expected.items():
        result = unbiased_pass_at_k(n=10, c=c, k=k)
        assert result["exact_rational"] == rational
        assert result["float_value"] == pytest.approx(value)
        assert result["combination_n_choose_k"] > 0


def test_per_problem_batch_seed_is_stable_and_identity_bound():
    first = pass_k_batch_seed(problem_id="p0", content_hash="h0")
    assert first == pass_k_batch_seed(problem_id="p0", content_hash="h0")
    assert first != pass_k_batch_seed(problem_id="p1", content_hash="h0")
    assert 0 <= first < 2**63


def test_formula_rejects_invalid_n_c_and_k():
    for kwargs in (
        {"n": 9, "c": 1, "k": 1},
        {"n": 10, "c": -1, "k": 1},
        {"n": 10, "c": 11, "k": 1},
        {"n": 10, "c": 1, "k": 11},
    ):
        with pytest.raises(ValueError):
            unbiased_pass_at_k(**kwargs)


def test_shared_batch_identity_candidates_and_verifier_contract():
    rows = [candidate("p0", i, correct=i in {2, 7}) for i in range(10)]
    result = validate_shared_candidate_batch(rows)
    assert result["c"] == 2
    assert result["estimates"]["4"]["exact_rational"] == "2/3"
    for invalid in (rows[:9], rows + [candidate("p0", 10)]):
        with pytest.raises(ValueError):
            validate_shared_candidate_batch(invalid)
    duplicate = [candidate("p0", i) for i in range(9)] + [candidate("p0", 8)]
    with pytest.raises(ValueError, match="indices"):
        validate_shared_candidate_batch(duplicate)
    missing = [candidate("p0", i) for i in range(9)] + [candidate("p0", 10)]
    with pytest.raises(ValueError, match="indices"):
        validate_shared_candidate_batch(missing)
    drift = [candidate("p0", i) for i in range(10)]
    drift[-1]["problem_id"] = "other"
    with pytest.raises(ValueError, match="problem_id"):
        validate_shared_candidate_batch(drift)
    mismatch = [candidate("p0", i) for i in range(10)]
    mismatch[0]["canonical_correct"] = True
    with pytest.raises(ValueError, match="disagree"):
        validate_shared_candidate_batch(mismatch)
    sampling = [candidate("p0", i) for i in range(10)]
    sampling[-1]["sampling_config"]["top_p"] = 0.9
    with pytest.raises(ValueError, match="sampling_config"):
        validate_shared_candidate_batch(sampling)
    missing_identity = [candidate("p0", i) for i in range(10)]
    missing_identity[0].pop("prompt_hash")
    with pytest.raises(ValueError, match="required"):
        validate_shared_candidate_batch(missing_identity)


def make_ledger():
    all_ids = {f"p{i}" for i in range(400)}
    shared_ids = {f"p{i}" for i in range(100)}
    rows = []
    for problem_id in sorted(all_ids):
        indices = range(10) if problem_id in shared_ids else (0,)
        rows.extend(candidate(problem_id, index, correct=index == 0) for index in indices)
    return rows, all_ids, shared_ids


def test_exact_1300_row_ledger_and_candidate0_not_duplicated():
    rows, all_ids, shared_ids = make_ledger()
    result = validate_model_evaluation_ledger(
        rows, all_problem_ids=all_ids, shared_problem_ids=shared_ids
    )
    assert result == {
        "all_problem_count": 400,
        "shared_problem_count": 100,
        "candidate0_rows": 400,
        "shared_candidate_rows": 1000,
        "completion_rows": 1300,
    }
    with pytest.raises(ValueError):
        validate_model_evaluation_ledger(
            rows + [dict(rows[0])], all_problem_ids=all_ids, shared_problem_ids=shared_ids
        )
    drift = [dict(row) for row in rows]
    target = next(row for row in drift if row["problem_id"] == "p399")
    target["problem_id"] = "rogue"
    with pytest.raises(ValueError, match="universe"):
        validate_model_evaluation_ledger(
            drift, all_problem_ids=all_ids, shared_problem_ids=shared_ids
        )


def test_problem_and_aggregate_monotonicity_and_rebuild_examples():
    problem_rows = [shared_pass_k_estimates([i < c for i in range(10)]) for c in (0, 1, 2, 10)]
    for row in problem_rows:
        values = [row["estimates"][str(k)]["float_value"] for k in (1, 4, 10)]
        assert values == sorted(values)
    aggregate = aggregate_unbiased_pass_k(problem_rows, bootstrap_samples=100)
    means = [aggregate["metrics"][str(k)]["mean"] for k in (1, 4, 10)]
    assert means == sorted(means)
    assert json.loads(json.dumps(aggregate)) == aggregate
    with (ROOT / "reports/grpo_v2/pass_k_estimator_examples.csv").open() as handle:
        examples = list(csv.DictReader(handle))
    assert len(examples) == 12
    for row in examples:
        rebuilt = unbiased_pass_at_k(n=int(row["n"]), c=int(row["c"]), k=int(row["k"]))
        assert rebuilt["exact_rational"] == row["exact_rational"]
        assert rebuilt["float_value"] == pytest.approx(float(row["float_value"]))


def test_active_shared_subset_and_superseded_legacy_contract():
    assert validate_contract_tree(ROOT)["nested"] == 100
    shared = json.loads((ROOT / "configs/grpo_v2/manifests/pass4_nested_subset.json").read_text())[
        "problems"
    ]
    assert len(shared) == 100
    assert sum(row["source"] == "gsm8k" for row in shared) == 50
    assert {
        level: sum(row["source"] == "math" and row["difficulty"] == str(level) for row in shared)
        for level in range(1, 6)
    } == {1: 3, 2: 8, 3: 10, 4: 14, 5: 15}
    assert not (ROOT / "configs/grpo_v2/manifests/pass10_nested_subset.json").exists()
    assert (ROOT / "configs/grpo_v2/manifests/legacy/pass10_nested_subset_o2.json").is_file()


def test_evaluation_budget_and_distinct_metric_names():
    contract = json.loads((ROOT / "configs/grpo_v2/evaluation.json").read_text())
    assert contract["completion_ledger"]["per_model"] == 1300
    assert contract["completion_ledger"]["four_models"] == 5200
    assert contract["metrics"]["candidate0_accuracy_all_400"]["problem_denominator"] == 400
    for k in (1, 4, 10):
        assert contract["metrics"][f"unbiased_pass_at_{k}_subset_100"]["problem_denominator"] == 100
    assert contract["sampling"]["per_problem_batch_seed"] == {
        "evaluation_seed": 42,
        "inputs": ["problem_id", "content_hash"],
        "method": "sha256_first_64_bits_masked_to_63_bits",
        "namespace": "grpo_v2/pass_k_shared_n10",
        "shared_across_models": True,
    }


def test_protected_sha_and_registry_transitive_identity():
    import hashlib

    expected = [
        (
            "configs/grpo_v2/manifests/train_v2.jsonl",
            "ca3403ae7b0c1f2689e21aca3283348f89b2ffb65498329e74cc4fa7fde8b664",
        ),
        (
            "configs/grpo_v2/manifests/warmstart_v2.jsonl",
            "a83ffb60c6aecd5e23b980dfe8606fae8321b70bedf0d3e906f3d0d05e7106f7",
        ),
        (
            "configs/grpo_v2/manifests/dev_v2.jsonl",
            "bdf02e1202e564177fea59f80f0b0ac8a36649daf8636ed6dd5bf3e5f6356b80",
        ),
        (
            "configs/grpo_v2/manifests/test_v2_hidden.jsonl",
            "1da04a0093382711d618f515261b417e7df085d8e4fe93ddb34d314868062285",
        ),
        (
            "configs/grpo_v2/manifests/pass4_nested_subset.json",
            "86864437418a1a112b6385991bdba83617b0cc66f85ec6c7032f0ee79763a553",
        ),
        (
            "configs/grpo_v2/warmstart_seed42.json",
            "c8e3e0a52d55f46b201c1b4a95e0f9b2f910ae558e477b55a35d03fd4ec8549a",
        ),
        (
            "configs/grpo_v2/grpo_v2_seed42.json",
            "059553888fdc997a5b9f214fde526d4be8c309ca84abe212c243fd74305b1b66",
        ),
        (
            "configs/grpo_v2/curriculum.json",
            "7f7dcfa1218828e72dd6d42783bc2c790897c7e2a8f2f84d59ce2189710e3b41",
        ),
    ]
    for path, digest in expected:
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    data_registry = json.loads((ROOT / "configs/grpo_v2/data_registry.json").read_text())
    data_claim = data_registry.pop("registry_sha256")
    assert canonical_json_sha256(data_registry) == data_claim
    runtime_registry = json.loads((ROOT / "configs/grpo_v2/runtime_registry.json").read_text())
    runtime_claim = runtime_registry.pop("registry_sha256")
    assert canonical_json_sha256(runtime_registry) == runtime_claim
    assert runtime_registry["warmstart"]["transitive_identity_change"] == (
        "evaluation_registry_only_no_training_field_change"
    )
    assert (
        runtime_registry["warmstart"]["data_registry_sha256"]
        == hashlib.sha256((ROOT / "configs/grpo_v2/data_registry.json").read_bytes()).hexdigest()
    )
