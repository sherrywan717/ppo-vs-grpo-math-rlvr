"""CPU-only contracts for the pre-registered GRPO-v2 experiment."""

from __future__ import annotations

import hashlib
import json
import math
import random
from fractions import Fraction
from pathlib import Path
from statistics import fmean, stdev

CHECKPOINT_STEPS = (32, 64, 96, 128)
PASS_K_VALUES = (1, 4, 10)
VERIFIED_PASS_STATUS = "VERIFIED_PASS"
PASS_K_SEED_NAMESPACE = "grpo_v2/pass_k_shared_n10"


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


def pass_k_batch_seed(*, problem_id: str, content_hash: str, seed: int = 42) -> int:
    """Derive one stable 63-bit batch seed shared by all four evaluated models."""
    digest = canonical_json_sha256(
        {
            "namespace": PASS_K_SEED_NAMESPACE,
            "evaluation_seed": seed,
            "problem_id": problem_id,
            "content_hash": content_hash,
        }
    )
    return int(digest[:16], 16) & ((1 << 63) - 1)


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


def unbiased_pass_at_k(*, n: int, c: int, k: int) -> dict[str, object]:
    """Return the exact combinatorial unbiased pass@k estimate for n=10."""
    if n != 10:
        raise ValueError("shared pass@k contract requires n=10")
    if c < 0 or c > n:
        raise ValueError("canonical pass count must satisfy 0 <= c <= n")
    if k < 1 or k > n or k not in PASS_K_VALUES:
        raise ValueError("k must be one of 1, 4, or 10 and no greater than n")
    denominator = math.comb(n, k)
    miss_combinations = math.comb(n - c, k) if n - c >= k else 0
    rational = Fraction(denominator - miss_combinations, denominator)
    return {
        "n": n,
        "c": c,
        "k": k,
        "combination_n_minus_c_choose_k": miss_combinations,
        "combination_n_choose_k": denominator,
        "pass_numerator": rational.numerator,
        "pass_denominator": rational.denominator,
        "exact_rational": str(rational),
        "float_value": float(rational),
    }


def shared_pass_k_estimates(candidate_correctness: list[bool]) -> dict[str, object]:
    """Compute all frozen unbiased estimates from one exchangeable ten-draw batch."""
    if len(candidate_correctness) != 10 or any(
        type(value) is not bool for value in candidate_correctness
    ):
        raise ValueError("shared pass@k requires exactly ten boolean candidates")
    c = sum(candidate_correctness)
    estimates = {str(k): unbiased_pass_at_k(n=10, c=c, k=k) for k in PASS_K_VALUES}
    values = [estimates[str(k)]["float_value"] for k in PASS_K_VALUES]
    if values != sorted(values):
        raise ValueError("problem-level unbiased pass@k monotonicity failed")
    return {
        "n": 10,
        "c": c,
        "candidate_correctness": candidate_correctness,
        "estimates": estimates,
    }


def validate_shared_candidate_batch(rows: list[dict]) -> dict[str, object]:
    """Validate one problem's single generate(n=10) evidence batch."""
    if len(rows) != 10:
        raise ValueError("shared pass@k requires exactly ten candidate evidence rows")
    indices = [row.get("candidate_index") for row in rows]
    if len(indices) != len(set(indices)) or set(indices) != set(range(10)):
        raise ValueError("candidate indices must be unique and exactly 0..9")
    identity_fields = (
        "problem_id",
        "content_hash",
        "model_identity",
        "checkpoint_identity",
        "batch_seed",
        "prompt_hash",
        "generate_call_id",
        "sampling_config",
    )
    for field in identity_fields:
        values = [row.get(field) for row in rows]
        if any(value is None or value == "" for value in values):
            raise ValueError(f"candidate batch {field} is required")
        if len({canonical_json_sha256(value) for value in values}) != 1:
            raise ValueError(f"candidate batch {field} mismatch")
    sampling = rows[0].get("sampling_config")
    if not isinstance(sampling, dict) or sampling.get("num_return_sequences") != 10:
        raise ValueError("candidate batch must use one num_return_sequences=10 sampling config")
    if rows[0]["batch_seed"] != pass_k_batch_seed(
        problem_id=rows[0]["problem_id"], content_hash=rows[0]["content_hash"]
    ):
        raise ValueError("candidate batch seed does not match the frozen derivation")
    ordered = sorted(rows, key=lambda row: row["candidate_index"])
    correctness: list[bool] = []
    references: list[str] = []
    for row in ordered:
        canonical_correct = row.get("canonical_correct")
        if type(canonical_correct) is not bool:
            raise ValueError("canonical correctness must be boolean")
        if canonical_correct != (row.get("verifier_status") == VERIFIED_PASS_STATUS):
            raise ValueError("verifier status and canonical correctness disagree")
        reference = row.get("evidence_ref")
        if not isinstance(reference, str) or not reference:
            raise ValueError("candidate evidence reference is required")
        correctness.append(canonical_correct)
        references.append(reference)
    result = shared_pass_k_estimates(correctness)
    result.update(
        {
            "problem_id": ordered[0]["problem_id"],
            "content_hash": ordered[0]["content_hash"],
            "candidate_evidence_references": references,
        }
    )
    return result


def validate_model_evaluation_ledger(
    rows: list[dict], *, all_problem_ids: set[str], shared_problem_ids: set[str]
) -> dict[str, int]:
    """Validate the frozen 400/100 shared ledger and exact 1,300-row budget."""
    if len(all_problem_ids) != 400 or len(shared_problem_ids) != 100:
        raise ValueError("evaluation problem universe must be 400 overall and 100 shared")
    if not shared_problem_ids < all_problem_ids:
        raise ValueError("shared pass@k universe must be a strict subset of all 400 problems")
    if len(rows) != 1300:
        raise ValueError("one model must contain exactly 1,300 completion rows")
    keys = [(row.get("problem_id"), row.get("candidate_index")) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate problem/candidate evidence key")
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row.get("problem_id"), []).append(row)
    if set(grouped) != all_problem_ids:
        raise ValueError("evaluation problem universe drift")
    for problem_id, problem_rows in grouped.items():
        if problem_id in shared_problem_ids:
            validate_shared_candidate_batch(problem_rows)
        else:
            if len(problem_rows) != 1 or problem_rows[0].get("candidate_index") != 0:
                raise ValueError("non-subset problems require exactly candidate 0")
            row = problem_rows[0]
            if row.get("canonical_correct") != (row.get("verifier_status") == VERIFIED_PASS_STATUS):
                raise ValueError("candidate-0 verifier/correctness mismatch")
            if not isinstance(row.get("evidence_ref"), str) or not row["evidence_ref"]:
                raise ValueError("candidate-0 evidence reference is required")
    if sum(row.get("candidate_index") == 0 for row in rows) != 400:
        raise ValueError("candidate 0 must occur exactly once for each of 400 problems")
    return {
        "all_problem_count": 400,
        "shared_problem_count": 100,
        "candidate0_rows": 400,
        "shared_candidate_rows": 1000,
        "completion_rows": 1300,
    }


def bootstrap_mean_interval(
    values: list[float], *, samples: int = 10_000, seed: int = 42
) -> tuple[float, float]:
    if not values or samples < 1:
        raise ValueError("bootstrap requires values and a positive sample count")
    rng = random.Random(seed)
    size = len(values)
    means = sorted(fmean(values[rng.randrange(size)] for _ in range(size)) for _ in range(samples))
    return means[int(0.025 * (samples - 1))], means[int(0.975 * (samples - 1))]


def aggregate_unbiased_pass_k(
    problem_rows: list[dict[str, object]], *, bootstrap_samples: int = 10_000
) -> dict[str, object]:
    """Aggregate problem-level estimates and enforce aggregate monotonicity."""
    if not problem_rows:
        raise ValueError("pass@k aggregation requires problem rows")
    result: dict[str, object] = {"problem_denominator": len(problem_rows), "metrics": {}}
    means: list[float] = []
    for k in PASS_K_VALUES:
        values = [float(row["estimates"][str(k)]["float_value"]) for row in problem_rows]
        mean = fmean(values)
        standard_error = stdev(values) / math.sqrt(len(values)) if len(values) > 1 else None
        interval = bootstrap_mean_interval(values, samples=bootstrap_samples)
        result["metrics"][str(k)] = {
            "mean": mean,
            "standard_error": standard_error,
            "bootstrap_95": {"lower": interval[0], "upper": interval[1]},
            "candidate_denominator": 10 * len(values),
        }
        means.append(mean)
    if means != sorted(means):
        raise ValueError("aggregate unbiased pass@k monotonicity failed")
    return result


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
    registry = json.loads((root / "configs/grpo_v2/data_registry.json").read_text())
    active = registry.get("pass_k_shared_n10_subset", {})
    if (
        active.get("active") is not True
        or active.get("count") != 100
        or active.get("path") != "configs/grpo_v2/manifests/pass4_nested_subset.json"
        or active.get("sha256")
        != hashlib.sha256((manifest_dir / "pass4_nested_subset.json").read_bytes()).hexdigest()
    ):
        raise ValueError("active shared n=10 subset registry contract failed")
    legacy = registry.get("legacy_pass10_nested_subset_o2", {})
    if (
        legacy.get("active") is not False
        or legacy.get("status") != "superseded_before_any_evaluation"
        or legacy.get("superseded_by") != "pass_k_shared_n10_subset"
    ):
        raise ValueError("legacy O.2 pass@10 manifest remains active")
    evaluation_identity = registry.get("evaluation_contract", {})
    evaluation_path = root / evaluation_identity.get("path", "missing")
    pass_k_path = root / evaluation_identity.get("pass_k_contract_path", "missing")
    if (
        not evaluation_path.is_file()
        or not pass_k_path.is_file()
        or hashlib.sha256(evaluation_path.read_bytes()).hexdigest()
        != evaluation_identity.get("sha256")
        or hashlib.sha256(pass_k_path.read_bytes()).hexdigest()
        != evaluation_identity.get("pass_k_contract_sha256")
    ):
        raise ValueError("evaluation/pass@k contract registry SHA mismatch")
    evaluation = json.loads(evaluation_path.read_text())
    if (
        evaluation.get("pass_k_shared_n10_problem_count") != 100
        or evaluation.get("shared_candidates_per_problem") != 10
        or evaluation.get("completion_ledger", {}).get("per_model") != 1300
        or evaluation.get("completion_ledger", {}).get("four_models") != 5200
    ):
        raise ValueError("shared n=10 evaluation ledger contract failed")
    if evaluation.get("sampling", {}).get("per_problem_batch_seed") != {
        "evaluation_seed": 42,
        "inputs": ["problem_id", "content_hash"],
        "method": "sha256_first_64_bits_masked_to_63_bits",
        "namespace": PASS_K_SEED_NAMESPACE,
        "shared_across_models": True,
    }:
        raise ValueError("shared n=10 batch-seed derivation contract failed")
    curriculum = json.loads((root / "configs/grpo_v2/curriculum.json").read_text())
    positions = curriculum["positions"]
    if len(positions) != 512 or {p["problem_id"] for p in positions} != train_ids:
        raise ValueError("curriculum coverage failed")
    if [p["position"] for p in positions] != list(range(1, 513)):
        raise ValueError("curriculum order failed")
    return {
        "status": "passed",
        "counts": {k: len(v) for k, v in manifests.items()},
        "shared_n10_subset": 100,
        "nested": 100,
        "completions_per_model": 1300,
        "four_models_total": 5200,
    }


if __name__ == "__main__":
    print(json.dumps(validate_contract_tree(Path(__file__).resolve().parents[2]), sort_keys=True))
