"""Frozen baseline/final evaluation orchestration without eager model or CUDA imports."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Protocol

from math_rlvr.dataset import load_manifest
from math_rlvr.evaluation.formal import load_evaluation_config, validate_evaluation_config
from math_rlvr.training.formal import FORMAL_ACTIVE_SEEDS
from math_rlvr.training.formal_data import load_formal_data_registry
from math_rlvr.training.formal_runtime import FormalRuntimeError, _standard_json


class FormalEvaluationBackend(Protocol):
    def generate(self, plan: list[dict[str, Any]]) -> list[dict[str, Any]]: ...


REQUIRED_EVALUATION_FILES = (
    "completions.jsonl",
    "per_problem_metrics.csv",
    "aggregate_metrics.csv",
    "aggregate_metrics.json",
    "verifier_status.csv",
    "resource_metrics.csv",
    "resource_summary.json",
    "pytorch_allocator.json",
    "report.md",
)


def formal_evaluation_plan(
    config: dict[str, Any],
    phase: str,
    *,
    seed: int,
    algorithm: str | None = None,
    checkpoint_step: int | None = None,
) -> list[dict[str, Any]]:
    validate_evaluation_config(config, phase, algorithm=algorithm, seed=seed)
    if seed not in FORMAL_ACTIVE_SEEDS:
        raise FormalRuntimeError("evaluation seed is not active")
    if phase == "validation":
        if checkpoint_step not in {8, 16, 24, 32}:
            raise FormalRuntimeError("validation checkpoint step is not frozen")
        manifest_names = ("validation",)
    else:
        if phase == "final" and checkpoint_step != 32:
            raise FormalRuntimeError("formal final evaluation is fixed to checkpoint step 32")
        if phase == "baseline" and checkpoint_step is not None:
            raise FormalRuntimeError("untrained baseline cannot name a checkpoint")
        manifest_names = ("gsm8k_test", "math500_test")
    registry = load_formal_data_registry()
    plan: list[dict[str, Any]] = []
    loaded: dict[str, list[Any]] = {}
    for name in manifest_names:
        loaded[name] = load_manifest(Path(registry["manifests"][name]["path"]))
    if phase == "validation":
        for problem in loaded["validation"]:
            plan.append(
                {
                    "phase": phase,
                    "seed": seed,
                    "algorithm": algorithm,
                    "checkpoint_step": checkpoint_step,
                    "problem_id": problem.problem_id,
                    "problem_hash": problem.content_hash,
                    "domain": problem.source,
                    "difficulty": problem.difficulty,
                    "sample_kind": "validation",
                    "generation_index": 0,
                    "pair_key": f"{problem.problem_id}::validation::generation:0",
                }
            )
        return plan
    all_test = loaded["gsm8k_test"] + loaded["math500_test"]
    pass4_ids = {
        problem.problem_id
        for name in ("gsm8k_pass4", "math500_pass4")
        for problem in load_manifest(Path(registry["manifests"][name]["path"]))
    }
    for problem in all_test:
        common = {
            "phase": phase,
            "seed": seed,
            "algorithm": algorithm,
            "checkpoint_step": checkpoint_step,
            "problem_id": problem.problem_id,
            "problem_hash": problem.content_hash,
            "domain": "gsm8k" if problem.source == "gsm8k" else "math500",
            "difficulty": problem.difficulty,
        }
        plan.append(
            {
                **common,
                "sample_kind": "pass1",
                "generation_index": 0,
                "pair_key": f"{problem.problem_id}::pass1::generation:0",
            }
        )
        if problem.problem_id in pass4_ids:
            for generation_index in range(4):
                plan.append(
                    {
                        **common,
                        "sample_kind": "pass4",
                        "generation_index": generation_index,
                        "pair_key": (f"{problem.problem_id}::pass4::generation:{generation_index}"),
                    }
                )
    expected = 64 if phase == "validation" else 800
    if len(plan) != expected or len({row["pair_key"] for row in plan}) != expected:
        raise FormalRuntimeError("formal evaluation plan count/key mismatch")
    return plan


def _validate_completion_rows(
    plan: list[dict[str, Any]], rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    if len(rows) != len(plan):
        raise FormalRuntimeError("formal evaluation completion count mismatch")
    validated = []
    for expected, row in zip(plan, rows, strict=True):
        if any(row.get(key) != expected[key] for key in expected):
            raise FormalRuntimeError("formal evaluation completion identity/order mismatch")
        ids = row.get("completion_ids")
        mask = row.get("completion_mask")
        if not isinstance(ids, list) or not isinstance(mask, list) or len(ids) != len(mask):
            raise FormalRuntimeError("formal evaluation token evidence missing")
        if any(not isinstance(value, int) or isinstance(value, bool) for value in ids):
            raise FormalRuntimeError("formal evaluation token IDs invalid")
        if any(value not in (0, 1) for value in mask):
            raise FormalRuntimeError("formal evaluation mask invalid")
        count = sum(mask)
        if row.get("exact_token_count") != count or count > 256:
            raise FormalRuntimeError("formal evaluation exact token count invalid")
        for name in ("format_valid", "valid_answer", "canonical_correct", "truncated"):
            if not isinstance(row.get(name), bool):
                raise FormalRuntimeError(f"formal evaluation {name} must be boolean")
        reward = row.get("scalar_reward")
        if (
            isinstance(reward, bool)
            or not isinstance(reward, (int, float))
            or not math.isfinite(float(reward))
        ):
            raise FormalRuntimeError("formal evaluation scalar reward non-finite")
        if not isinstance(row.get("verifier_status"), str) or not isinstance(
            row.get("raw_completion"), str
        ):
            raise FormalRuntimeError("formal evaluation verifier/text evidence missing")
        _standard_json(row)
        validated.append(dict(row))
    return validated


def _aggregate(
    plan: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]] | None,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    by_problem: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_problem[row["problem_id"]].append(row)
    baseline_by_key = {
        row["pair_key"]: row for row in (baseline_rows or []) if isinstance(row, dict)
    }
    per_problem = []
    for problem_id, problem_rows in by_problem.items():
        pass1 = next((row for row in problem_rows if row["sample_kind"] == "pass1"), None)
        pass4 = [row for row in problem_rows if row["sample_kind"] == "pass4"]
        canonical = bool(pass1 and pass1["canonical_correct"])
        baseline = baseline_by_key.get(pass1["pair_key"]) if pass1 else None
        per_problem.append(
            {
                "problem_id": problem_id,
                "domain": problem_rows[0]["domain"],
                "difficulty": problem_rows[0]["difficulty"],
                "sampled_pass_at_1": int(canonical),
                "pass_at_4": int(any(row["canonical_correct"] for row in pass4)) if pass4 else "",
                "format_valid": int(bool(pass1 and pass1["format_valid"])),
                "valid_answer": int(bool(pass1 and pass1["valid_answer"])),
                "canonical_correct": int(canonical),
                "paired_pre_post_delta": (
                    int(canonical) - int(bool(baseline["canonical_correct"]))
                    if baseline is not None
                    else ""
                ),
            }
        )
    pass1_rows = [row for row in rows if row["sample_kind"] == "pass1"]
    pass4_groups = [
        group
        for group in by_problem.values()
        if any(row["sample_kind"] == "pass4" for row in group)
    ]
    aggregate = {
        "completion_count": len(rows),
        "unique_problem_count": len(by_problem),
        "pass1": (
            sum(row["canonical_correct"] for row in pass1_rows) / len(pass1_rows)
            if pass1_rows
            else None
        ),
        "pass4": (
            sum(
                any(row["canonical_correct"] for row in group if row["sample_kind"] == "pass4")
                for group in pass4_groups
            )
            / len(pass4_groups)
            if pass4_groups
            else None
        ),
        "format_accuracy": sum(row["format_valid"] for row in rows) / len(rows),
        "valid_answer_rate": sum(row["valid_answer"] for row in rows) / len(rows),
        "canonical_correctness": sum(row["canonical_correct"] for row in rows) / len(rows),
        "mean_completion_length": sum(row["exact_token_count"] for row in rows) / len(rows),
        "truncation_rate": sum(row["truncated"] for row in rows) / len(rows),
        "reward_mean": sum(float(row["scalar_reward"]) for row in rows) / len(rows),
        "test_driven_tuning": False,
    }
    aggregate.update(
        {
            "sampled_pass_at_1": aggregate["pass1"],
            "pass_at_4": aggregate["pass4"],
            "greedy_accuracy": None,
            "greedy_accuracy_available": False,
            "greedy_accuracy_unavailable_reason": (
                "frozen protocol has no separate greedy completion"
            ),
        }
    )
    domain_rows = []
    for domain in sorted({row["domain"] for row in rows}):
        selected = [row for row in pass1_rows if row["domain"] == domain]
        domain_rows.append(
            {
                "slice": domain,
                "problems": len(selected),
                "sampled_pass_at_1": (
                    sum(row["canonical_correct"] for row in selected) / len(selected)
                    if selected
                    else None
                ),
            }
        )
    math_rows = [row for row in pass1_rows if row["domain"] == "math500"]
    for level in ("1", "2", "3", "4", "5"):
        selected = [row for row in math_rows if str(row["difficulty"]) == level]
        domain_rows.append(
            {
                "slice": f"math500_level_{level}",
                "problems": len(selected),
                "sampled_pass_at_1": (
                    sum(row["canonical_correct"] for row in selected) / len(selected)
                    if selected
                    else None
                ),
            }
        )
    return per_problem, aggregate, domain_rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        if fields:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)


def execute_formal_evaluation(
    backend: FormalEvaluationBackend,
    *,
    phase: str,
    seed: int,
    run_dir: Path,
    algorithm: str | None = None,
    checkpoint_step: int | None = None,
    baseline_rows: list[dict[str, Any]] | None = None,
    config: dict[str, Any] | None = None,
    precreated_run_dir: bool = False,
) -> dict[str, Any]:
    config = config or load_evaluation_config()
    plan = formal_evaluation_plan(
        config, phase, seed=seed, algorithm=algorithm, checkpoint_step=checkpoint_step
    )
    rows = _validate_completion_rows(plan, backend.generate(plan))
    per_problem, aggregate, slices = _aggregate(plan, rows, baseline_rows)
    if precreated_run_dir:
        if not run_dir.is_dir():
            raise FormalRuntimeError("precreated evaluation run directory is missing")
        (run_dir / "figures").mkdir(exist_ok=True)
    else:
        run_dir.mkdir(parents=True, exist_ok=False)
        (run_dir / "figures").mkdir()
    (run_dir / "completions.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    _write_csv(run_dir / "per_problem_metrics.csv", per_problem)
    aggregate_rows = [{"slice": "all", **aggregate}, *slices]
    _write_csv(run_dir / "aggregate_metrics.csv", aggregate_rows)
    (run_dir / "aggregate_metrics.json").write_text(
        json.dumps({"aggregate": aggregate, "slices": slices}, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    statuses = Counter(row["verifier_status"] for row in rows)
    _write_csv(
        run_dir / "verifier_status.csv",
        [{"status": status, "count": count} for status, count in sorted(statuses.items())],
    )
    resource_probe = getattr(backend, "resource_metrics", None)
    resource_rows = resource_probe() if callable(resource_probe) else []
    if not resource_rows:
        resource_rows = [{"available": False, "reason": "resource telemetry unavailable"}]
    _write_csv(run_dir / "resource_metrics.csv", resource_rows)
    resource_summary = getattr(
        backend,
        "resource_summary",
        {"available": False, "reason": "resource summary unavailable"},
    )
    allocator = getattr(
        backend,
        "pytorch_allocator",
        {"available": False, "reason": "CUDA allocator telemetry unavailable"},
    )
    _standard_json(resource_summary)
    _standard_json(allocator)
    (run_dir / "resource_summary.json").write_text(
        json.dumps(resource_summary, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (run_dir / "pytorch_allocator.json").write_text(
        json.dumps(allocator, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (run_dir / "report.md").write_text(
        "# Formal 1.5B evaluation\n\n"
        f"- Phase: {phase}\n- Algorithm: {algorithm}\n- Seed: {seed}\n"
        "- Test evidence is frozen and never used for tuning or checkpoint selection.\n",
        encoding="utf-8",
    )
    for name in REQUIRED_EVALUATION_FILES:
        if not (run_dir / name).exists():
            raise FormalRuntimeError(f"formal evaluation finalization missed {name}")
    return {"phase": phase, "seed": seed, "algorithm": algorithm, **aggregate}
