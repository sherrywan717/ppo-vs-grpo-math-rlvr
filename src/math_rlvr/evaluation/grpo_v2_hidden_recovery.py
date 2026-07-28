"""CPU-only supplemental finalization of the immutable Stage S.2 Base evidence."""

from __future__ import annotations

import csv
import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from math_rlvr.artifacts.manager import atomic_text
from math_rlvr.evaluation.grpo_v2_dev_model_runtime import _flat_metric_rows
from math_rlvr.evaluation.grpo_v2_dev_runtime import write_csv
from math_rlvr.evaluation.grpo_v2_hidden_model_runtime import _pass_k_rows, _pass_k_summary
from math_rlvr.evaluation.grpo_v2_hidden_runtime import (
    aggregate_hidden_candidate0,
    build_hidden_plan,
    load_hidden_contract,
    validate_hidden_rows,
)
from math_rlvr.grpo_v2_contract import validate_model_evaluation_ledger

BASE_RUN = Path("/root/autodl-tmp/runs/math_rlvr/base_hidden_grpo_v2_seed42_20260728T073339Z")
FAILURE_ARCHIVE = Path(
    "/root/autodl-fs/math-rlvr-backups/base_hidden_grpo_v2_seed42_20260728T073339Z.failure.tar.gz"
)
OUTPUT_DIR = Path("reports/grpo_v2/base_hidden_recovery")
COMPLETIONS_SHA256 = "bb14b3a8fa69e65311c48e5e610e7c08d56abcc09cb9e7765a61f86a51ec073a"
CHECKSUM_MANIFEST_SHA256 = "6f8cfa6c7d0d4b752ccd851d543286cdde1773677fea497d7443b8ac321da07a"
FAILURE_ARCHIVE_SHA256 = "532ac2854ade3374c3725410f509f6092e2508453fbd68522cf1b85c9660e215"


class HiddenRecoveryError(RuntimeError):
    """The immutable Base evidence cannot be recovered without changing science."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _snapshot_files(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): file_sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _validate_source(run_dir: Path, archive: Path) -> dict[str, Any]:
    if run_dir != BASE_RUN or archive != FAILURE_ARCHIVE:
        raise HiddenRecoveryError("Base recovery requires exact immutable source paths")
    completions = run_dir / "completions.jsonl"
    checksums = run_dir / "checksums.sha256"
    if file_sha256(completions) != COMPLETIONS_SHA256:
        raise HiddenRecoveryError("Base completion evidence SHA mismatch")
    if file_sha256(checksums) != CHECKSUM_MANIFEST_SHA256:
        raise HiddenRecoveryError("Base checksum manifest SHA mismatch")
    if file_sha256(archive) != FAILURE_ARCHIVE_SHA256:
        raise HiddenRecoveryError("Base failure archive SHA mismatch")
    for line in checksums.read_text().splitlines():
        if not line.strip():
            continue
        expected, relative = line.split("  ", 1)
        path = run_dir / relative
        if not path.is_file() or file_sha256(path) != expected:
            raise HiddenRecoveryError(f"Base source checksum mismatch: {relative}")
    return {
        "completions_sha256": COMPLETIONS_SHA256,
        "checksum_manifest_sha256": CHECKSUM_MANIFEST_SHA256,
        "failure_archive_sha256": FAILURE_ARCHIVE_SHA256,
    }


def _resource_summary(run_dir: Path) -> dict[str, float]:
    rows = list(csv.DictReader((run_dir / "resource_metrics.csv").open()))
    elapsed = float(rows[-1]["elapsed_seconds"])
    gpu_hours = elapsed / 3600
    return {
        "wall_seconds": elapsed,
        "peak_vram_mib": max(float(row["gpu_memory_used_mb"]) for row in rows),
        "mean_gpu_utilization_pct": statistics.fmean(
            float(row["gpu_utilization_pct"]) for row in rows
        ),
        "gpu_hours": gpu_hours,
        "cost_cny": gpu_hours * 8.88,
    }


def _write_figures(output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    figures = output_dir / "figures"
    figures.mkdir()
    metric_rows = list(csv.DictReader((output_dir / "candidate0_metrics.csv").open()))
    overall = next(row for row in metric_rows if row["slice"] == "all")
    names = ["pass@1", "format", "parseable", "truncation"]
    values = [
        float(overall["candidate0_pass_at_1"]),
        float(overall["format_rate"]),
        float(overall["parseable_rate"]),
        float(overall["truncation_rate"]),
    ]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(names, values, color="#7f7f7f")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Rate")
    ax.set_title("Recovered Base candidate-0 metrics (400 problems)")
    fig.tight_layout()
    fig.savefig(figures / "candidate0_metrics.png", dpi=160)
    plt.close(fig)

    pass_rows = list(csv.DictReader((output_dir / "pass_k_summary.csv").open()))
    overall_pass = sorted(
        (row for row in pass_rows if row["slice"] == "overall"),
        key=lambda row: int(row["k"]),
    )
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        [int(row["k"]) for row in overall_pass],
        [float(row["mean"]) for row in overall_pass],
        marker="o",
        color="#7f7f7f",
    )
    ax.set_xticks([1, 4, 10])
    ax.set_ylim(0, 1)
    ax.set_xlabel("k (shared n=10 pool)")
    ax.set_ylabel("Unbiased pass@k")
    ax.set_title("Recovered Base shared-pool inference scaling (100 problems)")
    fig.tight_layout()
    fig.savefig(figures / "unbiased_pass_k.png", dpi=160)
    plt.close(fig)


def recover_base_evidence(
    *,
    run_dir: Path = BASE_RUN,
    archive: Path = FAILURE_ARCHIVE,
    output_dir: Path = OUTPUT_DIR,
) -> dict[str, Any]:
    if output_dir.exists():
        raise HiddenRecoveryError("supplemental recovery output must be non-overwriting")
    source_before = _snapshot_files(run_dir)
    source_identity = _validate_source(run_dir, archive)
    config, contract_identity, public_rows, shared_ids = load_hidden_contract()
    rows = [
        json.loads(line)
        for line in (run_dir / "completions.jsonl").read_text().splitlines()
        if line.strip()
    ]
    plan = build_hidden_plan(public_rows, shared_ids)
    validated = validate_hidden_rows(plan, rows)
    ledger = validate_model_evaluation_ledger(
        validated,
        all_problem_ids={row["problem_id"] for row in public_rows},
        shared_problem_ids=shared_ids,
    )
    generated_tokens = sum(int(row["exact_token_count"]) for row in validated)
    if generated_tokens != 152_567:
        raise HiddenRecoveryError("Base generated-token total mismatch")
    candidate0 = [row for row in validated if row["candidate_index"] == 0]
    candidate_metrics = aggregate_hidden_candidate0(candidate0)
    known = {
        "canonical_correct": 6,
        "format_valid": 31,
        "valid_answer": 28,
        "parseable": 28,
        "eos": 357,
        "truncated": 43,
    }
    actual = {key: sum(bool(row[key]) for row in candidate0) for key in known}
    if actual != known:
        raise HiddenRecoveryError("Base known-count replay mismatch")
    pass_problem_rows = _pass_k_rows(validated, shared_ids)
    pass_summary, pass_summary_rows = _pass_k_summary(pass_problem_rows)
    output_dir.mkdir(parents=True)
    write_csv(output_dir / "candidate0_metrics.csv", _flat_metric_rows(candidate_metrics))
    write_csv(
        output_dir / "per_dataset_metrics.csv",
        [
            row
            for row in _flat_metric_rows(candidate_metrics)
            if row["slice"] in {"all", "gsm8k", "math"}
        ],
    )
    write_csv(
        output_dir / "math_level_metrics.csv",
        [
            row
            for row in _flat_metric_rows(candidate_metrics)
            if row["slice"].startswith("math_level_")
        ],
    )
    write_csv(
        output_dir / "pass_k_per_problem.csv",
        [
            {
                "problem_id": row["problem_id"],
                "dataset": row["dataset"],
                "math_level": row["math_level"],
                "n": row["n"],
                "c": row["c"],
                "candidate_indices": json.dumps(row["candidate_indices"]),
                "candidate_seeds": json.dumps(row["candidate_seeds"]),
                "duplicate_rate": row["duplicate_rate"],
                "generated_tokens": row["generated_tokens"],
                **{f"pass_at_{k}": row["estimates"][str(k)]["float_value"] for k in (1, 4, 10)},
            }
            for row in pass_problem_rows
        ],
    )
    write_csv(output_dir / "pass_k_summary.csv", pass_summary_rows)
    status_rows = []
    for scope, selected in (("candidate0", candidate0), ("all_candidates", validated)):
        for status, count in sorted(Counter(row["verifier_status"] for row in selected).items()):
            status_rows.append(
                {"scope": scope, "status": status, "count": count, "denominator": len(selected)}
            )
    write_csv(output_dir / "status_distribution.csv", status_rows)
    write_csv(
        output_dir / "truncation_analysis.csv",
        [
            {
                "scope": scope,
                "truncated": sum(bool(row["truncated"]) for row in selected),
                "denominator": len(selected),
                "rate": sum(bool(row["truncated"]) for row in selected) / len(selected),
            }
            for scope, selected in (("candidate0", candidate0), ("all_candidates", validated))
        ],
    )
    comparisons = {
        name: {
            "value": None,
            "available": False,
            "reason": "remaining_models_not_executed",
        }
        for name in (
            "base_vs_old_grpo_v1",
            "base_vs_warmstart",
            "base_vs_grpo_v2",
            "mcnemar",
            "four_model_aggregate",
            "four_model_error_analysis",
            "comparative_case_studies",
        )
    }
    summary = {
        "schema_version": 1,
        "original_run_id": run_dir.name,
        "original_run_status": "engineering_failure_after_generation_during_metric_finalization",
        "recovery_status": "scientifically_complete_with_recovered_metric_finalization",
        "source_identity": source_identity,
        "contract_identity": contract_identity,
        "ledger": {**ledger, "generated_tokens": generated_tokens},
        "candidate0_metrics": candidate_metrics,
        "pass_k_summary": pass_summary,
        "resource_summary": _resource_summary(run_dir),
        "comparisons": comparisons,
    }
    atomic_text(
        output_dir / "recovered_summary.json", json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    atomic_text(
        output_dir / "report.md",
        "# Recovered Base hidden-test metric finalization\n\n"
        "- Original run remains "
        "`engineering_failure_after_generation_during_metric_finalization`.\n"
        "- Composite status: `scientifically_complete_with_recovered_metric_finalization`.\n"
        f"- Candidate-0 canonical accuracy: {actual['canonical_correct']}/400.\n"
        f"- Generated tokens: {generated_tokens:,}.\n"
        "- Metrics were derived only from the immutable 1,300-row evidence; "
        "no generation occurred.\n"
        "- Four-model comparisons remain unavailable because three models have not run.\n",
    )
    _write_figures(output_dir)
    lines = []
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path.name != "checksums.sha256":
            lines.append(f"{file_sha256(path)}  {path.relative_to(output_dir)}")
    atomic_text(output_dir / "checksums.sha256", "\n".join(lines) + "\n")
    if _snapshot_files(run_dir) != source_before:
        raise HiddenRecoveryError("Base primary evidence changed during recovery")
    return summary


if __name__ == "__main__":
    print(json.dumps(recover_base_evidence(), sort_keys=True))
