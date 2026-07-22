#!/usr/bin/env python3
"""Build the GitHub payload manifest and checksums from the Git index."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "release"
CONTROL = {
    "release/portfolio_v1_manifest.json",
    "release/portfolio_v1_manifest.csv",
    "release/checksums.sha256",
}
FIGURE_SOURCES = {
    "portfolio_final_pass_metrics.png": (
        "reports/formal_1p5b/metrics/seed42_final_comparison_metrics.csv"
    ),
    "portfolio_paired_transitions.png": (
        "reports/formal_1p5b/metrics/seed42_final_paired_summary.json"
    ),
    "portfolio_validation_curves.png": (
        "reports/formal_1p5b/metrics/four_run_validation_metrics.csv"
    ),
    "portfolio_training_curves.png": ("reports/formal_1p5b/metrics/four_run_training_metrics.csv"),
    "portfolio_reward_group_variance.png": (
        "reports/formal_1p5b/metrics/grpo_reward_group_statistics.csv; "
        "reports/formal_1p5b/metrics/grpo_seed123_reward_group_statistics.csv"
    ),
    "portfolio_format_parseable_truncation.png": (
        "reports/formal_1p5b/metrics/baseline_metrics.csv; "
        "reports/formal_1p5b/metrics/ppo_seed42_final_metrics.json; "
        "reports/formal_1p5b/metrics/grpo_seed42_final_metrics.json"
    ),
    "portfolio_peak_vram.png": "reports/formal_1p5b/metrics/four_run_resources.csv",
    "portfolio_resource_costs.png": "reports/formal_1p5b/metrics/four_run_resources.csv",
    "portfolio_math500_levels.png": (
        "reports/formal_1p5b/metrics/seed42_final_comparison_metrics.csv"
    ),
    "portfolio_reward_status.png": (
        "reports/formal_1p5b/metrics/ppo_seed42_final_status_distribution.csv; "
        "reports/formal_1p5b/metrics/grpo_seed42_final_status_distribution.csv"
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tracked_paths() -> list[str]:
    raw = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT)
    return sorted(path.decode("utf-8") for path in raw.split(b"\0") if path)


def artifact_type(path: str) -> str:
    suffix = Path(path).suffix.lower()
    if suffix in {".png", ".svg"}:
        return "figure"
    if suffix == ".csv":
        return "tabular_evidence"
    if suffix == ".json":
        return "machine_readable_evidence"
    if suffix in {".md", ".rst"}:
        return "documentation"
    if suffix in {".py", ".sh"}:
        return "source_or_script"
    if path.startswith("configs/") or suffix in {".yaml", ".yml", ".toml"}:
        return "configuration"
    return "project_file"


def source_run(path: str) -> str:
    if path.startswith("reports/runs/"):
        parts = Path(path).parts
        return parts[2] if len(parts) > 2 else "unavailable"
    if path.startswith("reports/formal_1p5b/"):
        return "multiple_formal_runs_or_contract"
    if path.startswith("reports/pilot_0p5b/"):
        return "matched_0p5b_pilot"
    return "not_applicable"


def scientific_status(path: str) -> str:
    if path.startswith("reports/formal_1p5b/"):
        return "mixed_see_run_registry"
    if path.startswith("reports/runs/"):
        return "see_embedded_run_manifest"
    if path.startswith("reports/"):
        return "historical_or_supporting_evidence"
    return "not_scientific_artifact"


def main() -> int:
    RELEASE.mkdir(exist_ok=True)
    records = []
    for path in tracked_paths():
        if path in CONTROL:
            continue
        absolute = ROOT / path
        if not absolute.is_file():
            continue
        records.append(
            {
                "path": path,
                "size_bytes": absolute.stat().st_size,
                "sha256": sha256(absolute),
                "artifact_type": artifact_type(path),
                "source_run": source_run(path),
                "scientific_status": scientific_status(path),
                "tracked": True,
                "public": True,
                "regenerated_from": FIGURE_SOURCES.get(absolute.name, "not_applicable"),
                "notes": "Git-safe payload; control files use release/checksums.sha256"
                if path.startswith("release/")
                else "",
            }
        )

    payload = {
        "schema_version": 1,
        "release": "portfolio_v1",
        "tag": "v0.1.0-formal-rlvr",
        "repository": "https://github.com/sherrywan717/ppo-vs-grpo-math-rlvr.git",
        "entry_count": len(records),
        "total_size_bytes": sum(record["size_bytes"] for record in records),
        "records": records,
        "control_file_integrity": {
            "manifest_json": "hashed by release/checksums.sha256",
            "manifest_csv": "hashed by release/checksums.sha256",
            "checksums": "self-hash intentionally impossible; validate every listed payload line",
        },
    }
    json_path = RELEASE / "portfolio_v1_manifest.json"
    csv_path = RELEASE / "portfolio_v1_manifest.csv"
    checksum_path = RELEASE / "checksums.sha256"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    fields = list(records[0])
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)

    checksum_targets = [record["path"] for record in records]
    checksum_targets.extend(
        ["release/portfolio_v1_manifest.json", "release/portfolio_v1_manifest.csv"]
    )
    lines = [f"{sha256(ROOT / path)}  {path}" for path in sorted(checksum_targets)]
    checksum_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"manifested {len(records)} payload files ({payload['total_size_bytes']} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
