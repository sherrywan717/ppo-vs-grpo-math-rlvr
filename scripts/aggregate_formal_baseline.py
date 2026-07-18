#!/usr/bin/env python3
# ruff: noqa: E501
"""Aggregate the two frozen formal baseline runs from persisted artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXPECTED_COMPLETIONS = 800
EOS_TOKEN_ID = 151645
EXCLUDED_FAILURE_RUNS = (
    "baseline_formal_1p5b_seed42_20260718T114907Z",
    "baseline_formal_1p5b_seed42_20260718T120909Z",
)
GIT_SAFE_RUN_FILES = (
    "aggregate_metrics.csv",
    "aggregate_metrics.json",
    "completions.jsonl",
    "evaluation_manifest.json",
    "final_summary.json",
    "per_problem_metrics.csv",
    "pytorch_allocator.json",
    "report.md",
    "resolved_config.yaml",
    "resource_metrics.csv",
    "resource_summary.json",
    "run_manifest.json",
    "verifier_status.csv",
)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _sample_std(values: list[float]) -> float | None:
    return statistics.stdev(values) if len(values) > 1 else None


def _ratio(rows: list[dict[str, Any]], key: str) -> float:
    return statistics.fmean(bool(row[key]) for row in rows)


def _load_run(run_dir: Path) -> dict[str, Any]:
    manifest = _read_json(run_dir / "run_manifest.json")
    final = _read_json(run_dir / "final_summary.json")
    aggregate = _read_json(run_dir / "aggregate_metrics.json")["aggregate"]
    resource = _read_json(run_dir / "resource_summary.json")
    allocator = _read_json(run_dir / "pytorch_allocator.json")
    completions = _read_jsonl(run_dir / "completions.jsonl")
    problems = _read_csv(run_dir / "per_problem_metrics.csv")
    resources = _read_csv(run_dir / "resource_metrics.csv")

    run_id = str(manifest["run_id"])
    if run_id in EXCLUDED_FAILURE_RUNS:
        raise ValueError(f"refusing excluded failure run: {run_id}")
    if final.get("status") != "success":
        raise ValueError(f"run is not successful: {run_id}")
    if len(completions) != EXPECTED_COMPLETIONS:
        raise ValueError(f"{run_id}: expected 800 completions, found {len(completions)}")
    if len({row["pair_key"] for row in completions}) != EXPECTED_COMPLETIONS:
        raise ValueError(f"{run_id}: pair keys are incomplete or duplicated")
    if len(problems) != 400:
        raise ValueError(f"{run_id}: expected 400 per-problem rows, found {len(problems)}")
    if not resources:
        raise ValueError(f"{run_id}: resource metrics are empty")

    return {
        "run_dir": run_dir,
        "run_id": run_id,
        "seed": int(manifest["seed"]),
        "aggregate": aggregate,
        "resource": resource,
        "allocator": allocator,
        "completions": completions,
        "problems": problems,
        "wall_time_seconds": float(resources[-1]["elapsed_seconds"]),
    }


def _problem_slices(problems: list[dict[str, str]]) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in problems:
        groups["overall"].append(row)
        groups[row["domain"]].append(row)
        if row["domain"] == "math500":
            groups[f"math500_level_{row['difficulty']}"].append(row)

    result: dict[str, dict[str, Any]] = {}
    for name, rows in groups.items():
        pass4_values = [float(row["pass_at_4"]) for row in rows if row["pass_at_4"] != ""]
        result[name] = {
            "problem_count": len(rows),
            "pass4_problem_count": len(pass4_values),
            "pass_at_1": statistics.fmean(float(row["sampled_pass_at_1"]) for row in rows),
            "pass_at_4": _mean(pass4_values),
        }
    return result


def _baseline_row(run: dict[str, Any]) -> dict[str, Any]:
    completions = run["completions"]
    aggregate = run["aggregate"]
    slices = _problem_slices(run["problems"])
    status = Counter(row["canonical_status"] for row in completions)
    lengths = [int(row["exact_token_count"]) for row in completions]
    generated_tokens = sum(lengths)
    eos_count = sum(
        bool(row["completion_ids"]) and row["completion_ids"][-1] == EOS_TOKEN_ID
        for row in completions
    )
    wall_time = run["wall_time_seconds"]
    return {
        "run_id": run["run_id"],
        "seed": run["seed"],
        "series_index": 0 if run["seed"] == 42 else 1,
        "completion_count": len(completions),
        "unique_problem_count": len(run["problems"]),
        "generated_tokens": generated_tokens,
        "greedy_accuracy": "",
        "greedy_accuracy_available": "false",
        "greedy_accuracy_unavailable_reason": aggregate["greedy_accuracy_unavailable_reason"],
        "sampled_pass_at_1": aggregate["sampled_pass_at_1"],
        "pass_at_4": aggregate["pass_at_4"],
        "gsm8k_pass_at_1": slices["gsm8k"]["pass_at_1"],
        "gsm8k_pass_at_4": slices["gsm8k"]["pass_at_4"],
        "math500_pass_at_1": slices["math500"]["pass_at_1"],
        "math500_pass_at_4": slices["math500"]["pass_at_4"],
        "format_accuracy": aggregate["format_accuracy"],
        "valid_answer_rate": aggregate["valid_answer_rate"],
        "canonical_correctness": aggregate["canonical_correctness"],
        "reward_mean": aggregate["reward_mean"],
        "completion_length_mean": statistics.fmean(lengths),
        "completion_length_std_population": statistics.pstdev(lengths),
        "eos_rate": eos_count / len(completions),
        "truncation_rate": aggregate["truncation_rate"],
        "format_error_count": status["format_error"],
        "parse_error_count": status["parse_error"],
        "wrong_answer_count": status["wrong_answer"],
        "verified_pass_count": status["verified_pass"],
        "wall_time_seconds": wall_time,
        "tokens_per_second": generated_tokens / wall_time,
        "pytorch_peak_allocated_mib": run["allocator"]["max_memory_allocated"]["mib"],
        "pytorch_peak_reserved_mib": run["allocator"]["max_memory_reserved"]["mib"],
        "nvidia_smi_peak_vram_mib": run["resource"]["peak_vram_mb"],
        "mean_gpu_utilization_pct": run["resource"]["mean_gpu_utilization"],
        "gpu_hours": run["resource"]["gpu_hours"],
        "cost_cny": run["resource"]["estimated_cost_cny"],
    }


def _per_problem_rows(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for run in runs:
        for row in run["problems"]:
            output.append(
                {
                    "run_id": run["run_id"],
                    "seed": run["seed"],
                    "problem_id": row["problem_id"],
                    "domain": row["domain"],
                    "math500_level": row["difficulty"] if row["domain"] == "math500" else "",
                    "sampled_pass_at_1": row["sampled_pass_at_1"],
                    "pass_at_4": row["pass_at_4"],
                    "format_valid": row["format_valid"],
                    "valid_answer": row["valid_answer"],
                    "canonical_correct": row["canonical_correct"],
                }
            )
    return output


def _level_rows(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for run in runs:
        slices = _problem_slices(run["problems"])
        for level in range(1, 6):
            values = slices[f"math500_level_{level}"]
            output.append(
                {
                    "run_id": run["run_id"],
                    "seed": run["seed"],
                    "math500_level": level,
                    **values,
                }
            )
    return output


def _resource_rows(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for run in runs:
        generated_tokens = sum(row["exact_token_count"] for row in run["completions"])
        rows.append(
            {
                "run_id": run["run_id"],
                "seed": run["seed"],
                "wall_time_seconds": run["wall_time_seconds"],
                "generated_tokens": generated_tokens,
                "tokens_per_second": generated_tokens / run["wall_time_seconds"],
                "pytorch_peak_allocated_mib": run["allocator"]["max_memory_allocated"]["mib"],
                "pytorch_peak_reserved_mib": run["allocator"]["max_memory_reserved"]["mib"],
                "nvidia_smi_peak_vram_mib": run["resource"]["peak_vram_mb"],
                "mean_gpu_utilization_pct": run["resource"]["mean_gpu_utilization"],
                "gpu_hours": run["resource"]["gpu_hours"],
                "cost_cny": run["resource"]["estimated_cost_cny"],
            }
        )
    rows.append(
        {
            "run_id": "total",
            "seed": "",
            "wall_time_seconds": sum(float(row["wall_time_seconds"]) for row in rows),
            "generated_tokens": sum(int(row["generated_tokens"]) for row in rows),
            "tokens_per_second": "",
            "pytorch_peak_allocated_mib": max(
                float(row["pytorch_peak_allocated_mib"]) for row in rows
            ),
            "pytorch_peak_reserved_mib": max(
                float(row["pytorch_peak_reserved_mib"]) for row in rows
            ),
            "nvidia_smi_peak_vram_mib": max(float(row["nvidia_smi_peak_vram_mib"]) for row in rows),
            "mean_gpu_utilization_pct": "",
            "gpu_hours": sum(float(row["gpu_hours"]) for row in rows),
            "cost_cny": sum(float(row["cost_cny"]) for row in rows),
        }
    )
    return rows


def _copy_git_safe_run(run: dict[str, Any], reports_root: Path) -> None:
    destination = reports_root / "runs" / run["run_id"]
    destination.mkdir(parents=True, exist_ok=True)
    for name in GIT_SAFE_RUN_FILES:
        shutil.copy2(run["run_dir"] / name, destination / name)
    lines = [
        f"# Git-safe evidence: {run['run_id']}",
        "",
        "This directory is a Git-safe copy of the successful frozen baseline evidence.",
        f"The complete run remains at `{run['run_dir']}`.",
        "Model cache, weights, checkpoints, credentials, and proxy data are excluded.",
        "",
    ]
    (destination / "README.md").write_text("\n".join(lines), encoding="utf-8")
    inventory = []
    for path in sorted(destination.iterdir()):
        if path.name == "checksums.sha256" or not path.is_file():
            continue
        inventory.append(f"{_sha256(path)}  {path.name}")
    (destination / "checksums.sha256").write_text("\n".join(inventory) + "\n", encoding="utf-8")


def _plot_csvs(metrics_path: Path, levels_path: Path, resources_path: Path, figures: Path) -> None:
    metrics = _read_csv(metrics_path)
    levels = _read_csv(levels_path)
    resources = [row for row in _read_csv(resources_path) if row["run_id"] != "total"]
    figures.mkdir(parents=True, exist_ok=True)
    labels = [f"seed {row['seed']}" for row in metrics]

    for name, overall_key, domain_keys, title in (
        (
            "baseline_pass_at_1",
            "sampled_pass_at_1",
            ("gsm8k_pass_at_1", "math500_pass_at_1"),
            "Frozen baseline sampled pass@1",
        ),
        (
            "baseline_pass_at_4",
            "pass_at_4",
            ("gsm8k_pass_at_4", "math500_pass_at_4"),
            "Frozen baseline pass@4 subset",
        ),
    ):
        fig, axis = plt.subplots(figsize=(8, 4.8))
        x = list(range(len(metrics)))
        width = 0.24
        series = (("overall", overall_key), ("GSM8K", domain_keys[0]), ("MATH500", domain_keys[1]))
        for index, (label, key) in enumerate(series):
            axis.bar(
                [value + (index - 1) * width for value in x],
                [float(row[key]) for row in metrics],
                width,
                label=label,
            )
        axis.set_xticks(x, labels)
        axis.set(
            ylabel="Rate",
            title=title,
            ylim=(0, max(0.14, max(float(row[overall_key]) for row in metrics) * 1.35)),
        )
        axis.grid(axis="y", alpha=0.25)
        axis.legend()
        fig.tight_layout()
        fig.savefig(figures / f"{name}.png", dpi=180)
        plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, 4.8))
    width = 0.36
    level_values = sorted({int(row["math500_level"]) for row in levels})
    for index, metric in enumerate(metrics):
        selected = [row for row in levels if row["seed"] == metric["seed"]]
        selected.sort(key=lambda row: int(row["math500_level"]))
        axis.bar(
            [level + (index - 0.5) * width for level in level_values],
            [float(row["pass_at_1"]) for row in selected],
            width,
            label=f"seed {metric['seed']}",
        )
    axis.set(
        xticks=level_values,
        xlabel="MATH500 level",
        ylabel="Sampled pass@1",
        title="Frozen baseline MATH500 by level",
    )
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(figures / "baseline_math500_by_level.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, 4.8))
    bottom = [0, 0]
    for label, key in (
        ("format error", "format_error_count"),
        ("parse error", "parse_error_count"),
        ("wrong answer", "wrong_answer_count"),
        ("verified pass", "verified_pass_count"),
    ):
        values = [int(row[key]) for row in metrics]
        axis.bar(labels, values, bottom=bottom, label=label)
        bottom = [left + right for left, right in zip(bottom, values, strict=True)]
    axis.set(ylabel="Completion count", title="Canonical verifier status distribution")
    axis.legend()
    fig.tight_layout()
    fig.savefig(figures / "baseline_status_distribution.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, 4.8))
    means = [float(row["completion_length_mean"]) for row in metrics]
    stds = [float(row["completion_length_std_population"]) for row in metrics]
    axis.bar(labels, means, yerr=stds, capsize=5)
    axis.set(
        ylabel="Generated tokens per completion (mean ± population SD)",
        title="Frozen baseline completion length",
    )
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures / "baseline_completion_length.png", dpi=180)
    plt.close(fig)

    fig, left = plt.subplots(figsize=(8, 4.8))
    right = left.twinx()
    x = list(range(len(resources)))
    left.bar(
        [value - 0.18 for value in x],
        [float(row["gpu_hours"]) for row in resources],
        0.36,
        label="GPU-hours",
        color="#4c78a8",
    )
    right.bar(
        [value + 0.18 for value in x],
        [float(row["cost_cny"]) for row in resources],
        0.36,
        label="Cost",
        color="#f58518",
    )
    left.set_xticks(x, [f"seed {row['seed']}" for row in resources])
    left.set_ylabel("GPU-hours")
    right.set_ylabel("Cost (CNY at ¥8.88/GPU-hour)")
    left.set_title("Frozen baseline resource cost")
    left.legend(loc="upper left")
    right.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(figures / "baseline_resource_cost.png", dpi=180)
    plt.close(fig)


def _write_reports(
    output: Path,
    metric_rows: list[dict[str, Any]],
    level_rows: list[dict[str, Any]],
    resource_rows: list[dict[str, Any]],
    runs: list[dict[str, Any]],
) -> None:
    mean_pass1 = statistics.fmean(float(row["sampled_pass_at_1"]) for row in metric_rows)
    std_pass1 = statistics.stdev(float(row["sampled_pass_at_1"]) for row in metric_rows)
    mean_pass4 = statistics.fmean(float(row["pass_at_4"]) for row in metric_rows)
    std_pass4 = statistics.stdev(float(row["pass_at_4"]) for row in metric_rows)
    total = resource_rows[-1]
    lines = [
        "# Qwen 1.5B frozen baseline results",
        "",
        "This report aggregates only the two successful post-amendment baseline runs. The two immutable engineering failures `baseline_formal_1p5b_seed42_20260718T114907Z` and `baseline_formal_1p5b_seed42_20260718T120909Z` are excluded.",
        "",
        "The frozen protocol has no separate greedy completion, so greedy accuracy is `null/unavailable` with reason: `frozen protocol has no separate greedy completion`. Sampled pass@1 uses one sampled completion for all 400 problems. Pass@4 is the fraction with at least one canonical pass among four samples on the fixed 50 GSM8K + 50 MATH500 subset.",
        "",
        "| Seed | Completions | Tokens | Sampled pass@1 | Pass@4 | GSM8K pass@1 | MATH500 pass@1 | Format | Valid answer | EOS | Truncated |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metric_rows:
        lines.append(
            f"| {row['seed']} | {row['completion_count']} | {row['generated_tokens']} | {float(row['sampled_pass_at_1']):.3f} | {float(row['pass_at_4']):.3f} | {float(row['gsm8k_pass_at_1']):.3f} | {float(row['math500_pass_at_1']):.3f} | {float(row['format_accuracy']):.3f} | {float(row['valid_answer_rate']):.3f} | {float(row['eos_rate']):.3f} | {float(row['truncation_rate']):.3f} |"
        )
    lines.extend(
        [
            "",
            f"Across seeds, sampled pass@1 is {mean_pass1:.4f} ± {std_pass1:.4f} sample SD and pass@4 is {mean_pass4:.4f} ± {std_pass4:.4f} sample SD. These two seeds quantify baseline sampling variation; they are not tuning observations.",
            "",
            "## MATH500 levels",
            "",
            "| Seed | Level | Problems | Sampled pass@1 | pass@4 subset problems | Pass@4 |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in level_rows:
        pass4 = "unavailable" if row["pass_at_4"] is None else f"{float(row['pass_at_4']):.3f}"
        lines.append(
            f"| {row['seed']} | {row['math500_level']} | {row['problem_count']} | {float(row['pass_at_1']):.3f} | {row['pass4_problem_count']} | {pass4} |"
        )
    lines.extend(
        [
            "",
            "## Resources",
            "",
            f"Total wall time was {float(total['wall_time_seconds']):.1f} seconds, {float(total['gpu_hours']):.6f} GPU-hours, and ¥{float(total['cost_cny']):.4f} at ¥8.88/GPU-hour. The maximum observed nvidia-smi memory was {float(total['nvidia_smi_peak_vram_mib']):.0f} MiB; PyTorch allocator and nvidia-smi peaks remain separately reported in `metrics/resource_costs.csv`.",
            "",
            "## Figures",
            "",
            "![Sampled pass@1 by seed and domain](figures/baseline_pass_at_1.png)",
            "",
            "*Sampled pass@1 on the frozen 400-problem evaluation set.*",
            "",
            "![Pass@4 by seed and domain](figures/baseline_pass_at_4.png)",
            "",
            "*Pass@4 on the fixed 100-problem subset.*",
            "",
            "![MATH500 level results](figures/baseline_math500_by_level.png)",
            "",
            "*Sampled pass@1 by frozen MATH500 level.*",
            "",
            "![Verifier status distribution](figures/baseline_status_distribution.png)",
            "",
            "*Canonical completion-status counts; format failures are distinct from parseable wrong answers.*",
            "",
            "![Completion length](figures/baseline_completion_length.png)",
            "",
            "*Mean generated completion tokens with population standard deviation.*",
            "",
            "![Resource cost](figures/baseline_resource_cost.png)",
            "",
            "*Measured GPU-hours and cost at the frozen ¥8.88/GPU-hour rate.*",
            "",
        ]
    )
    (output / "01_baseline_results.md").write_text("\n".join(lines), encoding="utf-8")

    error_lines = [
        "# Frozen baseline error analysis",
        "",
        "The analysis uses only the two successful post-amendment runs and does not alter the prompt, reward, verifier, evaluation set, or sampling protocol.",
        "",
    ]
    for run, row in zip(runs, metric_rows, strict=True):
        truncated = [item for item in run["completions"] if item["truncated"]]
        nontruncated = [item for item in run["completions"] if not item["truncated"]]
        error_lines.extend(
            [
                f"## Seed {run['seed']}",
                "",
                f"Of 800 completions, {row['format_error_count']} were strict format failures, {row['parse_error_count']} were parse failures after clearing format, {row['wrong_answer_count']} were parseable canonical wrong answers, and {row['verified_pass_count']} were canonical passes.",
                f"GSM8K sampled pass@1 was {float(row['gsm8k_pass_at_1']):.3f}; MATH500 was {float(row['math500_pass_at_1']):.3f}. Truncated completions were {len(truncated)}/800 with canonical correctness {_ratio(truncated, 'canonical_correct'):.4f}; non-truncated correctness was {_ratio(nontruncated, 'canonical_correct'):.4f}.",
                "",
            ]
        )
    error_lines.extend(
        [
            "## Interpretation",
            "",
            "The dominant failure mode is strict output-format failure, not a verifier infrastructure error. The smaller parse-error group is kept separate from outputs that parse but produce a wrong mathematical answer. MATH500 level results are non-monotonic at this low baseline accuracy, and two seeds are insufficient to interpret small level-to-level differences as a stable capability curve.",
            f"Sampled pass@1 changed from {float(metric_rows[0]['sampled_pass_at_1']):.3f} to {float(metric_rows[1]['sampled_pass_at_1']):.3f}; pass@4 changed from {float(metric_rows[0]['pass_at_4']):.3f} to {float(metric_rows[1]['pass_at_4']):.3f}. This is disclosed seed variation, not a prompt-selection or checkpoint-selection signal. Truncation is material and is reported rather than silently treated as an ordinary full completion.",
            "",
        ]
    )
    (output / "baseline_error_analysis.md").write_text("\n".join(error_lines), encoding="utf-8")

    resource_lines = [
        "# Frozen baseline resource and cost report",
        "",
        "| Run | Seed | Wall seconds | Tokens | Tokens/s | PyTorch peak allocated MiB | PyTorch peak reserved MiB | nvidia-smi peak MiB | Mean GPU util % | GPU-hours | Cost CNY |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in resource_rows[:-1]:
        resource_lines.append(
            f"| {row['run_id']} | {row['seed']} | {float(row['wall_time_seconds']):.1f} | {row['generated_tokens']} | {float(row['tokens_per_second']):.3f} | {float(row['pytorch_peak_allocated_mib']):.1f} | {float(row['pytorch_peak_reserved_mib']):.1f} | {float(row['nvidia_smi_peak_vram_mib']):.0f} | {float(row['mean_gpu_utilization_pct']):.2f} | {float(row['gpu_hours']):.6f} | {float(row['cost_cny']):.4f} |"
        )
    resource_lines.extend(
        [
            "",
            f"Combined: {float(total['wall_time_seconds']):.1f} wall seconds, {int(total['generated_tokens'])} generated tokens, {float(total['gpu_hours']):.6f} GPU-hours, ¥{float(total['cost_cny']):.4f}. Cost uses ¥8.88 per GPU-hour.",
            "",
            "Worker-exit allocator residue is a warning only. Independent post-process checks found 0 MiB and no compute process after both runs.",
            "",
        ]
    )
    (output / "baseline_resource_cost.md").write_text("\n".join(resource_lines), encoding="utf-8")


def aggregate(run_dirs: list[Path], output: Path, reports_root: Path) -> None:
    runs = sorted((_load_run(path) for path in run_dirs), key=lambda run: run["seed"])
    if [run["seed"] for run in runs] != [42, 123]:
        raise ValueError("expected exactly successful baseline seeds 42 and 123")
    metric_rows = [_baseline_row(run) for run in runs]
    problem_rows = _per_problem_rows(runs)
    level_rows = _level_rows(runs)
    resource_rows = _resource_rows(runs)

    metrics_dir = output / "metrics"
    _write_csv(metrics_dir / "baseline_metrics.csv", list(metric_rows[0]), metric_rows)
    _write_csv(metrics_dir / "baseline_per_problem.csv", list(problem_rows[0]), problem_rows)
    _write_csv(metrics_dir / "per_level_results.csv", list(level_rows[0]), level_rows)
    _write_csv(metrics_dir / "resource_costs.csv", list(resource_rows[0]), resource_rows)
    _plot_csvs(
        metrics_dir / "baseline_metrics.csv",
        metrics_dir / "per_level_results.csv",
        metrics_dir / "resource_costs.csv",
        output / "figures",
    )
    _write_reports(output, metric_rows, level_rows, resource_rows, runs)
    for run in runs:
        _copy_git_safe_run(run, reports_root)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs=2, type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reports-root", type=Path, required=True)
    args = parser.parse_args()
    aggregate(args.run_dirs, args.output, args.reports_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
