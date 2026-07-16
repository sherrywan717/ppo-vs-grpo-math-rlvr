#!/usr/bin/env python3
"""Regenerate formal 1.5B figures exclusively from persisted CSV artifacts."""

from __future__ import annotations

import argparse
import csv
from collections.abc import Iterable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _float_points(
    rows: Iterable[dict[str, str]], x_key: str, y_key: str
) -> list[tuple[float, float]]:
    return [
        (float(row[x_key]), float(row[y_key]))
        for row in rows
        if row.get(x_key) not in {None, ""} and row.get(y_key) not in {None, ""}
    ]


def _line_plot(
    output: Path,
    name: str,
    rows: list[dict[str, str]],
    x_key: str,
    y_keys: tuple[str, ...],
    ylabel: str,
) -> bool:
    fig, axis = plt.subplots(figsize=(8, 4.5))
    plotted = False
    for key in y_keys:
        points = _float_points(rows, x_key, key)
        if points:
            axis.plot([x for x, _ in points], [y for _, y in points], marker="o", label=key)
            plotted = True
    if not plotted:
        plt.close(fig)
        return False
    axis.set(xlabel=x_key, ylabel=ylabel, title="Formal 1.5B PPO vs GRPO")
    axis.grid(alpha=0.25)
    if len(y_keys) > 1:
        axis.legend()
    fig.tight_layout()
    fig.savefig(output / f"{name}.png", dpi=180)
    fig.savefig(output / f"{name}.svg")
    plt.close(fig)
    return True


def generate(input_dir: Path, output: Path) -> dict[str, list[str]]:
    output.mkdir(parents=True, exist_ok=True)
    training = _rows(input_dir / "training_metrics.csv")
    evaluation = _rows(input_dir / "aggregate_metrics.csv")
    resources = _rows(input_dir / "resource_metrics.csv")
    specs = (
        ("reward_vs_update", training, "update", ("reward_mean",), "reward"),
        (
            "validation_pass_vs_update",
            training,
            "update",
            ("canonical_validation_pass_rate",),
            "rate",
        ),
        (
            "format_valid_answer_vs_update",
            training,
            "update",
            ("format_accuracy", "valid_answer_rate"),
            "rate",
        ),
        (
            "policy_value_loss",
            training,
            "update",
            ("policy_loss", "value_loss", "loss"),
            "loss",
        ),
        (
            "kl_entropy_grad_norm",
            training,
            "update",
            ("kl", "entropy", "grad_norm"),
            "metric",
        ),
        (
            "completion_length",
            training,
            "update",
            ("mean_completion_length", "truncation_rate"),
            "tokens / rate",
        ),
        (
            "baseline_post_pass",
            evaluation,
            "series_index",
            ("pass_at_1", "pass_at_4"),
            "rate",
        ),
        (
            "gsm8k_math500_domain",
            evaluation,
            "series_index",
            ("gsm8k_pass_at_1", "math500_pass_at_1"),
            "rate",
        ),
        (
            "math500_levels",
            evaluation,
            "series_index",
            tuple(f"math500_level_{level}_pass_at_1" for level in range(1, 6)),
            "rate",
        ),
        (
            "confidence_intervals",
            evaluation,
            "series_index",
            ("paired_delta", "bootstrap_ci_low", "bootstrap_ci_high"),
            "paired delta",
        ),
        (
            "vram_utilization_timeline",
            resources,
            "elapsed_seconds",
            ("nvidia_smi_memory_mib", "gpu_utilization_pct"),
            "MiB / percent",
        ),
        (
            "wall_time_cost",
            resources,
            "series_index",
            ("wall_time_seconds", "gpu_hours", "cost_cny"),
            "resource",
        ),
        (
            "algorithm_seed_comparison",
            evaluation,
            "series_index",
            ("reward_mean", "pass_at_1", "pass_at_4"),
            "metric",
        ),
    )
    made: list[str] = []
    unavailable: list[str] = []
    for name, rows, x_key, y_keys, ylabel in specs:
        destination = made if _line_plot(output, name, rows, x_key, y_keys, ylabel) else unavailable
        destination.append(name)
    return {"made": made, "unavailable": unavailable}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(generate(args.input_dir, args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
