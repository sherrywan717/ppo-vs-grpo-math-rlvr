#!/usr/bin/env python3
"""Rebuild the public portfolio figures from committed CSV/JSON evidence."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "reports" / "formal_1p5b" / "metrics"
FIGURES = ROOT / "reports" / "formal_1p5b" / "figures"
COLORS = {"base": "#7f7f7f", "ppo": "#1f77b4", "grpo": "#ff7f0e"}


def read_csv(name: str) -> list[dict[str, str]]:
    with (METRICS / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(name: str) -> dict[str, object]:
    return json.loads((METRICS / name).read_text(encoding="utf-8"))


def save(fig: plt.Figure, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIGURES / name, dpi=180, bbox_inches="tight")
    plt.close(fig)


def final_pass_metrics() -> None:
    rows = [
        row for row in read_csv("seed42_final_comparison_metrics.csv") if row["slice"] == "overall"
    ]
    metrics = ["sampled_pass_at_1", "independent_pass_at_4"]
    labels = ["Sampled pass@1\n400 problems × 1", "Independent pass@4\n100 problems × 4"]
    values = {
        algorithm: [
            100 * float(next(row[algorithm] for row in rows if row["metric"] == metric))
            for metric in metrics
        ]
        for algorithm in COLORS
    }
    x = np.arange(len(metrics))
    width = 0.23
    fig, axis = plt.subplots(figsize=(8.4, 4.8))
    for offset, algorithm in zip((-width, 0, width), COLORS, strict=True):
        bars = axis.bar(
            x + offset,
            values[algorithm],
            width,
            label=algorithm.upper(),
            color=COLORS[algorithm],
        )
        axis.bar_label(bars, fmt="%.2f%%", padding=3)
    axis.set(
        title="Seed 42 Held-out Test: Independent Candidate Pools",
        ylabel="Pass rate (%)",
        xticks=x,
        xticklabels=labels,
    )
    axis.set_ylim(0, 17)
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    save(fig, "portfolio_final_pass_metrics.png")


def paired_transitions() -> None:
    comparisons = [
        row
        for row in read_json("seed42_final_paired_summary.json")["comparisons"]
        if row["pool"] == "pass1"
    ]
    labels = ["Base → GRPO", "PPO → GRPO"]
    improved = [
        row["transitions"].get(
            "base_fail_to_grpo_pass", row["transitions"].get("ppo_fail_to_grpo_pass")
        )
        for row in comparisons
    ]
    regressed = [
        row["transitions"].get(
            "base_pass_to_grpo_fail", row["transitions"].get("ppo_pass_to_grpo_fail")
        )
        for row in comparisons
    ]
    x = np.arange(2)
    fig, axis = plt.subplots(figsize=(8.4, 4.8))
    bars_a = axis.bar(x - 0.18, improved, 0.36, label="Fail → GRPO pass", color=COLORS["grpo"])
    bars_b = axis.bar(x + 0.18, regressed, 0.36, label="Pass → GRPO fail", color="#a94442")
    axis.bar_label(bars_a, padding=3)
    axis.bar_label(bars_b, padding=3)
    axis.set(
        title="Seed 42 Paired pass@1 Transitions (400 Problems)",
        ylabel="Problem count",
        xticks=x,
        xticklabels=labels,
    )
    axis.set_ylim(0, 19)
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    save(fig, "portfolio_paired_transitions.png")


def validation_curves() -> None:
    rows = read_csv("four_run_validation_metrics.csv")
    fig, axis = plt.subplots(figsize=(8.4, 4.8))
    for algorithm in ("ppo", "grpo"):
        for seed, marker in (("42", "o"), ("123", "s")):
            selected = [
                row for row in rows if row["algorithm"] == algorithm and row["seed"] == seed
            ]
            x = [int(row["checkpoint_step"]) for row in selected]
            y = [100 * float(row["sampled_pass_at_1"]) for row in selected]
            axis.plot(
                x,
                y,
                marker=marker,
                color=COLORS[algorithm],
                linestyle="-" if seed == "42" else "--",
                label=f"{algorithm.upper()} seed {seed}",
            )
            for step, rate, row in zip(x, y, selected, strict=True):
                axis.annotate(
                    f"{row['correct_count']}/64",
                    (step, rate),
                    xytext=(0, 7),
                    textcoords="offset points",
                    ha="center",
                    fontsize=8,
                )
    axis.set(
        title="Frozen 64-problem Validation: Single-candidate pass@1",
        xlabel="Checkpoint update",
        ylabel="Correct (%)",
        xticks=[8, 16, 24, 32],
    )
    axis.set_ylim(0, 12)
    axis.grid(alpha=0.25)
    axis.legend(ncol=2)
    save(fig, "portfolio_validation_curves.png")


def training_curves() -> None:
    rows = read_csv("four_run_training_metrics.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharex=True)
    for algorithm in ("ppo", "grpo"):
        for seed in ("42", "123"):
            selected = [
                row for row in rows if row["algorithm"] == algorithm and row["seed"] == seed
            ]
            style = "-" if seed == "42" else "--"
            label = f"{algorithm.upper()} seed {seed}"
            tokens = [float(row["training_tokens"]) for row in selected]
            axes[0].plot(
                tokens,
                [float(row["reward_mean"]) for row in selected],
                color=COLORS[algorithm],
                linestyle=style,
                label=label,
            )
            axes[1].plot(
                tokens,
                [100 * float(row["canonical_pass_rate"]) for row in selected],
                color=COLORS[algorithm],
                linestyle=style,
                label=label,
            )
    axes[0].set(
        title="Training reward", xlabel="Cumulative rollout tokens", ylabel="Mean shaped reward"
    )
    axes[1].set(
        title="Training canonical pass",
        xlabel="Cumulative rollout tokens",
        ylabel="Canonical pass (%)",
    )
    for axis in axes:
        axis.grid(alpha=0.25)
    axes[1].legend(fontsize=8)
    fig.suptitle("Four Formal Runs: 32 Updates / 512 Completions Each")
    save(fig, "portfolio_training_curves.png")


def reward_group_variance() -> None:
    fig, axis = plt.subplots(figsize=(8.4, 4.8))
    for seed, name in (
        (42, "grpo_reward_group_statistics.csv"),
        (123, "grpo_seed123_reward_group_statistics.csv"),
    ):
        rows = [row for row in read_csv(name) if row["scope"] == "update"]
        axis.plot(
            [int(row["update"]) for row in rows],
            [100 * float(row["nonzero_variance_group_fraction"]) for row in rows],
            color=COLORS["grpo"],
            linestyle="-" if seed == 42 else "--",
            label=f"GRPO seed {seed}",
        )
    axis.set(
        title="GRPO Reward-group Learning Signal (4 Groups per Update)",
        xlabel="Update",
        ylabel="Groups with nonzero reward variance (%)",
    )
    axis.set_ylim(0, 105)
    axis.grid(alpha=0.25)
    axis.legend()
    save(fig, "portfolio_reward_group_variance.png")


def final_behavior_metrics() -> None:
    baseline = next(row for row in read_csv("baseline_metrics.csv") if row["seed"] == "42")
    ppo = read_json("ppo_seed42_final_metrics.json")["overall"]
    grpo = read_json("grpo_seed42_final_metrics.json")["overall"]
    metrics = ["format_valid", "parseable", "truncation"]
    values = {
        "base": [
            float(baseline["format_accuracy"]),
            float(baseline["valid_answer_rate"]),
            float(baseline["truncation_rate"]),
        ],
        "ppo": [
            float(ppo["format_valid_rate"]),
            float(ppo["parseable_rate"]),
            float(ppo["truncation_rate"]),
        ],
        "grpo": [
            float(grpo["format_valid_rate"]),
            float(grpo["parseable_rate"]),
            float(grpo["truncation_rate"]),
        ],
    }
    x = np.arange(len(metrics))
    width = 0.23
    fig, axis = plt.subplots(figsize=(8.4, 4.8))
    for offset, algorithm in zip((-width, 0, width), COLORS, strict=True):
        axis.bar(
            x + offset,
            [100 * value for value in values[algorithm]],
            width,
            color=COLORS[algorithm],
            label=algorithm.upper(),
        )
    axis.set(
        title="Seed 42 Held-out Output Behavior (All 800 Candidates)",
        ylabel="Rate (%)",
        xticks=x,
        xticklabels=["Format valid", "Parseable", "Truncated"],
    )
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    save(fig, "portfolio_format_parseable_truncation.png")


def resources() -> None:
    rows = read_csv("four_run_resources.csv")
    labels = [f"{row['algorithm'].upper()} {row['seed']}" for row in rows]
    colors = [COLORS[row["algorithm"]] for row in rows]
    x = np.arange(len(rows))

    fig, axis = plt.subplots(figsize=(8.4, 4.8))
    bars = axis.bar(x, [float(row["peak_vram_mb"]) / 1024 for row in rows], color=colors)
    axis.bar_label(bars, fmt="%.1f GiB", padding=3)
    axis.set(
        title="Formal Training + Checkpoint Validation Peak VRAM",
        ylabel="Peak VRAM (GiB)",
        xticks=x,
        xticklabels=labels,
    )
    axis.set_ylim(0, 58)
    axis.grid(axis="y", alpha=0.25)
    save(fig, "portfolio_peak_vram.png")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    specs = (
        ("wall_time_seconds", 60, "Wall time (min)"),
        ("gpu_hours", 1, "GPU-hours"),
        ("cost_cny", 1, "Cost (CNY)"),
    )
    for axis, (field, divisor, title) in zip(axes, specs, strict=True):
        values = [float(row[field]) / divisor for row in rows]
        bars = axis.bar(x, values, color=colors)
        axis.bar_label(bars, fmt="%.2f", padding=3, fontsize=8)
        axis.set(title=title, xticks=x, xticklabels=labels)
        axis.tick_params(axis="x", rotation=30)
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("Formal Training + Checkpoint Validation Resources")
    save(fig, "portfolio_resource_costs.png")


def math_levels() -> None:
    rows = [
        row
        for row in read_csv("seed42_final_comparison_metrics.csv")
        if row["slice"].startswith("math500_level_") and row["metric"] == "sampled_pass_at_1"
    ]
    x = np.arange(1, 6)
    width = 0.23
    fig, axis = plt.subplots(figsize=(8.4, 4.8))
    for offset, algorithm in zip((-width, 0, width), COLORS, strict=True):
        values = [100 * float(row[algorithm]) for row in rows]
        bars = axis.bar(x + offset, values, width, color=COLORS[algorithm], label=algorithm.upper())
        for bar, value in zip(bars, values, strict=True):
            axis.annotate(
                f"{round(value * 0.4):.0f}/40",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                fontsize=7,
            )
    axis.set(
        title="Seed 42 MATH500 pass@1 by Difficulty (40 Problems per Level)",
        xlabel="MATH500 level",
        ylabel="Pass@1 (%)",
        xticks=x,
    )
    axis.set_ylim(0, 20)
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    save(fig, "portfolio_math500_levels.png")


def reward_status() -> None:
    ppo_rows = read_csv("ppo_seed42_final_status_distribution.csv")
    grpo_rows = read_csv("grpo_seed42_final_status_distribution.csv")
    statuses = ["format_error", "parse_error", "wrong_answer", "verified_pass"]
    values: dict[str, list[float]] = {}
    source_models = {"base": "base", "ppo": "ppo_checkpoint32"}
    for algorithm, source_model in source_models.items():
        values[algorithm] = [
            100
            * float(
                next(
                    row["rate"]
                    for row in ppo_rows
                    if row["model"] == source_model
                    and row["slice"] == "overall"
                    and row["status"] == status
                )
            )
            for status in statuses
        ]
    values["grpo"] = [
        100
        * float(
            next(
                row["rate"]
                for row in grpo_rows
                if row["slice"] == "overall" and row["canonical_status"] == status
            )
        )
        for status in statuses
    ]
    x = np.arange(len(statuses))
    width = 0.23
    fig, axis = plt.subplots(figsize=(9.2, 4.8))
    for offset, algorithm in zip((-width, 0, width), COLORS, strict=True):
        axis.bar(
            x + offset, values[algorithm], width, color=COLORS[algorithm], label=algorithm.upper()
        )
    axis.set(
        title="Seed 42 Held-out Canonical RewardStatus (All 800 Candidates)",
        ylabel="Candidate rate (%)",
        xticks=x,
        xticklabels=["Format error", "Parse error", "Wrong answer", "Verified pass"],
    )
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    save(fig, "portfolio_reward_status.png")


def main() -> int:
    FIGURES.mkdir(parents=True, exist_ok=True)
    final_pass_metrics()
    paired_transitions()
    validation_curves()
    training_curves()
    reward_group_variance()
    final_behavior_metrics()
    resources()
    math_levels()
    reward_status()
    print("rebuilt 10 portfolio figures from committed CSV/JSON evidence")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
