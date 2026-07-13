#!/usr/bin/env python3
"""Regenerate smoke figures exclusively from persisted CSV metrics."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PLOTS = {
    "reward_curve": ("step", "reward", "Reward"),
    "correctness_and_validity": (
        "step",
        ("correctness", "format_accuracy", "parse_success_rate"),
        "Rate",
    ),
    "policy_loss": ("step", "policy_loss", "Loss"),
    "ppo_value_loss": ("step", "value_loss", "Loss"),
    "kl_curve": ("step", "kl", "KL"),
    "entropy_curve": ("step", "entropy", "Entropy"),
    "generated_tokens": ("step", "cumulative_generated_tokens", "Tokens"),
    "completion_length": ("step", "mean_completion_length", "Tokens"),
}
GPU = {
    "gpu_memory": ("elapsed_seconds", "gpu_memory_used_mb", "MiB"),
    "gpu_utilization": ("elapsed_seconds", "gpu_utilization_pct", "Percent"),
}


def rows(path):
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="") as h:
        return list(csv.DictReader(h))


def plot_one(output, name, data, xkey, ykeys, ylabel, title):
    ykeys = (ykeys,) if isinstance(ykeys, str) else ykeys
    made = False
    fig, ax = plt.subplots(figsize=(7, 4))
    for key in ykeys:
        points = [
            (float(r[xkey]), float(r[key]))
            for r in data
            if r.get(xkey) not in (None, "") and r.get(key) not in (None, "")
        ]
        if points:
            ax.plot([p[0] for p in points], [p[1] for p in points], marker="o", label=key)
            made = True
    if not made:
        plt.close(fig)
        return False
    ax.set_xlabel(xkey)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    if len(ykeys) > 1:
        ax.legend()
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(output / f"{name}.{ext}", dpi=180 if ext == "png" else None)
    plt.close(fig)
    return True


def generate(run_dir, run_id, algorithm, seed):
    output = run_dir / "figures"
    output.mkdir(exist_ok=True)
    title = f"{run_id} | {algorithm} | seed {seed}\nSmoke test — not a benchmark result"
    made = []
    unavailable = []
    metric_rows = rows(run_dir / "metrics.csv")
    gpu_rows = rows(run_dir / "gpu_metrics.csv")
    for name, (x, y, label) in PLOTS.items():
        if name == "ppo_value_loss" and algorithm != "ppo":
            continue
        (made if plot_one(output, name, metric_rows, x, y, label, title) else unavailable).append(
            name
        )
    for name, (x, y, label) in GPU.items():
        (made if plot_one(output, name, gpu_rows, x, y, label, title) else unavailable).append(name)
    return made, unavailable


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_dir", type=Path)
    p.add_argument("--run-id", required=True)
    p.add_argument("--algorithm", required=True)
    p.add_argument("--seed", type=int, required=True)
    a = p.parse_args()
    print(generate(a.run_dir, a.run_id, a.algorithm, a.seed))


if __name__ == "__main__":
    main()
