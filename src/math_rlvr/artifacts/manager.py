"""Atomic, secret-aware run artifact management."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import yaml

SECRET_PATTERN = re.compile(
    r"(?:HF_TOKEN|GITHUB_TOKEN|OPENAI_API_KEY|auth\.json|proxy[^\s]*://[^\s:@]+:[^\s@]+@|\.env)",
    re.I,
)
RUN_ROOT = Path("/root/autodl-tmp/runs/math_rlvr")
REPORT_ROOT = Path("reports/runs")


def atomic_bytes(path: Path, data: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp.write_bytes(data)
    os.replace(temp, path)


def atomic_text(path: Path, text: str):
    atomic_bytes(path, text.encode())


def safe_text(text: str):
    if SECRET_PATTERN.search(text):
        raise ValueError("secret-like content rejected")
    return text


def make_run_id(stage, algorithm, model, seed, now=None):
    stamp = (now or datetime.now(UTC)).strftime("%Y%m%d-%H%M%S")
    model_slug = model.rsplit("/", 1)[-1].lower().replace("-instruct", "")
    return f"{stamp}_{stage}_{algorithm}_{model_slug}_seed{seed}"


class ArtifactManager:
    def __init__(
        self,
        stage,
        algorithm,
        model,
        seed,
        command,
        config,
        run_id=None,
        *,
        create_checkpoints=True,
    ):
        self.run_id = run_id or make_run_id(stage, algorithm, model, seed)
        self.run_dir = RUN_ROOT / self.run_id
        self.report_dir = REPORT_ROOT / self.run_id
        directories = ("checkpoints", "figures") if create_checkpoints else ("figures",)
        for name in directories:
            (self.run_dir / name).mkdir(
                parents=True, exist_ok=False if name == "checkpoints" else True
            )
        self._touch_required()
        self.write_json(
            "run_manifest.json",
            {
                "run_id": self.run_id,
                "stage": stage,
                "algorithm": algorithm,
                "model": model,
                "seed": seed,
                "created_at": datetime.now(UTC).isoformat(),
            },
        )
        self.write_yaml("resolved_config.yaml", config)
        self.write_text("command.txt", safe_text(command) + "\n")

    def _touch_required(self):
        for name in (
            "training.log",
            "metrics.jsonl",
            "metrics.csv",
            "gpu_metrics.csv",
            "completions.jsonl",
            "verification_results.jsonl",
        ):
            atomic_text(self.run_dir / name, "")

    def write_text(self, name, text):
        atomic_text(self.run_dir / name, safe_text(text))

    def write_json(self, name, payload):
        self.write_text(name, json.dumps(payload, indent=2, ensure_ascii=False) + "\n")

    def write_yaml(self, name, payload):
        self.write_text(name, yaml.safe_dump(payload, sort_keys=True))

    def append_jsonl(self, name, payload):
        path = self.run_dir / name
        existing = path.read_text() if path.exists() else ""
        self.write_text(name, existing + json.dumps(payload, ensure_ascii=False) + "\n")

    def write_csv(self, name, rows, fieldnames):
        temp = self.run_dir / f".{name}.csvtmp"
        temp.parent.mkdir(parents=True, exist_ok=True)
        with temp.open("w", newline="", encoding="utf-8") as h:
            writer = csv.DictWriter(h, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temp, self.run_dir / name)

    def environment(self, versions, gpu, cuda, git_commit):
        lines = (
            [f"Python: {sys.version.split()[0]}"]
            + [f"{k}: {v}" for k, v in versions.items()]
            + [f"CUDA: {cuda}", f"GPU: {gpu}", f"Git commit: {git_commit}"]
        )
        self.write_text("environment.txt", "\n".join(lines) + "\n")

    def pip_freeze(self):
        output = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True
        ).stdout
        self.write_text("pip_freeze.txt", output)

    def finalize(
        self,
        status,
        failed_stage=None,
        exception=None,
        stop_reason=None,
        counters=None,
        summary=None,
    ):
        payload = {
            "status": status,
            "failed_stage": failed_stage,
            "exception_type": type(exception).__name__ if exception else None,
            "stop_reason": stop_reason,
            "counters": counters or {"completions": 0, "generated_tokens": 0},
            **(summary or {}),
        }
        self.write_json("final_summary.json", payload)
        self.checksums()
        return payload

    def checksums(self):
        lines = []
        for path in sorted(self.run_dir.rglob("*")):
            if (
                path.is_file()
                and path.name != "checksums.sha256"
                and "model.safetensors" not in path.name
            ):
                lines.append(
                    f"{hashlib.sha256(path.read_bytes()).hexdigest()}  "
                    f"{path.relative_to(self.run_dir)}"
                )
        self.write_text("checksums.sha256", "\n".join(lines) + "\n")

    def publish_summary(self, report_md, summary, samples):
        self.report_dir.mkdir(parents=True, exist_ok=False)
        for name in ("resolved_config.yaml", "metrics.csv", "gpu_metrics.csv", "environment.txt"):
            shutil.copy2(self.run_dir / name, self.report_dir / name)
        shutil.copytree(self.run_dir / "figures", self.report_dir / "figures")
        atomic_text(self.report_dir / "report.md", report_md)
        atomic_text(self.report_dir / "summary.json", json.dumps(summary, indent=2) + "\n")
        atomic_bytes(
            self.report_dir / "samples.jsonl.gz",
            gzip.compress("".join(json.dumps(x) + "\n" for x in samples).encode()),
        )
        files = []
        for p in sorted(self.report_dir.rglob("*")):
            if p.is_file():
                files.append(
                    {
                        "path": str(p.relative_to(self.report_dir)),
                        "size": p.stat().st_size,
                        "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                    }
                )
        atomic_text(
            self.report_dir / "artifact_manifest.json",
            json.dumps({"run_id": self.run_id, "files": files}, indent=2) + "\n",
        )
