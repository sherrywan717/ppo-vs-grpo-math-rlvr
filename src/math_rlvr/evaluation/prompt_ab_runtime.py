"""Delayed real backend for the guarded generation-only diagnostic.

Dry-run never imports this module. It contains no Trainer, PEFT, optimizer,
backward, adapter, or checkpoint path.
"""

from __future__ import annotations

import csv
import gc
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import tarfile
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

from math_rlvr.artifacts.manager import SECRET_PATTERN
from math_rlvr.artifacts.monitor import ResourceMonitor
from math_rlvr.evaluation.prompt_ab import (
    GeneratedSequence,
    run_diagnostic,
    split_completion_ids,
)
from math_rlvr.evaluation.prompt_ab_evidence import validate_cross_file_consistency
from math_rlvr.prompt import render_prompt_version
from math_rlvr.training.model_source import ValidatedModelSource
from math_rlvr.training.resource_evidence import CudaAllocatorEvidence

RUN_ROOT = Path("/root/autodl-tmp/runs/math_rlvr")
REPORT_ROOT = Path("reports/runs")
BACKUP_ROOT = Path("/root/autodl-fs/math-rlvr-backups")


class RealGenerationBackend:
    backward_count = 0
    optimizer_steps = 0
    training_steps = 0
    checkpoint_writes = 0
    model_writes = 0

    def __init__(self, source, config):
        self.source = source
        self.config = config
        self.model = None
        self.tokenizer = None
        self.torch = None
        self.device = None
        self._peak_vram_gib = 0.0
        self.rng_records = []
        self.allocator = None
        self.eval_called = False
        self.inference_mode_used = False
        self.parameters_frozen = False

    def prepare(self):
        import random

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
            raise RuntimeError("CUDA device 0 unavailable")
        self.torch = torch
        self.device = torch.device("cuda:0")
        self.allocator = CudaAllocatorEvidence(torch.cuda, device=0)
        self.allocator.start()
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.source.snapshot_path), local_files_only=True
        )
        self.tokenizer.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(
            str(self.source.snapshot_path),
            local_files_only=True,
            dtype=torch.bfloat16,
            device_map={"": 0},
        )
        self.model.eval()
        self.eval_called = not self.model.training
        self.model.requires_grad_(False)
        self.parameters_frozen = not any(p.requires_grad for p in self.model.parameters())
        if self.model.training or any(
            parameter.requires_grad for parameter in self.model.parameters()
        ):
            raise RuntimeError("base model must be eval-only with gradients disabled")
        random.seed(self.config["experiment"]["seed"])
        torch.manual_seed(self.config["experiment"]["seed"])
        torch.cuda.manual_seed_all(self.config["experiment"]["seed"])

    def render(self, problem, prompt_version):
        text = render_prompt_version(self.tokenizer, problem, prompt_version)
        return text, hashlib.sha256(text.encode()).hexdigest()

    def generate(self, prompt, *, seed, sampling, max_new_tokens):
        import random

        torch = self.torch
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        self.rng_records.append(
            {"python_seed": seed, "torch_cpu_seed": seed, "torch_cuda_seed": seed}
        )
        encoded = self.tokenizer(
            [prompt], return_tensors="pt", padding=True, add_special_tokens=False
        )
        encoded = {name: tensor.to(self.device) for name, tensor in encoded.items()}
        kwargs = {key: value for key, value in sampling.items() if value is not None}
        with torch.inference_mode():
            self.inference_mode_used = not torch.is_grad_enabled()
            sequences = self.model.generate(**encoded, max_new_tokens=max_new_tokens, **kwargs)
        if sequences.requires_grad:
            raise RuntimeError("generated tensor unexpectedly requires gradients")
        split = split_completion_ids(
            sequences[0].detach().cpu().tolist(),
            padded_input_width=encoded["input_ids"].shape[1],
            input_attention_mask=encoded["attention_mask"][0].detach().cpu().tolist(),
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        return GeneratedSequence(
            input_token_count=split.input_token_count,
            completion_ids=split.completion_ids,
            decoded_text=self.tokenizer.decode(split.completion_ids, skip_special_tokens=True),
            eos_reached=split.eos_reached,
        )

    def peak_vram_gib(self):
        output = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.splitlines()
        for line in output:
            fields = [field.strip() for field in line.split(",")]
            if len(fields) == 2 and fields[0] == str(os.getpid()):
                self._peak_vram_gib = max(self._peak_vram_gib, float(fields[1]) / 1024)
        return self._peak_vram_gib

    def close(self):
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        gc.collect()
        if self.torch is None or not self.torch.cuda.is_initialized():
            return {
                "available": False,
                "lifecycle": ["not_started"],
                "unavailable_reason": "CUDA allocator was not initialized",
            }
        index = self.allocator.device_index
        self.torch.cuda.synchronize(index)
        self.torch.cuda.empty_cache()
        self.torch.cuda.synchronize(index)
        evidence = self.allocator.finalize()
        current_allocated = evidence["memory_allocated"]["bytes"]
        current_reserved = evidence["memory_reserved"]["bytes"]
        evidence["lifecycle"] = ["not_started", "active", "finalized"]
        evidence["worker_cleanup"] = {
            "gc_collect_called": True,
            "empty_cache_called": True,
            "synchronized": True,
            "current_allocated_bytes": current_allocated,
            "current_reserved_bytes": current_reserved,
        }
        if current_allocated != 0 or current_reserved != 0:
            raise RuntimeError("worker CUDA allocator memory was not fully released")
        return evidence


class PromptABArtifacts:
    backed_up = False

    def __init__(self, config, git_info):
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        self.run_id = f"prompt_ab_qwen25_05b_{stamp}"
        self.run_dir = RUN_ROOT / self.run_id
        self.report_dir = REPORT_ROOT / self.run_id
        self.config = config
        self.git_info = git_info
        self.monitor = ResourceMonitor(self.run_dir / "resource_metrics.csv", interval=0.25)
        self.summary = {}

    @staticmethod
    def _json(path, payload):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
        json.loads(path.read_text())

    def start(self, config, problems, seed_map):
        self.run_dir.mkdir(parents=True, exist_ok=False)
        (self.run_dir / "figures").mkdir()
        for name in ("stdout.log", "stderr.log", "completions.jsonl"):
            (self.run_dir / name).write_text("")
        self._json(self.run_dir / "resolved_config.json", config)
        self._json(
            self.run_dir / "run_manifest.json",
            {
                "run_id": self.run_id,
                "status": "running",
                "diagnostic_only": True,
                "training": False,
                "git": self.git_info,
                "problem_ids": [problem.problem_id for problem in problems],
            },
        )
        self._json(
            self.run_dir / "prompt_hashes.json",
            [
                {"problem_id": problem.problem_id, "prompt_hash": problem.content_hash}
                for problem in problems
            ],
        )
        self._json(self.run_dir / "seed_map.json", seed_map)
        self._json(
            self.run_dir / "environment.json",
            {
                "python": sys.version.split()[0],
                "torch": version("torch"),
                "transformers": version("transformers"),
                "offline_required": True,
            },
        )
        self.monitor.start()

    def persist_jsonl(self, name, rows):
        (self.run_dir / name).write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
        )

    def persist_csv(self, name, rows):
        if not rows:
            raise RuntimeError("cannot persist empty CSV evidence")
        with (self.run_dir / name).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def persist(self, name, payload):
        self._json(self.run_dir / name, payload)
        if name == "summary.json":
            self.summary = payload

    def _write_metrics_csv(self):
        metrics = self.summary.get("condition_metrics", {})
        if not metrics:
            return
        fields = [
            "condition",
            "completions",
            "complete_envelope_rate",
            "format_accuracy",
            "truncation_rate",
            "reward_mean",
            "reward_variance",
            "nonzero_advantage_potential_groups",
            "zero_advantage_group_count",
            "nonzero_variance_group_count",
            "completion_token_mean",
        ]
        with (self.run_dir / "per_condition_metrics.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for condition, values in metrics.items():
                writer.writerow(
                    {key: condition if key == "condition" else values[key] for key in fields}
                )

    def _save_plot(self, plt, fig, filename):
        fig.suptitle("generation-only prompt diagnostic; no training")
        fig.tight_layout()
        fig.savefig(self.run_dir / "figures" / f"{filename}.png", dpi=160)
        plt.close(fig)

    def _plots(self):
        import matplotlib.pyplot as plt

        metrics = self.summary.get("condition_metrics", {})
        if not metrics:
            return
        labels = list(metrics)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(labels, [metrics[name]["complete_envelope_rate"] for name in labels])
        ax.set_ylabel("complete-envelope rate")
        self._save_plot(plt, fig, "complete_envelope")

        tag_keys = (
            "reasoning_open_rate",
            "reasoning_close_rate",
            "answer_open_rate",
            "answer_close_rate",
        )
        fig, ax = plt.subplots(figsize=(7, 4))
        width = 0.18
        for index, key in enumerate(tag_keys):
            ax.bar(
                [value + index * width for value in range(len(labels))],
                [metrics[name][key] for name in labels],
                width=width,
                label=key,
            )
        ax.set_xticks([value + 1.5 * width for value in range(len(labels))], labels)
        ax.legend(fontsize=7)
        self._save_plot(plt, fig, "tag_compliance")

        statuses = sorted(
            {status for values in metrics.values() for status in values["reward_status_counts"]}
        )
        fig, ax = plt.subplots(figsize=(7, 4))
        for index, status in enumerate(statuses):
            ax.bar(
                [value + index * 0.25 for value in range(len(labels))],
                [metrics[name]["reward_status_counts"].get(status, 0) for name in labels],
                width=0.25,
                label=status,
            )
        ax.legend(fontsize=7)
        self._save_plot(plt, fig, "reward_status")

        fig, ax = plt.subplots(figsize=(7, 4))
        problem_ids = sorted(
            {
                problem_id
                for values in metrics.values()
                for problem_id in values["group_reward_variance"]
            }
        )
        for index, problem_id in enumerate(problem_ids):
            ax.bar(
                [value + index * 0.3 for value in range(len(labels))],
                [metrics[name]["group_reward_variance"][problem_id] for name in labels],
                width=0.3,
                label=problem_id,
            )
        ax.legend(fontsize=7)
        self._save_plot(plt, fig, "group_reward_variance")

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(labels, [metrics[name]["completion_token_mean"] for name in labels])
        ax.set_ylabel("mean completion tokens")
        trunc = ax.twinx()
        trunc.plot(labels, [metrics[name]["truncation_rate"] for name in labels], marker="o")
        trunc.set_ylabel("truncation rate")
        self._save_plot(plt, fig, "completion_length_truncation")

        pairs_path = self.run_dir / "paired_comparison.json"
        if pairs_path.is_file():
            pairs = json.loads(pairs_path.read_text(encoding="utf-8"))
            transitions = {}
            for pair in pairs:
                key = pair["status_transition"]
                transitions[key] = transitions.get(key, 0) + 1
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(list(transitions), list(transitions.values()))
            ax.tick_params(axis="x", rotation=20)
            self._save_plot(plt, fig, "paired_status_transition")

        fig, memory_ax = plt.subplots(figsize=(7, 4))
        elapsed = [row["elapsed_seconds"] for row in self.monitor.rows]
        memory = [row["gpu_memory_used_mb"] or 0 for row in self.monitor.rows]
        utilization = [row["gpu_utilization_pct"] or 0 for row in self.monitor.rows]
        memory_ax.plot(elapsed, memory, label="nvidia-smi memory MiB")
        util_ax = memory_ax.twinx()
        util_ax.plot(elapsed, utilization, color="tab:orange", label="utilization %")
        memory_ax.set_xlabel("seconds")
        memory_ax.set_ylabel("MiB")
        util_ax.set_ylabel("%")
        self._save_plot(plt, fig, "gpu_memory_utilization")

        budget = self.summary["budget"]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(
            ("completion plan", "completion actual", "token plan", "token actual"),
            (
                budget["limits"]["total_completions"],
                budget["total_completions"],
                budget["limits"]["total_generated_tokens"],
                budget["total_generated_tokens"],
            ),
        )
        ax.tick_params(axis="x", rotation=15)
        self._save_plot(plt, fig, "planned_vs_actual_budget")

    def refresh_checksums(self):
        lines = []
        for path in sorted(self.run_dir.rglob("*")):
            if path.is_file() and path.name != "checksums.sha256":
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                lines.append(f"{digest}  {path.relative_to(self.run_dir)}")
        (self.run_dir / "checksums.sha256").write_text("\n".join(lines) + "\n")

    def finalize(self, summary):
        self.monitor.stop()
        self.summary = summary
        self._write_metrics_csv()
        self._plots()
        self._json(
            self.run_dir / "resource_cost.json",
            self.monitor.summary(price=self.config["budget"]["gpu_hour_price_cny"]),
        )
        (self.run_dir / "report.md").write_text(
            "# Generation-only prompt diagnostic\n\n"
            f"- Status: {summary['status']}\n"
            "- generation-only prompt diagnostic; no training\n"
        )
        self._json(
            self.run_dir / "run_manifest.json",
            {
                "run_id": self.run_id,
                "status": summary["status"],
                "training": False,
                "completion_count": summary.get("completion_count", 0),
                "total_generated_tokens": summary.get("budget", {}).get(
                    "total_generated_tokens", 0
                ),
                "safety_counters": summary.get("safety_counters", {}),
            },
        )
        lines = []
        for path in sorted(self.run_dir.rglob("*")):
            if path.is_file() and path.name != "checksums.sha256":
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                lines.append(f"{digest}  {path.relative_to(self.run_dir)}")
        (self.run_dir / "checksums.sha256").write_text("\n".join(lines) + "\n")

    def backup_and_verify(self, *, failure=False):
        if list(self.run_dir.rglob("*.safetensors")) or list(self.run_dir.rglob("checkpoint-*")):
            raise RuntimeError("model/checkpoint artifact forbidden")
        for path in self.run_dir.rglob("*"):
            if not path.is_file():
                continue
            if path.stat().st_size > 10 * 1024 * 1024:
                raise RuntimeError(f"unexpected large artifact: {path.name}")
            if path.suffix in {".json", ".jsonl", ".csv", ".md", ".log", ".txt", ".yaml"}:
                text = path.read_text(encoding="utf-8", errors="replace")
                if SECRET_PATTERN.search(text):
                    raise RuntimeError(f"secret-like artifact rejected: {path.name}")
        BACKUP_ROOT.mkdir(parents=True, exist_ok=True)
        suffix = "-failure" if failure else ""
        archive = BACKUP_ROOT / f"{self.run_id}{suffix}.tar.gz"
        with tarfile.open(archive, "w:gz") as handle:
            handle.add(self.run_dir, arcname=self.run_id)
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        sidecar = Path(f"{archive}.sha256")
        sidecar.write_text(f"{digest}  {archive}\n")
        with tarfile.open(archive, "r:gz") as handle:
            names = handle.getnames()
        if not names or any(
            "checkpoint" in name.lower() or "cache" in name.lower() for name in names
        ):
            raise RuntimeError("backup content validation failed")
        if hashlib.sha256(archive.read_bytes()).hexdigest() != digest:
            raise RuntimeError("backup checksum verification failed")
        self._json(
            self.run_dir / "backup_manifest.json",
            {
                "archive": str(archive),
                "sha256": digest,
                "verified": True,
                "failure_archive": failure,
                "member_count": len(names),
            },
        )
        self.backed_up = True

    def publish_git_safe(self):
        self.report_dir.mkdir(parents=True, exist_ok=False)
        for path in self.run_dir.iterdir():
            if path.is_file() and path.suffix not in {".bin", ".safetensors"}:
                shutil.copy2(path, self.report_dir / path.name)
        shutil.copytree(self.run_dir / "figures", self.report_dir / "figures")


def execute_real_diagnostic(*, config, source, git_info):
    if not isinstance(source, ValidatedModelSource):
        raise RuntimeError("real diagnostic requires ValidatedModelSource")

    def timeout_handler(_signum, _frame):
        raise TimeoutError("generation diagnostic exceeded hard wall-time limit")

    previous = signal.signal(signal.SIGALRM, timeout_handler)
    signal.setitimer(signal.ITIMER_REAL, config["budget"]["max_wall_time_seconds"])
    try:
        return run_diagnostic(
            config,
            RealGenerationBackend(source, config),
            PromptABArtifacts(config, git_info),
        )
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def execute_worker_diagnostic(*, config, source, git_info):
    """Execute once in the spawned CUDA worker and defer backup to its parent."""
    if not isinstance(source, ValidatedModelSource):
        raise RuntimeError("real diagnostic requires ValidatedModelSource")
    artifacts = PromptABArtifacts(config, git_info)

    def timeout_handler(_signum, _frame):
        raise TimeoutError("generation diagnostic exceeded hard wall-time limit")

    previous = signal.signal(signal.SIGALRM, timeout_handler)
    signal.setitimer(signal.ITIMER_REAL, config["budget"]["max_wall_time_seconds"])
    try:
        result = run_diagnostic(
            config,
            RealGenerationBackend(source, config),
            artifacts,
            defer_parent_finalization=True,
        )
        return {"run_id": artifacts.run_id, "result": result}
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def finalize_parent_diagnostic(*, payload, post_worker, config, git_info):
    """Finalize, verify, back up and publish only after the worker process exits."""
    result = payload.get("result", {})
    run_id = payload.get("run_id")
    if not run_id:
        return result
    artifacts = PromptABArtifacts(config, git_info)
    artifacts.run_id = run_id
    artifacts.run_dir = RUN_ROOT / run_id
    artifacts.report_dir = REPORT_ROOT / run_id
    artifacts.summary = result
    artifacts._json(artifacts.run_dir / "post_worker_gpu_verification.json", post_worker)
    failure = result.get("status") != "worker_complete"
    try:
        result["status"] = "failure" if failure else "pending_backup"
        result["post_worker_gpu_verification"] = post_worker
        artifacts._json(artifacts.run_dir / "summary.json", result)
        artifacts._json(
            artifacts.run_dir / "run_manifest.json",
            {
                "run_id": run_id,
                "status": result["status"],
                "training": False,
                "completion_count": result.get("completion_count", 0),
                "total_generated_tokens": result.get("budget", {}).get("total_generated_tokens", 0),
                "safety_counters": result.get("safety_counters", {}),
            },
        )
        artifacts.refresh_checksums()
        artifacts.backup_and_verify(failure=failure)
        artifacts.refresh_checksums()
        if failure:
            result["backed_up"] = artifacts.backed_up
            artifacts._json(artifacts.run_dir / "summary.json", result)
            artifacts.refresh_checksums()
            artifacts.publish_git_safe()
            return result
        consistency = validate_cross_file_consistency(artifacts.run_dir, require_backup=True)
        result.update(
            {
                "status": "success",
                "backed_up": True,
                "cross_file_consistency": consistency,
            }
        )
        artifacts._json(artifacts.run_dir / "summary.json", result)
        artifacts._json(
            artifacts.run_dir / "run_manifest.json",
            {
                "run_id": run_id,
                "status": "success",
                "training": False,
                "completion_count": 16,
                "total_generated_tokens": result["budget"]["total_generated_tokens"],
                "safety_counters": result["safety_counters"],
            },
        )
        artifacts.refresh_checksums()
        artifacts.publish_git_safe()
        return result
    except Exception as exc:
        result.update(
            {
                "status": "failure",
                "reason": f"{type(exc).__name__}: {exc}",
                "backed_up": artifacts.backed_up,
            }
        )
        artifacts._json(artifacts.run_dir / "failure_report.json", result)
        artifacts._json(artifacts.run_dir / "summary.json", result)
        try:
            artifacts.refresh_checksums()
        except Exception:
            pass
        try:
            artifacts.backup_and_verify(failure=True)
        except Exception as backup_exc:
            result["backup_error"] = f"{type(backup_exc).__name__}: {backup_exc}"
            try:
                artifacts._json(artifacts.run_dir / "backup_failure.json",
                    {"status": "failure", "backup_verified": False,
                     "error": result["backup_error"]})
            except Exception:
                pass
        return result
