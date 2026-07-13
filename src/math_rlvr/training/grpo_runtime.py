"""Delayed real GRPO smoke assembly; imported only after every CLI gate passes."""

from __future__ import annotations

import copy
import gc
import hashlib
import json
import shutil
import tarfile
from datetime import UTC, datetime
from pathlib import Path

from math_rlvr.artifacts.manager import ArtifactManager
from math_rlvr.artifacts.monitor import ResourceMonitor
from math_rlvr.dataset import MathProblem
from math_rlvr.prompt import format_training_problem
from math_rlvr.training.guarded_grpo import (
    REVISION,
    assert_json_safe,
    authoritative_checkpoint,
    require_clean_git,
    require_local_snapshot,
    run_guarded,
)
from math_rlvr.verifier import MathVerifier

BACKUP_ROOT = Path("/root/autodl-fs/math-rlvr-backups")


def validate_backup_inventory(names):
    forbidden_credentials = {
        "auth.json",
        "token.json",
        "tokens.json",
        "hf_token.txt",
        "auth_token.txt",
        "proxy.json",
        "proxy.txt",
    }
    for name in names:
        path = Path(name)
        basename = path.name.lower()
        parts = {part.lower() for part in path.parts}
        if (
            "huggingface" in parts
            or basename in forbidden_credentials
            or basename in {"model.safetensors", "pytorch_model.bin"}
        ):
            raise RuntimeError(f"backup inventory contains prohibited path: {name}")


class RealLifecycle:
    def __init__(self, config):
        self.config = config
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        self.manager = ArtifactManager(
            "single_update",
            "grpo",
            config["model"]["name_or_path"],
            config["experiment"]["seed"],
            "guarded dual-confirmation GRPO smoke",
            config,
            run_id=f"grpo_single_update_qwen25_05b_{stamp}",
        )
        self.manager.write_text("stdout.log", "")
        self.manager.write_text("stderr.log", "")
        self.manager.write_json("resolved_config.json", config)
        self.archive = None
        self._runtime_evidence = {}

    def start(self, config, problems):
        git = require_clean_git()
        self.manager.write_json("git.json", git)
        self.manager.write_json(
            "smoke_problems.json",
            [
                {
                    "problem_id": p.problem_id,
                    "source": p.source,
                    "split": p.split,
                    "prompt_hash": p.content_hash,
                }
                for p in problems
            ],
        )
        self.manager.write_json(
            "environment.json",
            {"offline": True, "revision": REVISION, "full_environment_dumped": False},
        )

    def persist(self, name, payload):
        assert_json_safe(payload)
        self.manager.write_json(name, payload)

    def persist_jsonl(self, name, rows):
        for row in rows:
            assert_json_safe(row)
        self.manager.write_text(
            name,
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        )

    def record_runtime(self, name, payload):
        assert_json_safe(payload)
        self._runtime_evidence[name] = payload
        self.manager.write_json(f"{name}.json", payload)

    def runtime_summary(self):
        return dict(self._runtime_evidence)

    def finalize(self, summary):
        manifest_path = self.manager.run_dir / "run_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest.update(
            {
                "status": summary["status"],
                "counters": summary["counters"],
                "completion_evidence_count": summary.get("completion_evidence_count", 0),
                "prompt_version": self.config["prompt_version"],
                "prompt_sha256": self.config["prompt_sha256"],
                "renderer_version": self.config["renderer_version"],
                "reward_policy_version": self.config["reward_policy_version"],
                "reward_component_weights": self.config["reward_component_weights"],
                "reward_policy_sha256": self.config["reward_policy_sha256"],
                "duplicate_checkpoint_count": summary.get("duplicate_checkpoint_count"),
            }
        )
        self.manager.write_json("run_manifest.json", manifest)
        self.manager.finalize(
            summary["status"],
            stop_reason=summary.get("reason"),
            counters=summary["counters"],
            summary=summary,
        )
        publish = summary["status"] == "failure" or summary.get("backed_up") is True
        if publish and not self.manager.report_dir.exists():
            if not (self.manager.run_dir / "environment.txt").exists():
                self.manager.write_text("environment.txt", "offline local-only guarded run\n")
            report = (
                "# GRPO single-update smoke\n\n"
                f"- Status: {summary['status']}\n"
                f"- Backed up: {summary.get('backed_up', False)}\n"
                "- Smoke diagnostic only; not an experiment result.\n"
            )
            self.manager.publish_summary(report, summary, [])
            self._publish_evidence_files()

    def _publish_evidence_files(self):
        safe_names = (
            "resolved_config.json",
            "run_manifest.json",
            "completions.jsonl",
            "trainer_metrics.json",
            "trainer_log_history.json",
            "pytorch_allocator.json",
            "checkpoint_inventory.json",
            "failure_report.json",
            "final_summary.json",
        )
        for name in safe_names:
            source = self.manager.run_dir / name
            if source.is_file():
                shutil.copy2(source, self.manager.report_dir / name)
        files = []
        for path in sorted(self.manager.report_dir.rglob("*")):
            if path.is_file() and path.name not in {
                "artifact_manifest.json",
                "checksums.sha256",
            }:
                files.append(
                    {
                        "path": str(path.relative_to(self.manager.report_dir)),
                        "size": path.stat().st_size,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    }
                )
        (self.manager.report_dir / "artifact_manifest.json").write_text(
            json.dumps({"run_id": self.manager.run_id, "files": files}, indent=2) + "\n"
        )
        (self.manager.report_dir / "checksums.sha256").write_text(
            "\n".join(f"{item['sha256']}  {item['path']}" for item in files) + "\n"
        )

    def backup_and_verify(self):
        BACKUP_ROOT.mkdir(parents=True, exist_ok=True)
        archive = BACKUP_ROOT / f"{self.manager.run_id}.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            tar.add(self.manager.run_dir, arcname=self.manager.run_id)
        with tarfile.open(archive, "r:gz") as tar:
            names = tar.getnames()
        validate_backup_inventory(names)
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        checksum = archive.with_suffix(archive.suffix + ".sha256")
        checksum.write_text(f"{digest}  {archive}\n")
        if hashlib.sha256(archive.read_bytes()).hexdigest() != digest:
            raise RuntimeError("backup checksum verification failed")
        self.archive = str(archive)


class RealMonitor:
    def __init__(self, lifecycle):
        self.monitor = ResourceMonitor(lifecycle.manager.run_dir / "resource_metrics.csv", 0.25)

    def start(self):
        self.monitor.start()

    def stop(self):
        self.monitor.stop()


class RealBackend:
    def __init__(self, config, lifecycle, model_source):
        self.config = copy.deepcopy(config)
        self.lifecycle = lifecycle
        self.model_source = model_source

    def run(self, problems: list[MathProblem], guard, _unused_reward):
        import torch
        from datasets import Dataset

        from math_rlvr.rewards.staged import reward_policy_from_config
        from math_rlvr.training.builders import build_grpo_trainer, load_policy_and_tokenizer
        from math_rlvr.training.resource_evidence import CudaAllocatorEvidence
        from math_rlvr.training.trl_compat import (
            CompletionEvidenceRecorder,
            extract_kl_metric,
            guarded_trainer_class,
            optimizer_guard_callback,
        )

        allocator = CudaAllocatorEvidence(torch.cuda)
        evidence = CompletionEvidenceRecorder(expected_completions=8)
        model = tokenizer = trainer = None
        run_result = {}
        try:
            allocator.start()
            model, tokenizer = load_policy_and_tokenizer(self.config, self.model_source)
            trainable = [name for name, p in model.named_parameters() if p.requires_grad]
            targets = tuple(self.config["lora"]["target_modules"])
            if not trainable or any(
                "lora_" not in name or not any(t in name for t in targets) for name in trainable
            ):
                raise RuntimeError("unexpected trainable parameter outside frozen LoRA")
            tokenizer.padding_side = "left"
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token
            problem_map = {p.problem_id: p for p in problems}
            rows = [
                {
                    "prompt": format_training_problem(p, self.config),
                    "problem_id": p.problem_id,
                    "prompt_hash": p.content_hash,
                }
                for p in problems
            ]
            dataset = Dataset.from_list(rows)
            verifier = MathVerifier()
            policy = reward_policy_from_config(self.config)

            def reward_func(completions, problem_id, **kwargs):
                values = []
                for completion, pid in zip(completions, problem_id, strict=True):
                    text = completion if isinstance(completion, str) else completion[-1]["content"]
                    problem = problem_map[pid]

                    def bound_verifier(candidate, problem=problem):
                        return verifier(problem, candidate)

                    evaluation = policy.evaluate(text, bound_verifier)
                    result = evaluation.canonical_result
                    scalar = evaluation.scalar_reward
                    reward_evidence = evaluation.to_dict()
                    evidence.record_reward(pid, text, result, scalar, reward_evidence)
                    guard.record_reward(result, scalar, reward_evidence)
                    values.append(scalar)
                return values

            trainer = build_grpo_trainer(
                self.config,
                dataset,
                reward_func,
                self.lifecycle.manager.run_dir,
                model=model,
                tokenizer=tokenizer,
                trainer_factory=guarded_trainer_class(guard, evidence),
                cpu_only=False,
                model_source=self.model_source,
            )
            trainer.add_callback(optimizer_guard_callback(guard))
            output = trainer.train()
            global_step = int(trainer.state.global_step)
            checkpoint = authoritative_checkpoint(self.lifecycle.manager.run_dir, global_step)
            log_history = [dict(row) for row in trainer.state.log_history]
            kl = extract_kl_metric(log_history, float(trainer.args.beta))
            metrics = {
                "trainer_output": dict(output.metrics),
                "kl": kl,
            }
            run_result.update(
                {
                    "run_dir": str(self.lifecycle.manager.run_dir),
                    "checkpoint_dir": str(checkpoint),
                    "metrics": metrics,
                    "trainer_log_history": log_history,
                    "completions": evidence.records(),
                }
            )
            return run_result
        finally:
            del trainer, model, tokenizer
            gc.collect()
            if torch.cuda.is_initialized():
                torch.cuda.empty_cache()
            allocator_payload = allocator.finalize()
            self.lifecycle.record_runtime("pytorch_allocator", allocator_payload)
            run_result["pytorch_allocator"] = allocator_payload


def execute_real_smoke(config):
    """The only real GRPO entry; caller has already passed dual CLI authorization."""
    model_source = require_local_snapshot()
    lifecycle = RealLifecycle(config)
    return run_guarded(
        config,
        RealBackend(config, lifecycle, model_source),
        lambda _: None,
        lifecycle,
        RealMonitor(lifecycle),
    )
