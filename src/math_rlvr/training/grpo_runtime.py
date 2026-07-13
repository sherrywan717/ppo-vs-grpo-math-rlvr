"""Delayed real GRPO smoke assembly; imported only after every CLI gate passes."""

from __future__ import annotations

import copy
import gc
import hashlib
import tarfile
from datetime import UTC, datetime
from pathlib import Path

from math_rlvr.artifacts.manager import ArtifactManager
from math_rlvr.artifacts.monitor import ResourceMonitor
from math_rlvr.dataset import MathProblem
from math_rlvr.prompt import format_problem
from math_rlvr.training.guarded_grpo import (
    REVISION,
    SNAPSHOT,
    checkpoint_inventory,
    require_clean_git,
    require_local_snapshot,
    run_guarded,
)
from math_rlvr.verifier import MathVerifier

BACKUP_ROOT = Path("/root/autodl-fs/math-rlvr-backups")


class RealLifecycle:
    def __init__(self, config):
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
        self.manager.write_json(name, payload)

    def finalize(self, summary):
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

    def backup_and_verify(self):
        BACKUP_ROOT.mkdir(parents=True, exist_ok=True)
        archive = BACKUP_ROOT / f"{self.manager.run_id}.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            tar.add(self.manager.run_dir, arcname=self.manager.run_id)
        with tarfile.open(archive, "r:gz") as tar:
            names = tar.getnames()
        forbidden = ("huggingface", "auth.json", "model.safetensors")
        if any(any(item in name.lower() for item in forbidden) for name in names):
            raise RuntimeError("backup inventory contains prohibited path")
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
    def __init__(self, config, lifecycle):
        self.config = copy.deepcopy(config)
        self.lifecycle = lifecycle

    def run(self, problems: list[MathProblem], guard, _unused_reward):
        import torch
        from datasets import Dataset

        from math_rlvr.rewards.result import DEFAULT_REWARD_POLICY
        from math_rlvr.training.builders import build_grpo_trainer, load_policy_and_tokenizer
        from math_rlvr.training.trl_compat import guarded_trainer_class, optimizer_guard_callback

        require_local_snapshot()
        self.config["model"]["name_or_path"] = str(SNAPSHOT)
        self.config["model"].pop("revision", None)
        model = tokenizer = trainer = None
        try:
            model, tokenizer = load_policy_and_tokenizer(self.config)
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
            rows = [{"prompt": format_problem(p), "problem_id": p.problem_id} for p in problems]
            dataset = Dataset.from_list(rows)
            verifier = MathVerifier()

            def reward_func(completions, problem_id, **kwargs):
                values = []
                for completion, pid in zip(completions, problem_id, strict=True):
                    text = completion if isinstance(completion, str) else completion[-1]["content"]
                    result = verifier(problem_map[pid], text)
                    scalar = DEFAULT_REWARD_POLICY.to_scalar(result)
                    guard.record_reward(result, scalar)
                    values.append(scalar)
                return values

            trainer = build_grpo_trainer(
                self.config,
                dataset,
                reward_func,
                self.lifecycle.manager.run_dir,
                model=model,
                tokenizer=tokenizer,
                trainer_factory=guarded_trainer_class(guard),
                cpu_only=False,
            )
            trainer.add_callback(optimizer_guard_callback(guard))
            output = trainer.train()
            checkpoint = self.lifecycle.manager.run_dir / "checkpoints" / "checkpoint-1"
            trainer.save_model(checkpoint)
            inventory = checkpoint_inventory(checkpoint)
            return {
                "checkpoint_dir": str(checkpoint),
                "metrics": dict(output.metrics),
                "checkpoint_inventory": inventory,
            }
        finally:
            del trainer, model, tokenizer
            gc.collect()
            if torch.cuda.is_initialized():
                torch.cuda.empty_cache()


def execute_real_smoke(config):
    """The only real GRPO entry; caller has already passed dual CLI authorization."""
    require_local_snapshot()
    lifecycle = RealLifecycle(config)
    return run_guarded(
        config, RealBackend(config, lifecycle), lambda _: None, lifecycle, RealMonitor(lifecycle)
    )
