"""Delayed real PPO smoke assembly; imported only after every CLI gate passes."""

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
from math_rlvr.prompt import render_training_prompt
from math_rlvr.training.execution_contract import expected_run_contract_for_config
from math_rlvr.training.grpo_runtime import validate_backup_inventory
from math_rlvr.training.guarded_grpo import (
    assert_json_safe,
    require_clean_git,
    require_local_snapshot,
)
from math_rlvr.training.guarded_ppo import (
    PPO_SMOKE_CONFIG,
    ppo_checkpoint_inventory,
    ppo_execution_problems_and_episodes,
    run_guarded_ppo,
)

BACKUP_ROOT = Path("/root/autodl-fs/math-rlvr-backups")


class RealPPOLifecycle:
    def __init__(self, config):
        self.config = config
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        is_pilot = config.get("pilot", {}).get("family") == "matched_0p5b_v1"
        self.manager = ArtifactManager(
            "pilot_0p5b" if is_pilot else "single_update",
            "ppo",
            config["model"]["name_or_path"],
            config["experiment"]["seed"],
            (
                "guarded dual-confirmation matched PPO pilot"
                if is_pilot
                else "guarded dual-confirmation PPO smoke"
            ),
            config,
            run_id=(
                f"ppo_matched_0p5b_seed{config['experiment']['seed']}_{stamp}"
                if is_pilot
                else f"ppo_single_update_qwen25_05b_{stamp}"
            ),
        )
        self.manager.write_text("stdout.log", "")
        self.manager.write_text("stderr.log", "")
        self.manager.write_json("resolved_config.json", config)
        self.archive = None

    def start(self, config, problems):
        self.manager.write_json("git.json", require_clean_git())
        self.manager.write_json(
            "smoke_problems.json",
            [
                {
                    "problem_id": problem.problem_id,
                    "source": problem.source,
                    "split": problem.split,
                    "prompt_hash": problem.content_hash,
                }
                for problem in problems
            ],
        )
        self.manager.write_json(
            "environment.json",
            {
                "offline": True,
                "revision": config["model"]["revision"],
                "full_environment_dumped": False,
            },
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

    def finalize(self, summary):
        manifest_path = self.manager.run_dir / "run_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest.update(
            {
                "status": summary["status"],
                "counters": summary["counters"],
                "model": self.config["model"]["name_or_path"],
                "revision": self.config["model"].get("revision"),
                "algorithm": self.config["experiment"]["algorithm"],
                "seed": self.config["experiment"]["seed"],
                "completion_evidence_count": summary.get("completion_evidence_count", 0),
                "expected_run_contract": summary.get("expected_run_contract"),
                "comparison_pair_keys": summary.get("expected_run_contract", {}).get("pair_keys"),
                "resolved_ppo_contract": summary.get("resolved_ppo_contract"),
                "model_roles": summary.get("model_roles"),
                "prompt_version": self.config["prompt_version"],
                "prompt_sha256": self.config["prompt_sha256"],
                "renderer_version": self.config["renderer_version"],
                "reward_policy_version": self.config["reward_policy_version"],
                "reward_component_weights": self.config["reward_component_weights"],
                "reward_policy_sha256": self.config["reward_policy_sha256"],
                "parser_contract": self.config.get("parser_contract"),
                "verifier_contract": self.config.get("verifier_contract"),
                "resolved_config_path": self.config.get("resolved_config_path"),
                "resolved_config_sha256": self.config.get("resolved_config_sha256"),
                "pilot_manifest_sha256": self.config.get("data", {}).get("pilot_manifest_sha256"),
                "report_disclaimer": self.config.get("reporting", {}).get(
                    "disclaimer", "Smoke diagnostic only; not an experiment result."
                ),
            }
        )
        self.manager.write_json("run_manifest.json", manifest)
        self.manager.finalize(
            summary["status"],
            stop_reason=summary.get("reason"),
            counters=summary["counters"],
            summary=summary,
        )
        publish = summary.get("backed_up") is True
        if publish and not self.manager.report_dir.exists():
            if not (self.manager.run_dir / "environment.txt").exists():
                self.manager.write_text("environment.txt", "offline local-only guarded PPO run\n")
            disclaimer = self.config.get("reporting", {}).get(
                "disclaimer", "Smoke diagnostic only; not an experiment result."
            )
            title = (
                "PPO matched pilot run"
                if self.config.get("pilot", {}).get("family") == "matched_0p5b_v1"
                else "PPO single-update smoke"
            )
            report = (
                f"# {title}\n\n"
                f"- Status: {summary['status']}\n"
                f"- Backed up: {summary.get('backed_up', False)}\n"
                f"- {disclaimer}\n"
            )
            self.manager.publish_summary(report, summary, [])
            self._publish_evidence_files()

    def _publish_evidence_files(self):
        safe_names = (
            "resolved_config.json",
            "run_manifest.json",
            "expected_run_contract.json",
            "prompt_scope_preflight.json",
            "ppo_episode_order.json",
            "ppo_loader_contract.json",
            "completions.jsonl",
            "trainer_metrics.json",
            "trainer_log_history.json",
            "model_roles.json",
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

    def backup_and_verify(self, failure=False):
        BACKUP_ROOT.mkdir(parents=True, exist_ok=True)
        suffix = ".failure" if failure else ""
        archive = BACKUP_ROOT / f"{self.manager.run_id}{suffix}.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            tar.add(self.manager.run_dir, arcname=self.manager.run_id)
        with tarfile.open(archive, "r:gz") as tar:
            names = tar.getnames()
        validate_backup_inventory(names)
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        checksum = archive.with_suffix(archive.suffix + ".sha256")
        checksum.write_text(f"{digest}  {archive}\n")
        if hashlib.sha256(archive.read_bytes()).hexdigest() != digest:
            raise RuntimeError("PPO backup checksum verification failed")
        self.archive = str(archive)


class RealPPOMonitor:
    def __init__(self, lifecycle):
        self.lifecycle = lifecycle
        self.config = lifecycle.config
        self.monitor = ResourceMonitor(lifecycle.manager.run_dir / "resource_metrics.csv", 0.25)

    def start(self):
        self.monitor.start()

    def stop(self):
        self.monitor.stop()
        self.lifecycle.manager.write_csv(
            "gpu_metrics.csv", self.monitor.rows, ResourceMonitor.FIELDS
        )
        self.lifecycle.persist_jsonl("resource_metrics.jsonl", self.monitor.rows)
        summary = self.monitor.summary(self.config["budget"]["gpu_hour_price_cny"])
        self.lifecycle.persist("resource_summary.json", summary)
        from math_rlvr.artifacts.plotting import generate

        made, unavailable = generate(
            self.lifecycle.manager.run_dir,
            self.lifecycle.manager.run_id,
            "ppo",
            self.config["experiment"]["seed"],
            self.config.get("reporting", {}).get(
                "disclaimer", "Smoke test — not a benchmark result"
            ),
        )
        self.lifecycle.persist(
            "plot_inventory.json", {"generated": made, "unavailable": unavailable}
        )
        limits = self.config["budget"]
        if (
            summary["gpu_hours"] > limits["max_gpu_hours"]
            or summary["estimated_cost_cny"] > limits["max_estimated_cost_cny"]
            or (
                summary["peak_vram_mb"] is not None
                and summary["peak_vram_mb"] > limits["max_vram_gib"] * 1024
            )
        ):
            raise RuntimeError(f"PPO resource budget exceeded: {summary}")


def _cpu_tensors(state):
    return {name: tensor.detach().cpu().contiguous() for name, tensor in state.items()}


def write_authoritative_ppo_checkpoint(
    root: Path, policy, value_model, trainer_state, model_roles
) -> Path:
    """Write only role-separated LoRA/head state; never serialize a base model."""
    from peft import get_peft_model_state_dict
    from safetensors.torch import save_file

    if root.exists() or root.name != "checkpoint-1":
        raise RuntimeError("PPO checkpoint-1 must be a new path")
    policy_state = _cpu_tensors(get_peft_model_state_dict(policy))
    value_state = _cpu_tensors(get_peft_model_state_dict(value_model))
    policy_adapter = {name: tensor for name, tensor in policy_state.items() if "lora_" in name}
    value_adapter = {name: tensor for name, tensor in value_state.items() if "lora_" in name}
    value_head = {name: tensor for name, tensor in value_state.items() if "score" in name}
    if (
        not policy_adapter
        or not value_adapter
        or not value_head
        or set(policy_state) != set(policy_adapter)
        or set(value_state) != set(value_adapter) | set(value_head)
    ):
        raise RuntimeError("PPO PEFT checkpoint role partition failed")

    policy_dir = root / "policy_adapter"
    value_dir = root / "value_adapter"
    head_dir = root / "value_head"
    policy_dir.mkdir(parents=True)
    value_dir.mkdir()
    head_dir.mkdir()
    policy.peft_config["default"].save_pretrained(policy_dir)
    value_model.peft_config["default"].save_pretrained(value_dir)
    save_file(policy_adapter, str(policy_dir / "adapter_model.safetensors"))
    save_file(value_adapter, str(value_dir / "adapter_model.safetensors"))
    save_file(value_head, str(head_dir / "value_head.safetensors"))
    (head_dir / "config.json").write_text(
        json.dumps({"architecture": "scalar_score_head", "num_labels": 1}, indent=2) + "\n"
    )
    (root / "trainer_state.json").write_text(
        json.dumps(trainer_state, indent=2, sort_keys=True) + "\n"
    )
    (root / "resume_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "checkpoint": "checkpoint-1",
                "base_weights_included": False,
                "policy_base": "fixed local snapshot from resolved config",
                "value_base": "same fixed local snapshot from resolved config",
                "roles": {
                    "policy_adapter": "policy LoRA only",
                    "value_adapter": "value LoRA only",
                    "value_head": "trainable scalar score head only",
                },
                "optimizer_state_included": False,
                "model_roles": model_roles,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    ppo_checkpoint_inventory(root)
    return root


def build_ppo_runtime_dataset_rows(
    config,
    tokenizer,
    problems,
    episode_records,
    scope,
):
    """Actual delayed PPO row builder, parameterized by validated scope evidence."""
    from math_rlvr.training.execution_contract import validated_scope_from_config
    from math_rlvr.training.pilot import rendered_prompt_payload_sha256

    validated = validated_scope_from_config(config, "ppo")
    if scope != validated:
        raise ValueError("PPO dataset-builder scope differs from validated config scope")
    prompt_lookup = {}
    dataset_rows = []
    for problem, episode in zip(problems, episode_records, strict=True):
        rendered = render_training_prompt(tokenizer, problem, config, scope=scope.scope)
        rendered_hash = rendered_prompt_payload_sha256(problem, config["prompt_version"])
        if rendered_hash != episode["rendered_prompt_hash"]:
            raise ValueError("PPO delayed renderer prompt hash drift")
        prompt_ids = tokenizer(rendered, add_special_tokens=False, truncation=False)["input_ids"]
        if len(prompt_ids) > config["generation"]["max_prompt_length"]:
            raise RuntimeError("PPO fixed prompt exceeds max_prompt_length")
        key = tuple(int(value) for value in prompt_ids)
        existing = prompt_lookup.get(key)
        if existing is not None and existing["problem_id"] != problem.problem_id:
            raise RuntimeError("distinct PPO problems have colliding tokenized prompts")
        prompt_lookup.setdefault(
            key,
            {
                "problem_id": problem.problem_id,
                "prompt_hash": episode["rendered_prompt_hash"],
                "problem": problem,
            },
        )
        dataset_rows.append({"input_ids": prompt_ids, **episode})
    return prompt_lookup, dataset_rows


class RealPPOBackend:
    def __init__(self, config, lifecycle, model_source, prompt_preflight):
        self.config = copy.deepcopy(config)
        self.lifecycle = lifecycle
        self.model_source = model_source
        self.prompt_preflight = copy.deepcopy(prompt_preflight)

    def run(self, problems: list[MathProblem], guard):
        import torch
        from datasets import Dataset

        from math_rlvr.rewards.adapters import PPOVerifierRewardModel
        from math_rlvr.rewards.staged import reward_policy_from_config
        from math_rlvr.training.builders import (
            audit_ppo_parameter_roles,
            build_ppo_trainer,
            load_policy_and_tokenizer,
            load_value_model,
        )
        from math_rlvr.training.resource_evidence import CudaAllocatorEvidence
        from math_rlvr.training.runtime_prompt_scope import validate_runtime_prompt_preflight
        from math_rlvr.training.trl_compat import (
            PPOCompletionEvidenceRecorder,
            extract_ppo_metrics,
            ppo_guarded_trainer_class,
        )
        from math_rlvr.verifier import MathVerifier

        contract = expected_run_contract_for_config(self.config, "ppo")
        scope = validate_runtime_prompt_preflight(
            self.config, "ppo", self.prompt_preflight
        )
        expected_problems, episode_records = ppo_execution_problems_and_episodes(
            self.config, contract
        )
        self.lifecycle.persist("prompt_scope_preflight.json", self.prompt_preflight)
        allocator = CudaAllocatorEvidence(torch.cuda)
        policy = tokenizer = value_model = reward_model = trainer = None
        run_result = {}
        try:
            allocator.start()
            policy, tokenizer = load_policy_and_tokenizer(self.config, self.model_source)
            value_model = load_value_model(self.config, self.model_source)
            if [problem.problem_id for problem in problems] != [
                problem.problem_id for problem in expected_problems
            ]:
                raise RuntimeError("PPO backend received an unexpected episode order")
            prompt_lookup, dataset_rows = build_ppo_runtime_dataset_rows(
                self.config, tokenizer, problems, episode_records, scope
            )
            dataset = Dataset.from_list(dataset_rows)
            verifier = MathVerifier()

            def prompt_verifier(prompt_ids, completion):
                metadata = prompt_lookup.get(tuple(prompt_ids))
                if metadata is None:
                    raise RuntimeError("reward prompt not in fixed PPO lookup")
                return verifier(metadata["problem"], completion)

            evidence = PPOCompletionEvidenceRecorder(contract, episode_records)
            reward_model = PPOVerifierRewardModel(
                tokenizer,
                lambda _completion: None,
                lambda decoded: decoded,
                reward_policy_from_config(self.config),
                evidence_callback=lambda completion, evaluation: evidence.record_reward(
                    completion, evaluation, guard
                ),
                prompt_verifier=prompt_verifier,
            )
            trainer_class = ppo_guarded_trainer_class(
                guard,
                evidence,
                prompt_lookup,
                {
                    "max_new_tokens": self.config["generation"]["max_new_tokens"],
                    "temperature": self.config["generation"]["temperature"],
                    "top_p": self.config["generation"]["top_p"],
                },
                ordered_episode_records=(
                    episode_records if contract.profile == "ppo_matched_pilot" else None
                ),
                expected_contract=contract,
            )
            trainer = build_ppo_trainer(
                self.config,
                dataset,
                policy,
                None,
                reward_model,
                value_model,
                tokenizer,
                self.lifecycle.manager.run_dir,
                trainer_factory=trainer_class,
                cpu_only=False,
            )
            loader_contract = getattr(trainer, "ordered_loader_evidence", None)
            if contract.profile == "ppo_matched_pilot" and loader_contract is None:
                raise RuntimeError("PPO pilot sequential loader evidence is missing")
            if loader_contract is not None:
                self.lifecycle.persist("ppo_loader_contract.json", loader_contract)
            model_roles = audit_ppo_parameter_roles(
                policy,
                value_model,
                reward_model,
                ref_model=None,
                optimizer=trainer.optimizer,
            )
            model_roles.update(
                {
                    "policy_base_source": str(self.model_source.snapshot_path),
                    "value_base_source": str(self.model_source.snapshot_path),
                    "policy_and_value_base_objects_distinct": True,
                    "reference_frozen": True,
                    "reward_parameter_free": True,
                }
            )
            self.lifecycle.persist("model_roles.json", model_roles)
            output = trainer.train()
            if int(trainer.state.global_step) != 1:
                raise RuntimeError("PPO Trainer did not finish at global_step 1")
            log_history = [dict(row) for row in trainer.state.log_history]
            metrics = extract_ppo_metrics(log_history)
            metrics["trainer_output"] = (
                dict(output.metrics) if output is not None and hasattr(output, "metrics") else {}
            )
            completions = evidence.records()
            normalized = metrics["normalized"]

            def metric(name):
                item = normalized[name]
                return item["value"] if item["available"] else ""

            statuses = [row["canonical_status"] for row in completions]
            lengths = [row["exact_token_count"] for row in completions]
            self.lifecycle.manager.write_csv(
                "metrics.csv",
                [
                    {
                        "step": 1,
                        "reward": sum(row["scalar_reward"] for row in completions)
                        / contract.expected_completions,
                        "policy_loss": metric("policy_loss"),
                        "value_loss": metric("value_loss"),
                        "kl": metric("objective_kl"),
                        "entropy": metric("entropy"),
                        "correctness": statuses.count("verified_pass")
                        / contract.expected_completions,
                        "format_accuracy": 1
                        - statuses.count("format_error") / contract.expected_completions,
                        "parse_success_rate": 1
                        - statuses.count("format_error") / contract.expected_completions,
                        "cumulative_generated_tokens": sum(lengths),
                        "mean_completion_length": sum(lengths) / contract.expected_completions,
                    }
                ],
                (
                    "step",
                    "reward",
                    "policy_loss",
                    "value_loss",
                    "kl",
                    "entropy",
                    "correctness",
                    "format_accuracy",
                    "parse_success_rate",
                    "cumulative_generated_tokens",
                    "mean_completion_length",
                ),
            )
            checkpoint = write_authoritative_ppo_checkpoint(
                self.lifecycle.manager.run_dir / "checkpoint-1",
                policy,
                value_model,
                {
                    "global_step": int(trainer.state.global_step),
                    "episode": int(trainer.state.episode),
                    "updates": guard.updates,
                    "optimizer_steps": guard.optimizer_steps,
                },
                model_roles,
            )
            run_result.update(
                {
                    "checkpoint_dir": str(checkpoint),
                    "metrics": metrics,
                    "trainer_log_history": metrics["raw_log_history"],
                    "completions": completions,
                    "model_roles": model_roles,
                    "episode_records": episode_records,
                    "loader_contract": loader_contract,
                }
            )
            return run_result
        finally:
            del trainer, reward_model, value_model, policy, tokenizer
            gc.collect()
            if torch.cuda.is_initialized():
                torch.cuda.empty_cache()
            allocator_payload = allocator.finalize()
            self.lifecycle.persist("pytorch_allocator.json", allocator_payload)
            run_result["pytorch_allocator"] = allocator_payload


def execute_real_ppo(config):
    """Real PPO entry after CLI authorization; profile selection is hash-bound."""
    from math_rlvr.training.runtime_prompt_scope import prepare_runtime_prompt_preflight

    prompt_preflight = prepare_runtime_prompt_preflight(config, "ppo")
    source = require_local_snapshot()
    lifecycle = RealPPOLifecycle(config)
    return run_guarded_ppo(
        config,
        RealPPOBackend(config, lifecycle, source, prompt_preflight),
        lifecycle,
        RealPPOMonitor(lifecycle),
    )


def execute_real_ppo_smoke(config):
    """Backward-compatible Stage D entry, still protected by the smoke config hash."""
    return execute_real_ppo(config)


def assert_runtime_config_path():
    """Static import-time-free helper used by CPU tests."""
    return PPO_SMOKE_CONFIG
