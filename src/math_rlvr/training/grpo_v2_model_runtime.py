"""Delayed model-bound worker for the frozen GRPO-v2 seed-42 run."""

from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any

from math_rlvr.training.grpo_v2_runtime import (
    CHECKPOINT_STEPS,
    RUN_ROOT,
    GRPOV2ContractError,
    GRPOV2Observer,
    normalized_training_config,
    select_dev_checkpoint,
    validate_resume_checkpoint,
)


def _underlying_optimizer(optimizer):
    """Safely unwrap Accelerate without using wrapper identity as audit evidence."""
    current = optimizer
    for _ in range(4):
        wrapped = getattr(current, "optimizer", None)
        if wrapped is None or wrapped is current:
            return current
        current = wrapped
    raise GRPOV2ContractError("GRPO-v2 optimizer wrapper nesting is unexpected")


def _scheduler_position(scheduler) -> dict[str, int | None]:
    return {
        "last_epoch": getattr(scheduler, "last_epoch", None),
        "step_count": getattr(scheduler, "_step_count", None),
    }


def _scheduler_advanced(before: dict[str, int | None], after: dict[str, int | None]) -> bool:
    return any(
        before[name] is not None
        and after[name] is not None
        and after[name] > before[name]
        for name in before
    )


def initial_grpo_v2_optimizer_roles(policy) -> dict[str, Any]:
    """Record the valid post-constructor lazy state without touching ``None.state``."""
    from math_rlvr.training.formal_model_runtime import audit_grpo_parameter_roles

    roles = audit_grpo_parameter_roles(policy, optimizer=None)
    return {
        **roles,
        "lifecycle": "lazy_not_initialized",
        "optimizer_present": False,
        "scheduler_present": False,
        "sft_optimizer_state_loaded": False,
        "optimizer_state_entries_before_training": None,
    }


def grpo_v2_optimizer_lifecycle_callback(
    policy,
    roles: dict[str, Any],
    guard,
    *,
    fresh: bool,
    step_offset: int = 0,
    restore_training_state=None,
):
    """Audit the native lazy optimizer before and after its first update."""
    from transformers import TrainerCallback

    from math_rlvr.training.formal_model_runtime import audit_grpo_parameter_roles

    class OptimizerLifecycleCallback(TrainerCallback):
        def __init__(self):
            self.parameter_ids: set[int] | None = None
            self.underlying_optimizer = None
            self.scheduler_before: dict[str, int | None] | None = None
            self.first_step_verified = False

        def on_train_begin(self, args, state, control, **kwargs):
            optimizer = kwargs.get("optimizer")
            scheduler = kwargs.get("lr_scheduler")
            if optimizer is None or scheduler is None:
                raise GRPOV2ContractError(
                    "native GRPO optimizer/scheduler missing at on_train_begin"
                )
            if int(state.global_step) != step_offset:
                raise GRPOV2ContractError("GRPO-v2 optimizer audit step offset drift")
            if restore_training_state is not None:
                restore_training_state()
            underlying = _underlying_optimizer(optimizer)
            audit = audit_grpo_parameter_roles(policy, optimizer=underlying)
            state_entries = len(underlying.state)
            if fresh and state_entries:
                raise GRPOV2ContractError(
                    "fresh GRPO optimizer contains inherited pre-update state"
                )
            scheduler_before = _scheduler_position(scheduler)
            if fresh and (
                scheduler_before["last_epoch"] not in {-1, 0}
                or scheduler_before["step_count"] not in {0, 1}
            ):
                raise GRPOV2ContractError("fresh GRPO scheduler is not at its initial step")
            self.parameter_ids = {
                id(parameter)
                for group in underlying.param_groups
                for parameter in group["params"]
            }
            self.underlying_optimizer = underlying
            self.scheduler_before = scheduler_before
            roles.update(
                {
                    **audit,
                    "lifecycle": "native_created_pre_first_update",
                    "optimizer_present": True,
                    "scheduler_present": True,
                    "sft_optimizer_state_loaded": False,
                    "optimizer_state_entries_before_training": state_entries,
                    "optimizer_state_fresh_before_training": fresh and state_entries == 0,
                    "scheduler_position_before_training": scheduler_before,
                }
            )
            return control

        def on_step_end(self, args, state, control, **kwargs):
            if int(state.global_step) != step_offset + 1 or self.first_step_verified:
                return control
            optimizer = kwargs.get("optimizer")
            scheduler = kwargs.get("lr_scheduler")
            if optimizer is None or scheduler is None:
                raise GRPOV2ContractError("GRPO optimizer/scheduler disappeared after first step")
            underlying = _underlying_optimizer(optimizer)
            if underlying is not self.underlying_optimizer:
                raise GRPOV2ContractError(
                    "GRPO optimizer changed before first step completed"
                )
            audit = audit_grpo_parameter_roles(policy, optimizer=underlying)
            parameter_ids = {
                id(parameter)
                for group in underlying.param_groups
                for parameter in group["params"]
            }
            if parameter_ids != self.parameter_ids:
                raise GRPOV2ContractError("GRPO optimizer parameter roles changed after first step")
            state_entries = len(underlying.state)
            if state_entries == 0:
                raise GRPOV2ContractError(
                    "GRPO optimizer state did not materialize after first step"
                )
            if any(id(parameter) not in parameter_ids for parameter in underlying.state):
                raise GRPOV2ContractError("GRPO optimizer state contains an unexpected parameter")
            scheduler_after = _scheduler_position(scheduler)
            if self.scheduler_before is None or not _scheduler_advanced(
                self.scheduler_before, scheduler_after
            ):
                raise GRPOV2ContractError("GRPO scheduler did not advance after first step")
            if not (
                guard.optimizer_steps == step_offset + 1
                and guard.global_steps == step_offset + 1
                and guard.updates == step_offset + 1
            ):
                raise GRPOV2ContractError("GRPO first-step counters are inconsistent")
            roles.update(
                {
                    **audit,
                    "lifecycle": "native_first_update_verified",
                    "optimizer_state_entries_after_first_update": state_entries,
                    "optimizer_state_materialized": True,
                    "scheduler_position_after_first_update": scheduler_after,
                    "first_update_counters": {
                        "optimizer_steps": guard.optimizer_steps,
                        "global_steps": guard.global_steps,
                        "updates": guard.updates,
                    },
                }
            )
            self.first_step_verified = True
            return control

    return OptimizerLifecycleCallback()


def _training_rows(design, normalized, contract, tokenizer, *, completed_updates=0):
    from math_rlvr.dataset import MathProblem
    from math_rlvr.prompt import ExperimentScope, format_training_problem, render_training_prompt

    public = {
        row["problem_id"]: row
        for row in map(json.loads, Path(design["data"]["manifest"]).read_text().splitlines())
    }
    trusted_path = Path(
        json.loads(Path("configs/grpo_v2/runtime_registry.json").read_text())["grpo_v2"][
            "trusted_train_manifest_path"
        ]
    )
    trusted = {
        row["problem_id"]: row for row in map(json.loads, trusted_path.read_text().splitlines())
    }
    curriculum = json.loads(Path(design["data"]["curriculum"]).read_text())["positions"]
    result, problems = [], {}
    for row in curriculum[completed_updates * 4 :]:
        source = public[row["problem_id"]]
        gold = trusted[row["problem_id"]]
        if source["content_hash"] != gold["content_hash"]:
            raise GRPOV2ContractError("trusted/public train content hash mismatch")
        problem = MathProblem(
            **{
                key: source[key]
                for key in (
                    "problem_id",
                    "source",
                    "prompt",
                    "category",
                    "difficulty",
                    "split",
                    "source_index",
                    "content_hash",
                    "metadata",
                )
            },
            gold_answer=gold["gold_answer"],
        )
        rendered = render_training_prompt(
            tokenizer, problem, normalized, scope=ExperimentScope.MAIN_FORMAL
        )
        prompt_ids = tokenizer(rendered, add_special_tokens=False, truncation=False)["input_ids"]
        if len(prompt_ids) > contract.max_prompt_length:
            raise GRPOV2ContractError("GRPO-v2 train prompt exceeds frozen cap")
        if len(prompt_ids) + contract.max_completion_length > contract.max_sequence_length:
            raise GRPOV2ContractError("GRPO-v2 train prompt exceeds frozen sequence ceiling")
        prompt_hash = __import__("hashlib").sha256(rendered.encode()).hexdigest()
        result.append(
            {
                "prompt": format_training_problem(
                    problem, normalized, scope=ExperimentScope.MAIN_FORMAL
                ),
                "problem_id": problem.problem_id,
                "prompt_hash": prompt_hash,
                "curriculum_position": row["position"],
                "curriculum_update": row["update"],
                "curriculum_slot": row["slot"],
            }
        )
        problems[problem.problem_id] = problem
    expected_ids = contract.problem_ids[completed_updates * 4 :]
    if tuple(row["problem_id"] for row in result) != expected_ids:
        raise GRPOV2ContractError("GRPO-v2 dataset differs from frozen curriculum order")
    return result, problems


def _load_policy_state(policy, adapter_file: Path) -> None:
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file

    state = load_file(str(adapter_file))
    if not state or any("lora_" not in name for name in state):
        raise GRPOV2ContractError("GRPO-v2 initial/resume policy state is not LoRA-only")
    set_peft_model_state_dict(policy, state)


def _with_parseable_metric(
    metric: dict[str, Any], rows: list[dict[str, Any]]
) -> dict[str, Any]:
    denominator = len(rows)
    if denominator != 16:
        raise GRPOV2ContractError("GRPO-v2 parseable metric requires one complete update")
    statuses = [row.get("canonical_status") for row in rows]
    numerator = sum(status in {"wrong_answer", "verified_pass"} for status in statuses)
    return {
        **metric,
        "parseable_rate": numerator / denominator,
        "parseable_rate_available": True,
        "parseable_rate_reason": None,
        "parseable_rate_numerator": numerator,
        "parseable_rate_denominator": denominator,
        "parseable_rate_definition": (
            "canonical_status in {wrong_answer, verified_pass} divided by all update completions"
        ),
    }


def _dev_summary(step: int, aggregate: dict[str, Any]) -> dict[str, Any]:
    overall = aggregate
    return {
        "checkpoint_step": step,
        "canonical_pass_rate": overall["candidate0_pass_at_1"]["value"],
        "parseable_rate": overall["parseable_rate"]["value"],
        "format_rate": overall["format_rate"]["value"],
        "truncation_rate": overall["truncation_rate"]["value"],
        "correct_numerator": overall["candidate0_pass_at_1"]["numerator"],
        "problem_denominator": 128,
    }


def execute_real_grpo_v2(
    design: dict[str, Any],
    *,
    identity: dict[str, Any],
    contract,
    model_source,
    warmstart_checkpoint: Path,
    run_dir: Path,
    environment: dict[str, str],
    resume_checkpoint: Path | None = None,
) -> dict[str, Any]:
    import torch
    from datasets import Dataset

    from math_rlvr.artifacts.manager import ArtifactManager
    from math_rlvr.artifacts.monitor import ResourceMonitor
    from math_rlvr.evaluation.grpo_v2_dev_model_runtime import execute_dev_worker
    from math_rlvr.evaluation.grpo_v2_dev_runtime import load_dev_contract
    from math_rlvr.rewards.staged import reward_policy_from_config
    from math_rlvr.training.builders import build_grpo_trainer, load_policy_and_tokenizer
    from math_rlvr.training.formal_model_runtime import (
        _normal_completion_rows,
        _normal_metrics,
        _restore_trusted_rng,
        _restore_trusted_training_state,
        _write_checkpoint,
    )
    from math_rlvr.training.formal_runtime import FormalOnlineGuard, create_formal_backup
    from math_rlvr.training.resource_evidence import CudaAllocatorEvidence
    from math_rlvr.training.trl_compat import (
        CompletionEvidenceRecorder,
        guarded_trainer_class,
        optimizer_guard_callback,
    )
    from math_rlvr.verifier import MathVerifier

    normalized = normalized_training_config(design, contract)
    completed_updates = 0
    validated_resume = None
    if resume_checkpoint is not None:
        validated_resume = validate_resume_checkpoint(resume_checkpoint, contract, run_dir)
        completed_updates = validated_resume.step
    manager = None
    if resume_checkpoint is None:
        command = (
            "HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src "
            "python -m math_rlvr.training.grpo_v2 --config configs/grpo_v2/grpo_v2_seed42.json "
            f"--warmstart-checkpoint {warmstart_checkpoint} --run-dir {run_dir} "
            "--execute --confirm-grpo-v2"
        )
        manager = ArtifactManager(
            "grpo_v2",
            "grpo",
            model_source.repo_id,
            42,
            command,
            design,
            run_id=run_dir.name,
        )
        if manager.run_dir != run_dir:
            raise GRPOV2ContractError("GRPO-v2 ArtifactManager run directory mismatch")
        manager.write_json("resolved_config.json", design)
        manager.write_json(
            "runtime_identity.json",
            {
                **contract.as_dict(),
                "environment": environment,
                "warmstart_handoff": identity["warmstart_handoff"],
                "sft_optimizer_inherited": False,
            },
        )
    observer = GRPOV2Observer(contract, run_dir, run_dir.name)
    if validated_resume:
        observer.restore(validated_resume)
    online_guard = (
        FormalOnlineGuard.from_resume_manifest(contract, validated_resume.manifest)
        if validated_resume
        else FormalOnlineGuard(contract)
    )
    monitor = ResourceMonitor(run_dir / "resource_metrics.csv", interval=0.25)
    allocator = CudaAllocatorEvidence(torch.cuda)
    trainer = policy = tokenizer = evidence = None
    try:
        monitor.start()
        allocator.start()
        policy, tokenizer = load_policy_and_tokenizer(normalized, model_source)
        if validated_resume:
            _load_policy_state(
                policy, validated_resume.checkpoint / "policy_adapter/adapter_model.safetensors"
            )
        else:
            _load_policy_state(policy, warmstart_checkpoint / "adapter/adapter_model.safetensors")
        rows, problem_map = _training_rows(
            design, normalized, contract, tokenizer, completed_updates=completed_updates
        )
        verifier = MathVerifier()
        reward_policy = reward_policy_from_config(normalized)
        evidence = CompletionEvidenceRecorder(contract)
        metric_prefix = tuple(validated_resume.metrics_prefix) if validated_resume else ()
        if validated_resume:
            evidence.restore_prefix([dict(row) for row in validated_resume.completion_prefix])

        def reward_func(completions, problem_id, **_kwargs):
            values = []
            for completion, problem_id_value in zip(completions, problem_id, strict=True):
                text = completion if isinstance(completion, str) else completion[-1]["content"]
                problem = problem_map[problem_id_value]
                evaluation = reward_policy.evaluate(
                    text, lambda candidate, problem=problem: verifier(problem, candidate)
                )
                evidence.record_reward(
                    problem_id_value,
                    text,
                    evaluation.canonical_result,
                    evaluation.scalar_reward,
                    evaluation.to_dict(),
                )
                online_guard.record_reward(
                    evaluation.canonical_result, evaluation.scalar_reward, evaluation.to_dict()
                )
                values.append(evaluation.scalar_reward)
            return values

        roles = initial_grpo_v2_optimizer_roles(policy)
        if roles["policy_trainable_parameters"] != 4_358_144:
            raise GRPOV2ContractError("GRPO-v2 trainable parameter count drift")
        roles.update(
            {
                "initial_policy_adapter_sha256": contract.warmstart_adapter_sha256,
                "initial_policy_adapter_role": "policy",
                "grpo_optimizer_initialization": (
                    "fresh" if not validated_resume else "same_run_resume"
                ),
            }
        )

        def update_callback(bound_trainer, step):
            partial = _normal_completion_rows(evidence.partial_records(), contract)
            expected = step * 16
            if len(partial) != expected:
                raise GRPOV2ContractError("GRPO-v2 update completion prefix is incomplete")
            update_rows = partial[expected - 16 : expected]
            metric = _normal_metrics(
                [dict(row) for row in bound_trainer.state.log_history],
                partial,
                contract,
                start_update=step,
            )[0]
            observer.update(step, update_rows, _with_parseable_metric(metric, update_rows))

        def checkpoint_callback(bound_trainer, step):
            if observer.guard.updates != step:
                raise GRPOV2ContractError(
                    "GRPO-v2 checkpoint callback preceded atomic update evidence"
                )
            return _write_checkpoint(
                run_dir / f"checkpoint-{step}",
                contract,
                policy,
                None,
                evidence,
                roles,
                step,
                trainer=bound_trainer,
                online_guard=online_guard,
                metric_prefix=metric_prefix,
            )

        trainer_class = guarded_trainer_class(
            online_guard,
            evidence,
            checkpoint_callback=checkpoint_callback,
            update_callback=update_callback,
            step_offset=completed_updates,
        )
        trainer = build_grpo_trainer(
            normalized,
            Dataset.from_list(rows),
            reward_func,
            run_dir,
            model=policy,
            tokenizer=tokenizer,
            trainer_factory=trainer_class,
            cpu_only=False,
            model_source=model_source,
        )
        trainer.add_callback(optimizer_guard_callback(online_guard, step_offset=completed_updates))
        restore_training_state = (
            (lambda: _restore_trusted_training_state(validated_resume, trainer))
            if validated_resume
            else None
        )
        trainer.add_callback(
            grpo_v2_optimizer_lifecycle_callback(
                policy,
                roles,
                online_guard,
                fresh=validated_resume is None,
                step_offset=completed_updates,
                restore_training_state=restore_training_state,
            )
        )
        if validated_resume:
            _restore_trusted_rng(validated_resume)
        trainer.train()
        if roles.get("lifecycle") != "native_first_update_verified":
            raise GRPOV2ContractError("GRPO-v2 native optimizer lifecycle was not verified")
        if int(trainer.state.global_step) != 128:
            raise GRPOV2ContractError("GRPO-v2 Trainer did not finish at step 128")
        online_counters = online_guard.assert_complete()
        if observer.guard.updates != 128 or len(observer.completions) != 2048:
            raise GRPOV2ContractError("GRPO-v2 incremental evidence is incomplete")
        for step in CHECKPOINT_STEPS:
            if step > completed_updates:
                observer.checkpoint(step, run_dir / f"checkpoint-{step}")
        model_roles = dict(roles)
        training_allocator = allocator.finalize()
        monitor.stop()
        training_resource = {"available": True, **monitor.summary()}
        trainer = policy = tokenizer = evidence = None
        if any(value is not None for value in (trainer, policy, tokenizer, evidence)):
            raise GRPOV2ContractError("GRPO-v2 training references were not released")
        gc.collect()
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()

        dev_config, dev_identity, dev_rows = load_dev_contract()
        dev_summaries = []
        for step in CHECKPOINT_STEPS:
            dev_run_dir = RUN_ROOT / f"grpo_v2_dev_seed42_checkpoint{step}_{run_dir.name}"
            if dev_run_dir.exists():
                raise GRPOV2ContractError("GRPO-v2 dev run directory conflict")
            checkpoint_identity = contract.checkpoint_identity(run_id=run_dir.name, step=step)
            outcome = execute_dev_worker(
                config=dev_config,
                identity=dev_identity,
                public_rows=dev_rows,
                mode="warmstart",
                checkpoint_identity=checkpoint_identity,
                model_source=model_source,
                run_dir=dev_run_dir,
                environment=environment,
                adapter_directory=run_dir / f"checkpoint-{step}" / "policy_adapter",
                evidence_mode="grpo_v2",
                checkpoint_step=step,
                backup_path=(
                    Path("/root/autodl-fs/math-rlvr-backups") / f"{dev_run_dir.name}.tar.gz"
                ),
            )
            rows_for_step = outcome.pop("completion_rows")
            observer.validation(step, rows_for_step)
            summary = _dev_summary(step, outcome["aggregate"])
            summary.update(
                {
                    "run_id": outcome["run_id"],
                    "generated_tokens": outcome["counters"]["generated_tokens"],
                }
            )
            dev_summaries.append(summary)
        selected = select_dev_checkpoint(dev_summaries)
        counters = {
            **observer.guard.assert_complete(),
            "microsteps": online_counters["microsteps"],
            "online_counters": online_counters,
        }
        payload = {
            "status": "success",
            "run_id": run_dir.name,
            "counters": counters,
            "model_roles": model_roles,
            "training_resource": training_resource,
            "pytorch_allocator": training_allocator,
            "dev_summaries": dev_summaries,
            "selected_checkpoint_step": selected,
            "selection_rule": ["canonical_pass", "parseable", "format", "truncation", "earlier"],
            "hidden_test_accesses": 0,
        }
        for name, value in (
            ("checkpoint_inventory.json", observer.checkpoint_inventory),
            ("dev_validation_summary.json", dev_summaries),
            ("selected_checkpoint.json", {"checkpoint_step": selected, "source": "dev_v2_only"}),
            ("summary.json", payload),
        ):
            (run_dir / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
        (run_dir / "dev_validation.jsonl").write_text(
            "".join(json.dumps(row, allow_nan=False) + "\n" for row in observer.validation_rows)
        )
        if manager:
            manager.finalize("success", counters=counters, summary=payload)
        backup_path = Path("/root/autodl-fs/math-rlvr-backups") / f"{run_dir.name}.tar.gz"
        backup = create_formal_backup(run_dir, backup_path)
        if manager:
            manager.write_json("backup_manifest.json", {"verified": True, **backup})
            manager.checksums()
        return {**payload, "backup": backup}
    except BaseException as exc:
        if manager:
            manager.finalize(
                "failure",
                failed_stage="grpo_v2_training_or_dev",
                exception=exc,
                stop_reason=str(exc),
                counters=observer.guard.snapshot(),
            )
            backup_path = (
                Path("/root/autodl-fs/math-rlvr-backups") / f"{run_dir.name}.failure.tar.gz"
            )
            backup = create_formal_backup(run_dir, backup_path)
            manager.write_json("backup_manifest.json", {"verified": True, **backup})
            manager.checksums()
        raise
    finally:
        monitor.stop()
        gc.collect()
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()
