"""Delayed model-bound assembly for the frozen formal PPO/GRPO suite."""

from __future__ import annotations

import copy
import json
import math
import random
import statistics
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from math_rlvr.training.formal import FORMAL_MODEL, FORMAL_REVISION
from math_rlvr.training.formal_runtime import (
    FORMAL_RESUME_SCHEMA,
    CompletedTrainerBackend,
    FormalOnlineGuard,
    FormalRuntimeError,
    ValidatedFormalResume,
    create_formal_backup,
    execute_formal_training,
    formal_checkpoint_inventory,
    formal_episode_records,
    formal_run_contract,
    formal_training_problems,
    formal_valid_answer_metric,
    validate_formal_resume_checkpoint,
    validate_formal_runtime_prompt_preflight,
    write_formal_checkpoint_artifact_manifest,
)
from math_rlvr.training.model_source import ValidatedModelSource

RUN_ROOT = Path("/root/autodl-tmp/runs/math_rlvr")
BACKUP_ROOT = Path("/root/autodl-fs/math-rlvr-backups")


def _validate_bound_inputs(
    config: dict[str, Any],
    algorithm: str,
    model_source: ValidatedModelSource,
    prompt_preflight: dict[str, Any],
    authorization: dict[str, Any],
):
    contract = formal_run_contract(config)
    if (
        contract.algorithm != algorithm
        or model_source.repo_id != FORMAL_MODEL
        or model_source.revision != FORMAL_REVISION
        or model_source.local_files_only is not True
        or authorization.get("algorithm") != algorithm
        or authorization.get("seed") != contract.seed
        or authorization.get("config_sha256") != contract.config_sha256
        or authorization.get("active_suite_sha256") != contract.active_suite_sha256
    ):
        raise FormalRuntimeError("formal model-bound execution identity mismatch")
    validate_formal_runtime_prompt_preflight(config, algorithm, prompt_preflight)
    return contract


def audit_grpo_parameter_roles(policy, reward_model=None, optimizer=None) -> dict[str, Any]:
    trainable = {
        id(parameter): name
        for name, parameter in policy.named_parameters()
        if parameter.requires_grad
    }
    if not trainable or any("lora_" not in name for name in trainable.values()):
        raise FormalRuntimeError("formal GRPO trainables must be policy LoRA only")
    optimizer_ids = (
        {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
        if optimizer is not None
        else set()
    )
    if optimizer is not None and optimizer_ids != set(trainable):
        raise FormalRuntimeError("formal GRPO optimizer is not the exact policy-LoRA set")
    if reward_model is not None and any(
        parameter.requires_grad for parameter in reward_model.parameters()
    ):
        raise FormalRuntimeError("formal GRPO reward role unexpectedly trainable")
    return {
        "policy_trainable_names": sorted(trainable.values()),
        "policy_trainable_parameters": sum(
            parameter.numel() for parameter in policy.parameters() if parameter.requires_grad
        ),
        "optimizer_parameter_tensors": len(optimizer_ids),
        "optimizer_exact_role_match": optimizer is not None,
        "reward_trainable_parameters": 0,
        "base_trainable_parameters": 0,
    }


def _normal_completion_rows(records, contract):
    normalized = []
    for index, raw in enumerate(records):
        row = dict(raw)
        if contract.algorithm == "ppo":
            row["completion_ids"] = list(row.pop("response_token_ids"))
            row["completion_mask"] = list(row.pop("response_mask"))
        row["raw_completion"] = row.get("verifier_input", row.get("decoded_completion"))
        row["update"] = index // contract.completions_per_update + 1
        if getattr(contract, "profile", None) == "grpo_v2_1p5b":
            row["curriculum_position"] = index // 4 + 1
            row["curriculum_update"] = index // contract.completions_per_update + 1
            row["curriculum_slot"] = (index // 4) % 4
        if row.get("pair_key") != contract.pair_keys[index]:
            raise FormalRuntimeError("formal trainer completion order drift")
        normalized.append(row)
    return normalized


def _number(row: dict[str, Any], names: tuple[str, ...], label: str) -> float:
    for name in names:
        value = row.get(name)
        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
        ):
            return float(value)
    raise FormalRuntimeError(f"formal Trainer did not expose required {label}")


def _optional_number(
    row: dict[str, Any],
    names: tuple[str, ...],
    label: str,
    *,
    allow_non_finite: bool = False,
) -> tuple[float | None, str | None]:
    raw_key = next((name for name in names if name in row), None)
    if raw_key is None:
        return None, None
    value = row[raw_key]
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        if allow_non_finite and isinstance(value, (int, float)) and not isinstance(value, bool):
            return None, raw_key
        raise FormalRuntimeError(f"formal Trainer exposed invalid optional {label}")
    return float(value), raw_key


def _normal_metrics(log_history, records, contract, *, start_update=1):
    if not records or len(records) % contract.completions_per_update or start_update < 1:
        raise FormalRuntimeError("formal metric normalization prefix is invalid")
    last_update = len(records) // contract.completions_per_update
    expected_rows = last_update - start_update + 1
    if expected_rows < 1:
        raise FormalRuntimeError("formal metric normalization range is empty")
    metric_rows = [
        row
        for row in log_history
        if any(key in row for key in ("loss", "loss/policy_avg", "loss/value_avg"))
    ]
    if len(metric_rows) < expected_rows:
        raise FormalRuntimeError("formal Trainer log history has fewer metric rows than expected")
    metric_rows = metric_rows[-expected_rows:]
    output = []
    for update, trainer_row in enumerate(metric_rows, start=start_update):
        start = (update - 1) * contract.completions_per_update
        completion_rows = records[start : start + contract.completions_per_update]
        rewards = [float(row["scalar_reward"]) for row in completion_rows]
        lengths = [int(row["exact_token_count"]) for row in completion_rows]
        texts = [str(row["raw_completion"]) for row in completion_rows]
        statuses = [
            row.get("canonical_status", row.get("reward_status")) for row in completion_rows
        ]
        groups = [rewards[offset : offset + 4] for offset in range(0, 16, 4)]
        text_groups = [texts[offset : offset + 4] for offset in range(0, 16, 4)]
        variances = [statistics.pvariance(group) for group in groups]
        unique_rates = [len(set(group)) / len(group) for group in text_groups]
        zero_advantage_groups = sum(value == 0 for value in variances)
        eos_available = all(isinstance(row.get("eos_reached"), bool) for row in completion_rows)
        truncation_available = all(
            isinstance(row.get("truncated"), bool) for row in completion_rows
        )
        policy_loss = None
        value_loss = None
        if contract.algorithm == "ppo":
            policy_loss = _number(trainer_row, ("loss/policy_avg", "policy_loss"), "policy loss")
            value_loss = _number(trainer_row, ("loss/value_avg", "value_loss"), "value loss")
            loss = policy_loss + 0.1 * value_loss
            loss_definition = "policy_loss + TRL_0.24.0_default_vf_coef_0.1 * value_loss"
            entropy_names = ("policy/entropy_avg", "objective/entropy")
            entropy_mask = "none in TRL policy/entropy_avg reduction"
            entropy_aggregation = (
                "unmasked mean over response-axis training logits per microbatch, then mean "
                "over PPO epoch/minibatch/gradient-accumulation cells"
            )
            entropy_excludes_pad = False
            unified_reason = (
                "TRL 0.24.0 PPO policy/entropy_avg is not response-mask weighted; computing "
                "the unified token mean would require intrusive or memory-expensive logits work"
            )
        else:
            loss = _number(trainer_row, ("loss",), "GRPO loss")
            loss_definition = "TRL 0.24.0 GRPO logged loss"
            entropy_names = ("entropy",)
            entropy_mask = "TRL completion_mask (true generated response tokens through EOS)"
            entropy_aggregation = (
                "completion-mask token mean per GRPO training microbatch, then Trainer log mean"
            )
            entropy_excludes_pad = True
            unified_reason = (
                "native GRPO entropy is microbatch-mean aggregated and cannot be made identical "
                "to PPO without an additional or intrusive logits computation"
            )
        entropy, entropy_raw_key = _optional_number(
            trainer_row, entropy_names, "native policy entropy"
        )
        entropy_reason = (
            None if entropy is not None else "reviewed TRL log row did not expose native entropy"
        )
        kl_names = (
            ("policy/approxkl_avg", "objective/kl") if contract.algorithm == "ppo" else ("kl",)
        )
        kl, kl_raw_key = _optional_number(trainer_row, kl_names, "KL")
        clip_fraction, clip_raw_key = _optional_number(
            trainer_row,
            (
                ("policy/clipfrac_avg",)
                if contract.algorithm == "ppo"
                else ("clip_ratio/region_mean",)
            ),
            "clip fraction",
        )
        ratio, ratio_raw_key = _optional_number(
            trainer_row, ("val/ratio",) if contract.algorithm == "ppo" else (), "ratio"
        )
        ratio_variance, ratio_variance_raw_key = _optional_number(
            trainer_row,
            ("val/ratio_var",) if contract.algorithm == "ppo" else (),
            "ratio variance",
            allow_non_finite=True,
        )
        grad_norm, grad_norm_raw_key = _optional_number(
            trainer_row, ("grad_norm", "train/grad_norm"), "grad norm"
        )
        policy_grad_norm, policy_grad_norm_raw_key = _optional_number(
            trainer_row,
            ("policy_grad_norm", "train/policy_grad_norm"),
            "policy grad norm",
        )
        value_grad_norm, value_grad_norm_raw_key = _optional_number(
            trainer_row,
            ("value_grad_norm", "train/value_grad_norm"),
            "value grad norm",
        )
        status_counts = dict(sorted(Counter(str(status) for status in statuses).items()))
        optional_reason = "reviewed TRL 0.24.0 per-update log row did not expose this metric"
        metrics = {
            "reward_mean": statistics.fmean(rewards),
            "reward_std": statistics.pstdev(rewards),
            "reward_variance": statistics.pvariance(rewards),
            "loss": loss,
            "total_loss": loss,
            "total_loss_definition": loss_definition,
            "grad_norm": grad_norm,
            "grad_norm_available": grad_norm is not None,
            "grad_norm_reason": None if grad_norm is not None else optional_reason,
            "grad_norm_raw_metric_key": grad_norm_raw_key,
            "policy_grad_norm": policy_grad_norm,
            "policy_grad_norm_available": policy_grad_norm is not None,
            "policy_grad_norm_reason": (
                None if policy_grad_norm is not None else optional_reason
            ),
            "policy_grad_norm_raw_metric_key": policy_grad_norm_raw_key,
            "value_grad_norm": value_grad_norm,
            "value_grad_norm_available": value_grad_norm is not None,
            "value_grad_norm_reason": None if value_grad_norm is not None else optional_reason,
            "value_grad_norm_raw_metric_key": value_grad_norm_raw_key,
            "entropy": entropy,
            "policy_entropy_mean": entropy,
            "policy_entropy_mean_available": entropy is not None,
            "policy_entropy_mean_reason": entropy_reason,
            "policy_entropy_std": None,
            "policy_entropy_std_available": False,
            "policy_entropy_std_reason": (
                "reviewed TRL 0.24.0 log row emits only an aggregated entropy mean"
            ),
            "response_token_entropy_mean": None,
            "response_token_entropy_mean_available": False,
            "response_token_entropy_mean_reason": unified_reason,
            "entropy_metric_source": "TRL 0.24.0 native trainer metric",
            "entropy_raw_metric_key": entropy_raw_key,
            "entropy_logits_or_log_probabilities": "current policy training logits",
            "entropy_response_axis_only": True,
            "entropy_excludes_prompt": True,
            "entropy_excludes_pad": entropy_excludes_pad,
            "entropy_excludes_eos": False,
            "entropy_token_mask": entropy_mask,
            "entropy_aggregation": entropy_aggregation,
            "entropy_dtype": (
                "BF16 policy logits under formal config; scalar converted to Python float"
            ),
            "entropy_formula": "H=-sum_v softmax(logits)_v * log_softmax(logits)_v (nats)",
            "entropy_trl_version": "0.24.0",
            "entropy_transformers_version": "4.57.6",
            "entropy_extra_model_forward": False,
            "entropy_full_logits_persisted": False,
            "learning_rate": _number(trainer_row, ("learning_rate", "lr"), "learning rate"),
            "mean_completion_length": statistics.fmean(lengths),
            "completion_length_mean": statistics.fmean(lengths),
            "completion_length_std": statistics.pstdev(lengths),
            "eos_rate": (
                sum(bool(row["eos_reached"]) for row in completion_rows) / 16
                if eos_available
                else None
            ),
            "eos_rate_available": eos_available,
            "eos_rate_reason": None if eos_available else "completion EOS evidence unavailable",
            "truncation_rate": (
                sum(bool(row["truncated"]) for row in completion_rows) / 16
                if truncation_available
                else None
            ),
            "truncation_rate_available": truncation_available,
            "truncation_rate_reason": (
                None if truncation_available else "completion truncation evidence unavailable"
            ),
            "unique_completion_rate": statistics.fmean(unique_rates),
            "completion_duplicate_rate": 1.0 - statistics.fmean(unique_rates),
            "completion_diversity_definition": (
                "exact raw-completion UTF-8 equality within each four-sample problem group; "
                "rates are the mean of four group rates"
            ),
            "zero_advantage_fraction": zero_advantage_groups / 4,
            "zero_advantage_group_count": zero_advantage_groups,
            "group_rewards": groups,
            "group_reward_variances": variances,
            "verifier_status_counts": status_counts,
            "format_accuracy": 1 - statuses.count("format_error") / 16,
            "canonical_pass_rate": statuses.count("verified_pass") / 16,
            "generated_tokens": sum(lengths),
            "cumulative_generated_tokens": sum(
                int(row["exact_token_count"]) for row in records[: start + 16]
            ),
            "kl": kl,
            "kl_available": kl is not None,
            "kl_raw_key": kl_raw_key,
            "kl_unavailable_reason": (
                None
                if kl is not None
                else "GRPO beta=0.0; reference-model KL is not computed"
                if contract.algorithm == "grpo"
                else optional_reason
            ),
            "clip_fraction": clip_fraction,
            "clip_fraction_available": clip_fraction is not None,
            "clip_fraction_raw_key": clip_raw_key,
            "clip_fraction_reason": None if clip_fraction is not None else optional_reason,
            "ratio_mean": ratio,
            "ratio_available": ratio is not None,
            "ratio_raw_key": ratio_raw_key,
            "ratio_reason": None if ratio is not None else optional_reason,
            "ratio_variance": ratio_variance,
            "ratio_variance_available": ratio_variance is not None,
            "ratio_variance_raw_key": ratio_variance_raw_key,
            "ratio_variance_reason": (
                None
                if ratio_variance is not None
                else "TRL val/ratio_var was non-finite and is preserved as unavailable"
                if ratio_variance_raw_key is not None
                else optional_reason
            ),
            "advantage_mean": None,
            "advantage_std": None,
            "advantage_available": False,
            "advantage_reason": optional_reason,
            "return_mean": None,
            "return_std": None,
            "return_available": False,
            "return_reason": optional_reason,
        }
        metrics.update(formal_valid_answer_metric(completion_rows))
        if policy_loss is not None:
            metrics["policy_loss"] = policy_loss
            metrics["value_loss"] = value_loss
            metrics["value_loss_coefficient"] = 0.1
        else:
            metrics["policy_loss"] = None
            metrics["value_loss"] = None
        output.append(metrics)
    return output


def _cpu_tensors(state):
    return {name: tensor.detach().cpu().contiguous() for name, tensor in state.items()}


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _rng_json_payload(torch_module) -> dict[str, Any]:
    import numpy as np

    python_state = random.getstate()
    numpy_state = np.random.get_state()
    cuda_count = (
        len(torch_module.cuda.get_rng_state_all()) if torch_module.cuda.is_available() else 0
    )
    return {
        "python": {
            "version": python_state[0],
            "internal": list(python_state[1]),
            "gauss_next": python_state[2],
        },
        "numpy": {
            "bit_generator": numpy_state[0],
            "keys": numpy_state[1].tolist(),
            "position": numpy_state[2],
            "has_gauss": numpy_state[3],
            "cached_gaussian": numpy_state[4],
        },
        "torch_cpu_rng": True,
        "torch_cuda_rng_device_count": cuda_count,
    }


def _write_trusted_training_state(root: Path, trainer, global_step: int) -> None:
    import torch
    from safetensors.torch import save_file

    if trainer.optimizer is None or trainer.lr_scheduler is None:
        raise FormalRuntimeError("formal checkpoint requires optimizer and scheduler state")
    torch.save(trainer.optimizer.state_dict(), root / "optimizer.pt")
    torch.save(trainer.lr_scheduler.state_dict(), root / "scheduler.pt")
    (root / "rng_state.json").write_text(
        json.dumps(_rng_json_payload(torch), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    rng_tensors = {"cpu": torch.random.get_rng_state().detach().cpu().contiguous()}
    if torch.cuda.is_available():
        rng_tensors.update(
            {
                f"cuda_{index}": state.detach().cpu().contiguous()
                for index, state in enumerate(torch.cuda.get_rng_state_all())
            }
        )
    save_file(rng_tensors, str(root / "torch_rng.safetensors"))
    local_step = int(trainer.state.global_step)
    trainer.state.global_step = global_step
    try:
        trainer.state.save_to_json(str(root / "trainer_state.json"))
    finally:
        trainer.state.global_step = local_step


def _restore_python_numpy_rng(payload: dict[str, Any]) -> None:
    import numpy as np

    python_state = payload.get("python")
    numpy_state = payload.get("numpy")
    if not isinstance(python_state, dict) or not isinstance(numpy_state, dict):
        raise FormalRuntimeError("formal checkpoint JSON RNG state is malformed")
    random.setstate(
        (
            int(python_state["version"]),
            tuple(int(value) for value in python_state["internal"]),
            python_state["gauss_next"],
        )
    )
    np.random.set_state(
        (
            str(numpy_state["bit_generator"]),
            np.asarray(numpy_state["keys"], dtype=np.uint32),
            int(numpy_state["position"]),
            int(numpy_state["has_gauss"]),
            float(numpy_state["cached_gaussian"]),
        )
    )


def _restore_trusted_training_state(
    validated: ValidatedFormalResume, trainer
) -> None:
    import torch

    root = validated.checkpoint
    optimizer_state = torch.load(root / "optimizer.pt", map_location="cpu", weights_only=True)
    scheduler_state = torch.load(root / "scheduler.pt", map_location="cpu", weights_only=True)
    trainer.optimizer.load_state_dict(optimizer_state)
    trainer.lr_scheduler.load_state_dict(scheduler_state)


def _restore_trusted_rng(validated: ValidatedFormalResume) -> None:
    import torch
    from safetensors.torch import load_file

    root = validated.checkpoint
    rng_payload = json.loads((root / "rng_state.json").read_text(encoding="utf-8"))
    rng_tensors = load_file(str(root / "torch_rng.safetensors"), device="cpu")
    expected_cuda = int(rng_payload.get("torch_cuda_rng_device_count", -1))
    expected_keys = {"cpu"} | {f"cuda_{index}" for index in range(expected_cuda)}
    if set(rng_tensors) != expected_keys:
        raise FormalRuntimeError("formal checkpoint torch RNG inventory mismatch")
    _restore_python_numpy_rng(rng_payload)
    torch.random.set_rng_state(rng_tensors["cpu"])
    if expected_cuda:
        if not torch.cuda.is_available() or torch.cuda.device_count() != expected_cuda:
            raise FormalRuntimeError("formal checkpoint CUDA RNG device identity mismatch")
        torch.cuda.set_rng_state_all(
            [rng_tensors[f"cuda_{index}"] for index in range(expected_cuda)]
        )


def _restore_adapter_state(
    validated: ValidatedFormalResume, policy, value_model
) -> None:
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file

    root = validated.checkpoint
    policy_state = load_file(str(root / "policy_adapter/adapter_model.safetensors"))
    if not policy_state or any("lora_" not in name for name in policy_state):
        raise FormalRuntimeError("formal resume policy state is not LoRA-only")
    set_peft_model_state_dict(policy, policy_state)
    if validated.manifest["algorithm"] == "ppo":
        if value_model is None:
            raise FormalRuntimeError("formal PPO resume requires the value model")
        value_state = load_file(str(root / "value_adapter/adapter_model.safetensors"))
        head_state = load_file(str(root / "value_head/value_head.safetensors"))
        if (
            not value_state
            or any("lora_" not in name for name in value_state)
            or not head_state
            or any("score" not in name for name in head_state)
        ):
            raise FormalRuntimeError("formal PPO resume value role state is invalid")
        set_peft_model_state_dict(value_model, {**value_state, **head_state})


def _write_checkpoint(
    root,
    contract,
    policy,
    value_model,
    recorder,
    model_roles,
    global_step,
    *,
    trainer,
    online_guard,
    metric_prefix,
):
    from peft import get_peft_model_state_dict
    from safetensors.torch import save_file

    if root.exists() or global_step not in contract.checkpoint_steps:
        raise FormalRuntimeError("formal checkpoint path/step is not new and frozen")
    policy_state = _cpu_tensors(get_peft_model_state_dict(policy))
    policy_adapter = {name: tensor for name, tensor in policy_state.items() if "lora_" in name}
    if not policy_adapter or set(policy_state) != set(policy_adapter):
        raise FormalRuntimeError("formal policy checkpoint is not LoRA-only")
    policy_dir = root / "policy_adapter"
    policy_dir.mkdir(parents=True)
    policy.peft_config["default"].save_pretrained(policy_dir)
    save_file(policy_adapter, str(policy_dir / "adapter_model.safetensors"))
    if contract.algorithm == "ppo":
        value_state = _cpu_tensors(get_peft_model_state_dict(value_model))
        value_adapter = {name: tensor for name, tensor in value_state.items() if "lora_" in name}
        value_head = {name: tensor for name, tensor in value_state.items() if "score" in name}
        if (
            not value_adapter
            or not value_head
            or set(value_state) != set(value_adapter) | set(value_head)
        ):
            raise FormalRuntimeError("formal PPO value checkpoint role partition failed")
        value_dir = root / "value_adapter"
        head_dir = root / "value_head"
        value_dir.mkdir()
        head_dir.mkdir()
        value_model.peft_config["default"].save_pretrained(value_dir)
        save_file(value_adapter, str(value_dir / "adapter_model.safetensors"))
        save_file(value_head, str(head_dir / "value_head.safetensors"))
        (head_dir / "config.json").write_text(
            json.dumps({"architecture": "scalar_score_head", "num_labels": 1}) + "\n"
        )
    partial_raw = recorder.partial_records()
    partial = _normal_completion_rows(partial_raw, contract)
    expected_count = global_step * contract.completions_per_update
    if len(partial) != expected_count:
        raise FormalRuntimeError("formal checkpoint completion prefix is incomplete")
    tokens = sum(int(row["exact_token_count"]) for row in partial)
    start_update = len(metric_prefix) + 1
    suffix_metrics = _normal_metrics(
        [dict(row) for row in trainer.state.log_history],
        partial,
        contract,
        start_update=start_update,
    )
    normalized_suffix = []
    for update, row in enumerate(suffix_metrics, start=start_update):
        evidence = {"update": update, **row}
        if getattr(contract, "profile", None) == "grpo_v2_1p5b":
            evidence.update(
                {
                    "optimizer_step": update,
                    "global_step": update,
                    "microsteps": update * 4,
                    "cumulative_completions": update * contract.completions_per_update,
                }
            )
        normalized_suffix.append(evidence)
    metrics = [dict(row) for row in metric_prefix] + normalized_suffix
    if len(metrics) != global_step:
        raise FormalRuntimeError("formal checkpoint metric prefix is incomplete")
    online_counters = online_guard.snapshot()
    if (
        online_counters["updates"] != global_step
        or online_counters["generated_tokens"] != tokens
    ):
        raise FormalRuntimeError("formal checkpoint online counters disagree with evidence")
    identity = contract.checkpoint_identity(run_id=root.parent.name, step=global_step)
    snapshot = {
        **identity,
        "schema": FORMAL_RESUME_SCHEMA,
        "project_created": True,
        "updates": global_step,
        "optimizer_steps": global_step,
        "global_steps": global_step,
        "completions": expected_count,
        "generated_tokens": tokens,
        "seen_pair_keys": list(contract.pair_keys[:expected_count]),
        "checkpoints": [step for step in contract.checkpoint_steps if step <= global_step],
        "validations": [],
        "base_weights_included": False,
        "optimizer_state_included": True,
        "scheduler_state_included": True,
        "rng_state_included": True,
        "sampler_position": {
            "comparison_key_count": expected_count,
            "ppo_episode_rows": expected_count if contract.algorithm == "ppo" else None,
            "grpo_prompt_rows": global_step * 4 if contract.algorithm == "grpo" else None,
        },
        "formal_runtime_counters": {
            "updates": global_step,
            "optimizer_steps": global_step,
            "global_steps": global_step,
            "completions": expected_count,
            "generated_tokens": tokens,
        },
        "online_counters": online_counters,
        "model_roles": model_roles,
    }
    _write_jsonl(root / "trainer_completion_prefix.jsonl", partial)
    _write_jsonl(root / "metrics_prefix.jsonl", metrics)
    _write_trusted_training_state(root, trainer, global_step)
    (root / "resume_manifest.json").write_text(
        json.dumps(snapshot, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    write_formal_checkpoint_artifact_manifest(root, contract, global_step)
    formal_checkpoint_inventory(root, contract, global_step)
    return root


def _dataset_rows(
    config, tokenizer, algorithm, scope, *, completed_updates=0
):
    from math_rlvr.prompt import format_training_problem, render_training_prompt

    problems = formal_training_problems()
    records = formal_episode_records(algorithm, config["experiment"]["seed"])
    problem_map = {problem.problem_id: problem for problem in problems}
    selected = (
        [problem_map[row["problem_id"]] for row in records] if algorithm == "ppo" else problems
    )
    prompt_lookup = {}
    rows = []
    for problem, record in zip(selected, records, strict=True):
        rendered = render_training_prompt(tokenizer, problem, config, scope=scope.scope)
        prompt_ids = tokenizer(rendered, add_special_tokens=False, truncation=False)["input_ids"]
        if len(prompt_ids) > config["generation"]["max_prompt_length"]:
            raise FormalRuntimeError("formal prompt exceeds frozen max_prompt_length")
        prompt_lookup[tuple(int(value) for value in prompt_ids)] = {
            "problem_id": problem.problem_id,
            "problem": problem,
        }
        if algorithm == "ppo":
            rows.append({"input_ids": prompt_ids, **record})
        else:
            rows.append(
                {
                    "prompt": format_training_problem(problem, config, scope=scope.scope),
                    "problem_id": problem.problem_id,
                    "prompt_hash": problem.content_hash,
                }
            )
    if algorithm == "grpo" and completed_updates:
        rows = rows[completed_updates * 4 :]
    return problems, records, prompt_lookup, rows


def _assemble_backend(
    config,
    model_source,
    prompt_preflight,
    run_dir,
    algorithm,
    *,
    resume_state=None,
):
    from datasets import Dataset

    from math_rlvr.rewards.staged import reward_policy_from_config
    from math_rlvr.training.builders import (
        audit_ppo_parameter_roles,
        build_grpo_trainer,
        build_ppo_trainer,
        load_policy_and_tokenizer,
        load_value_model,
    )
    from math_rlvr.training.trl_compat import (
        CompletionEvidenceRecorder,
        PPOCompletionEvidenceRecorder,
        guarded_trainer_class,
        optimizer_guard_callback,
        ppo_guarded_trainer_class,
    )
    from math_rlvr.verifier import MathVerifier

    contract = formal_run_contract(config)
    scope = validate_formal_runtime_prompt_preflight(config, algorithm, prompt_preflight)
    completed_updates = resume_state.step if resume_state is not None else 0
    metric_prefix = (
        tuple(dict(row) for row in resume_state.metrics_prefix) if resume_state else ()
    )
    online_guard = (
        FormalOnlineGuard.from_resume_manifest(contract, resume_state.manifest)
        if resume_state
        else FormalOnlineGuard(contract)
    )
    policy, tokenizer = load_policy_and_tokenizer(config, model_source)
    problems, records, prompt_lookup, rows = _dataset_rows(
        config, tokenizer, algorithm, scope, completed_updates=completed_updates
    )
    problem_map = {problem.problem_id: problem for problem in problems}
    verifier = MathVerifier()
    reward_policy = reward_policy_from_config(config)
    checkpoint_root = run_dir
    value_model = None
    reward_model = None

    if algorithm == "ppo":
        from math_rlvr.rewards.adapters import PPOVerifierRewardModel

        value_model = load_value_model(config, model_source)
        if resume_state is not None:
            _restore_adapter_state(resume_state, policy, value_model)
        evidence = PPOCompletionEvidenceRecorder(contract, records)
        if resume_state is not None:
            evidence.restore_prefix([dict(row) for row in resume_state.completion_prefix])

        def prompt_verifier(prompt_ids, completion):
            metadata = prompt_lookup.get(tuple(prompt_ids))
            if metadata is None:
                raise FormalRuntimeError("formal PPO reward prompt is not protected")
            return verifier(metadata["problem"], completion)

        reward_model = PPOVerifierRewardModel(
            tokenizer,
            lambda _completion: None,
            lambda decoded: decoded,
            reward_policy,
            evidence_callback=lambda completion, evaluation: evidence.record_reward(
                completion, evaluation, online_guard
            ),
            prompt_verifier=prompt_verifier,
        )
        model_roles = {"pending_optimizer_audit": True}
        update_observer_holder = {}

        def update_callback(trainer, step):
            observer = update_observer_holder.get("observer")
            if observer is None:
                raise FormalRuntimeError("formal PPO update observer is not bound")
            partial = _normal_completion_rows(evidence.partial_records(), contract)
            expected_count = step * contract.completions_per_update
            if len(partial) != expected_count:
                raise FormalRuntimeError("formal PPO update completion prefix is incomplete")
            metric = _normal_metrics(
                [dict(row) for row in trainer.state.log_history],
                partial,
                contract,
                start_update=step,
            )[0]
            start = expected_count - contract.completions_per_update
            observer.update(
                step,
                partial[start:expected_count],
                metric,
                optimizer_step=step,
                global_step=step,
            )

        def checkpoint_callback(trainer, step):
            return _write_checkpoint(
                run_dir / f"checkpoint-{step}",
                contract,
                policy,
                value_model,
                evidence,
                model_roles,
                step,
                trainer=trainer,
                online_guard=online_guard,
                metric_prefix=metric_prefix,
            )

        trainer_class = ppo_guarded_trainer_class(
            online_guard,
            evidence,
            prompt_lookup,
            {
                "max_new_tokens": config["generation"]["max_new_tokens"],
                "temperature": config["generation"]["temperature"],
                "top_p": config["generation"]["top_p"],
            },
            ordered_episode_records=records,
            expected_contract=contract,
            completed_updates=completed_updates,
            checkpoint_callback=checkpoint_callback,
            update_callback=update_callback,
        )
        trainer = build_ppo_trainer(
            config,
            Dataset.from_list(rows),
            policy,
            None,
            reward_model,
            value_model,
            tokenizer,
            run_dir,
            trainer_factory=trainer_class,
            cpu_only=False,
        )
        model_roles.clear()
        model_roles.update(
            audit_ppo_parameter_roles(
                policy, value_model, reward_model, ref_model=None, optimizer=trainer.optimizer
            )
        )
        if resume_state is not None:
            _restore_trusted_training_state(resume_state, trainer)
    else:
        if resume_state is not None:
            _restore_adapter_state(resume_state, policy, None)
        evidence = CompletionEvidenceRecorder(contract)
        if resume_state is not None:
            evidence.restore_prefix([dict(row) for row in resume_state.completion_prefix])

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

        model_roles = {"pending_optimizer_audit": True}

        def checkpoint_callback(trainer, step):
            return _write_checkpoint(
                run_dir / f"checkpoint-{step}",
                contract,
                policy,
                None,
                evidence,
                model_roles,
                step,
                trainer=trainer,
                online_guard=online_guard,
                metric_prefix=metric_prefix,
            )

        trainer_class = guarded_trainer_class(
            online_guard,
            evidence,
            checkpoint_callback=checkpoint_callback,
            step_offset=completed_updates,
        )
        trainer = build_grpo_trainer(
            config,
            Dataset.from_list(rows),
            reward_func,
            run_dir,
            model=policy,
            tokenizer=tokenizer,
            trainer_factory=trainer_class,
            cpu_only=False,
            model_source=model_source,
        )
        trainer.add_callback(
            optimizer_guard_callback(online_guard, step_offset=completed_updates)
        )
        model_roles.clear()
        model_roles.update(audit_grpo_parameter_roles(policy, optimizer=trainer.optimizer))
        if resume_state is not None:
            _restore_trusted_training_state(resume_state, trainer)

    def validation_runner(step):
        from math_rlvr.evaluation.formal_model_runtime import run_checkpoint_validation

        return run_checkpoint_validation(
            config=config,
            model_source=model_source,
            checkpoint=run_dir / f"checkpoint-{step}",
            algorithm=algorithm,
            seed=contract.seed,
            checkpoint_step=step,
        )

    backend = CompletedTrainerBackend(
        trainer=trainer,
        evidence_recorder=evidence,
        completion_normalizer=_normal_completion_rows,
        metric_normalizer=_normal_metrics,
        validation_runner=validation_runner,
        checkpoint_root=checkpoint_root,
        resume_checkpoint=resume_state.checkpoint if resume_state else None,
        metric_prefix=metric_prefix,
        before_train=(
            (lambda: _restore_trusted_rng(resume_state)) if resume_state else None
        ),
        online_guard=online_guard,
        update_observer_holder=(update_observer_holder if algorithm == "ppo" else None),
    )
    backend.model_roles = dict(model_roles)
    return backend


class _ResourceMonitoredBackend:
    """Start reviewed GPU evidence before delayed model/trainer construction."""

    def __init__(self, runtime_config, backend_factory, run_dir):
        self.runtime_config = runtime_config
        self.backend_factory = backend_factory
        self.run_dir = run_dir

    def execute(self, contract, observer, *, start_update):
        import torch

        from math_rlvr.artifacts.monitor import ResourceMonitor
        from math_rlvr.training.resource_evidence import CudaAllocatorEvidence

        monitor = ResourceMonitor(self.run_dir / "resource_metrics.csv", interval=0.25)
        allocator = CudaAllocatorEvidence(torch.cuda)
        monitor.start()
        try:
            allocator.start()
            backend = self.backend_factory()
            self.runtime_config["model_roles"] = dict(backend.model_roles)
            backend.execute(contract, observer, start_update=start_update)
        finally:
            monitor.stop()
            observer.resource_metrics = [dict(row) for row in monitor.rows]
            observer.resource_summary = {"available": True, **monitor.summary()}
            observer.pytorch_allocator = allocator.finalize()


def _finalize_artifacts(manager, run_dir, run_id, status, result=None, exception=None):
    counters = (result or {}).get("counters", {})
    manager.finalize(
        status,
        failed_stage=None if status == "success" else "formal_training",
        exception=exception,
        stop_reason=None if exception is None else str(exception),
        counters=counters,
    )
    suffix = "" if status == "success" else ".failure"
    backup = create_formal_backup(run_dir, BACKUP_ROOT / f"{run_id}{suffix}.tar.gz")
    manager.write_json("backup_manifest.json", {"verified": True, **backup})
    manager.checksums()
    return backup


def _execute(
    config,
    *,
    algorithm,
    model_source,
    prompt_preflight,
    authorization,
    resume_checkpoint=None,
    validated_resume=None,
):
    contract = _validate_bound_inputs(
        config, algorithm, model_source, prompt_preflight, authorization
    )
    runtime_config = copy.deepcopy(config)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{algorithm}_formal_1p5b_seed{contract.seed}_{stamp}"
    run_dir = RUN_ROOT / run_id
    manager = None
    if resume_checkpoint is not None:
        run_dir = resume_checkpoint.parent
        run_id = run_dir.name
        checked = validate_formal_resume_checkpoint(
            resume_checkpoint, contract, run_dir=run_dir
        )
        if validated_resume is not None and checked != validated_resume:
            raise FormalRuntimeError("formal resume validation changed across boundaries")
        validated_resume = checked
    elif validated_resume is not None:
        raise FormalRuntimeError("validated resume state requires a checkpoint path")
    else:
        from math_rlvr.artifacts.manager import ArtifactManager

        command = (
            f"PYTHONPATH=src python -m math_rlvr.training.{algorithm} "
            f"--config {contract.config_path} --execute --confirm-formal-{algorithm}"
        )
        manager = ArtifactManager(
            "formal_1p5b",
            algorithm,
            FORMAL_MODEL,
            contract.seed,
            command,
            runtime_config,
            run_id=run_id,
        )
        if manager.run_dir != run_dir:
            raise FormalRuntimeError("ArtifactManager formal run directory mismatch")
    runtime_config["formal_execution_authorization"] = dict(authorization)
    runtime_config["prompt_scope_preflight"] = copy.deepcopy(prompt_preflight)
    backend = _ResourceMonitoredBackend(
        runtime_config,
        lambda: _assemble_backend(
            runtime_config,
            model_source,
            prompt_preflight,
            run_dir,
            algorithm,
            resume_state=validated_resume,
        ),
        run_dir,
    )
    try:
        result = execute_formal_training(
            runtime_config,
            backend,
            run_dir=run_dir,
            run_id=run_id,
            resume_checkpoint=resume_checkpoint,
        )
    except Exception as exc:
        if manager is not None:
            _finalize_artifacts(manager, run_dir, run_id, "failure", exception=exc)
        raise
    if manager is not None:
        result["backup"] = _finalize_artifacts(
            manager, run_dir, run_id, "success", result=result
        )
    return result


def execute_real_formal_ppo(config, **kwargs):
    return _execute(config, algorithm="ppo", **kwargs)


def execute_real_formal_grpo(config, **kwargs):
    return _execute(config, algorithm="grpo", **kwargs)
