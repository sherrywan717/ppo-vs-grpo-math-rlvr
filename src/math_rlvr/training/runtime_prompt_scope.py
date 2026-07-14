"""CPU-only prompt/scope preflight shared by delayed PPO and GRPO runtimes."""

from __future__ import annotations

from typing import Any

from math_rlvr.prompt import format_training_problem, prompt_version_from_config
from math_rlvr.training.execution_contract import (
    expected_run_contract_for_config,
    validated_scope_from_config,
)
from math_rlvr.training.pilot import rendered_prompt_payload_sha256


def prepare_runtime_prompt_preflight(
    config: dict[str, Any], algorithm: str
) -> dict[str, Any]:
    """Render protected Python message rows before model/tokenizer/CUDA imports."""
    scope = validated_scope_from_config(config, algorithm)
    contract = expected_run_contract_for_config(config, algorithm)
    if scope.scope is not contract.experiment_scope:
        raise ValueError("CPU resolved scope differs from ExpectedRunContract scope")
    if scope.expected_run_profile != contract.profile:
        raise ValueError("CPU resolved profile differs from ExpectedRunContract profile")
    prompt_version = prompt_version_from_config(config, scope.scope)
    if prompt_version != contract.prompt_version:
        raise ValueError("prompt selector version differs from ExpectedRunContract")

    if algorithm == "ppo":
        from math_rlvr.training.guarded_ppo import ppo_execution_problems_and_episodes

        problems, episode_records = ppo_execution_problems_and_episodes(config, contract)
        rows = []
        for problem, episode in zip(problems, episode_records, strict=True):
            messages = format_training_problem(problem, config, scope=scope.scope)
            rendered_hash = rendered_prompt_payload_sha256(problem, prompt_version)
            if rendered_hash != episode["rendered_prompt_hash"]:
                raise ValueError("PPO preflight rendered prompt hash drift")
            rows.append(
                {
                    "episode_position": episode["episode_position"],
                    "problem_id": problem.problem_id,
                    "generation_index": episode["generation_index"],
                    "pair_key": episode["pair_key"],
                    "problem_hash": episode["problem_hash"],
                    "rendered_prompt_hash": rendered_hash,
                    "messages": messages,
                }
            )
        expected_rows = contract.expected_completions
    elif algorithm == "grpo":
        from math_rlvr.training.guarded_grpo import select_grpo_execution_problems

        problems = select_grpo_execution_problems(config, contract)
        expected_hashes = {}
        if contract.profile == "grpo_matched_pilot":
            from math_rlvr.training.pilot import pilot_episode_records

            expected_hashes = {
                row["problem_id"]: row["rendered_prompt_hash"]
                for row in pilot_episode_records("grpo", config["experiment"]["seed"])
            }
        rows = []
        for position, problem in enumerate(problems):
            messages = format_training_problem(problem, config, scope=scope.scope)
            rendered_hash = rendered_prompt_payload_sha256(problem, prompt_version)
            if expected_hashes and rendered_hash != expected_hashes[problem.problem_id]:
                raise ValueError("GRPO preflight rendered prompt hash drift")
            rows.append(
                {
                    "prompt_position": position,
                    "problem_id": problem.problem_id,
                    "rendered_prompt_hash": rendered_hash,
                    "messages": messages,
                }
            )
        expected_rows = contract.expected_prompt_count
    else:
        raise ValueError("runtime prompt preflight algorithm must be ppo or grpo")

    if len(rows) != expected_rows:
        raise ValueError("runtime prompt preflight row count mismatch")
    pair_keys = list(contract.pair_keys)
    if len(pair_keys) != contract.expected_completions or len(set(pair_keys)) != len(pair_keys):
        raise ValueError("runtime prompt preflight comparison keys mismatch")
    scope_value = scope.scope.value
    return {
        "algorithm": algorithm,
        "validated_scope": scope.to_dict(),
        "cpu_resolved_scope": scope_value,
        "expected_run_contract_scope": contract.experiment_scope.value,
        "delayed_runtime_scope": scope_value,
        "prompt_selector_scope": scope_value,
        "expected_run_profile": contract.profile,
        "prompt_version": prompt_version,
        "prompt_sha256": contract.prompt_sha256,
        "rendered_row_count": len(rows),
        "comparison_keys": pair_keys,
        "rows": rows,
    }


def validate_runtime_prompt_preflight(
    config: dict[str, Any], algorithm: str, evidence: dict[str, Any]
):
    """Revalidate immutable pre-model evidence and return its sole scope object."""
    expected = prepare_runtime_prompt_preflight(config, algorithm)
    if evidence != expected:
        raise ValueError("delayed runtime prompt preflight evidence mismatch")
    scope = validated_scope_from_config(config, algorithm)
    scope_values = {
        evidence["cpu_resolved_scope"],
        evidence["expected_run_contract_scope"],
        evidence["delayed_runtime_scope"],
        evidence["prompt_selector_scope"],
        scope.scope.value,
    }
    if len(scope_values) != 1:
        raise ValueError("runtime prompt scope layers disagree")
    return scope
