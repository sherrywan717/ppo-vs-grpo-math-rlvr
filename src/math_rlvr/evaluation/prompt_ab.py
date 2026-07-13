"""Fail-closed, generation-only v0/v1 prompt diagnostic.

Importing this module never loads a model or initializes CUDA. The real backend is
imported only after both CLI confirmations and every static/local gate pass.
"""

from __future__ import annotations

import argparse
import math
import os
import statistics
import subprocess
import time
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from math_rlvr.config import load_config
from math_rlvr.dataset import MathProblem, load_manifest
from math_rlvr.evaluation.prompt_ab_evidence import (
    EvidenceContractError,
    build_group_reward_evidence,
    build_paired_comparison,
    load_capability_manifest,
)
from math_rlvr.parser import ParsedCompletion, parse_completion
from math_rlvr.prompt import (
    PROMPT_V0_GRPO_SMOKE,
    PROMPT_V1_STRICT_CONCISE,
)
from math_rlvr.rewards.result import DEFAULT_REWARD_POLICY, RewardResult, RewardStatus
from math_rlvr.training.guarded_grpo import assert_json_safe
from math_rlvr.training.model_source import (
    DEFAULT_CACHE_ROOT,
    PINNED_REPO_ID,
    PINNED_REVISION,
    ValidatedModelSource,
)
from math_rlvr.verifier import MathVerifier

DIAGNOSTIC_CONFIG = Path("configs/diagnostics/prompt_ab.yaml")
CAPABILITY_MANIFEST = Path("configs/diagnostics/prompt_ab_capabilities.json")
CONDITION_ORDER = ("v0", "v1")
PROMPT_VERSIONS = (PROMPT_V0_GRPO_SMOKE, PROMPT_V1_STRICT_CONCISE)


class DiagnosticAuthorizationError(RuntimeError):
    pass


class GenerationBudgetExceededError(RuntimeError):
    pass


@dataclass(frozen=True)
class GeneratedSequence:
    input_token_count: int
    completion_ids: list[int]
    decoded_text: str
    eos_reached: bool


def split_completion_ids(
    sequence_ids: list[int],
    *,
    padded_input_width: int,
    input_attention_mask: list[int],
    eos_token_id: int | None,
    pad_token_id: int | None,
) -> GeneratedSequence:
    """Split after padded input width; left padding never becomes completion evidence."""
    if padded_input_width <= 0 or len(sequence_ids) < padded_input_width:
        raise ValueError("invalid padded input width")
    if len(input_attention_mask) != padded_input_width or any(
        value not in (0, 1) for value in input_attention_mask
    ):
        raise ValueError("invalid input attention mask")
    completion = list(sequence_ids[padded_input_width:])
    eos_reached = False
    if eos_token_id is not None and eos_token_id in completion:
        completion = completion[: completion.index(eos_token_id) + 1]
        eos_reached = True
    elif pad_token_id is not None:
        while completion and completion[-1] == pad_token_id:
            completion.pop()
    return GeneratedSequence(
        input_token_count=sum(input_attention_mask),
        completion_ids=completion,
        decoded_text="",
        eos_reached=eos_reached,
    )


@dataclass
class GenerationBudgetGuard:
    max_conditions: int
    unique_prompts_per_condition: int
    completions_per_prompt: int
    completions_per_condition: int
    max_total_completions: int
    max_tokens_per_completion: int
    max_total_generated_tokens: int
    deadline: float
    max_peak_vram_gib: float
    clock: Callable[[], float] = time.monotonic
    condition_prompts: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    problem_counts: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    condition_counts: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    total_completions: int = 0
    total_generated_tokens: int = 0
    exceeded_reason: str | None = None
    started_at: float | None = None

    def __post_init__(self):
        if self.started_at is None:
            self.started_at = self.clock()

    def _fail(self, reason: str):
        self.exceeded_reason = reason
        raise GenerationBudgetExceededError(reason)

    def _check_time(self):
        if self.clock() > self.deadline:
            self._fail("generation diagnostic exceeded 120-second deadline")

    def record(
        self, condition: str, problem_id: str, completion_ids: list[int], peak_vram_gib=None
    ):
        self._check_time()
        if condition not in CONDITION_ORDER:
            self._fail("unexpected diagnostic condition")
        if len(set(self.condition_counts) | {condition}) > self.max_conditions:
            self._fail("condition cap exceeded")
        tokens = len(completion_ids)
        if tokens > self.max_tokens_per_completion:
            self._fail("per-completion token cap exceeded")
        if self.total_completions + 1 > self.max_total_completions:
            self._fail("total completion cap exceeded")
        if self.total_generated_tokens + tokens > self.max_total_generated_tokens:
            self._fail("total generated-token cap exceeded")
        key = f"{condition}:{problem_id}"
        if self.problem_counts[key] + 1 > self.completions_per_prompt:
            self._fail("per-prompt completion cap exceeded")
        if self.condition_counts[condition] + 1 > self.completions_per_condition:
            self._fail("per-condition completion cap exceeded")
        if peak_vram_gib is not None and peak_vram_gib > self.max_peak_vram_gib:
            self._fail("nvidia-smi peak VRAM stop gate exceeded")
        self.condition_prompts[condition].add(problem_id)
        self.problem_counts[key] += 1
        self.condition_counts[condition] += 1
        self.total_completions += 1
        self.total_generated_tokens += tokens

    def assert_success(self):
        self._check_time()
        if tuple(self.condition_counts.get(name, 0) for name in CONDITION_ORDER) != (
            self.completions_per_condition,
            self.completions_per_condition,
        ):
            self._fail("incomplete per-condition completion counts")
        for condition in CONDITION_ORDER:
            if len(self.condition_prompts[condition]) != self.unique_prompts_per_condition:
                self._fail("unique prompt count mismatch")
            if any(
                self.problem_counts[f"{condition}:{problem_id}"] != self.completions_per_prompt
                for problem_id in self.condition_prompts[condition]
            ):
                self._fail("per-prompt completion count mismatch")
        if self.total_completions != self.max_total_completions:
            self._fail("total completion count mismatch")

    def snapshot(self) -> dict[str, Any]:
        now = self.clock()
        payload = {
            "limits": {
                "conditions": self.max_conditions,
                "unique_prompts_per_condition": self.unique_prompts_per_condition,
                "completions_per_prompt": self.completions_per_prompt,
                "completions_per_condition": self.completions_per_condition,
                "total_completions": self.max_total_completions,
                "tokens_per_completion": self.max_tokens_per_completion,
                "total_generated_tokens": self.max_total_generated_tokens,
                "peak_vram_gib": self.max_peak_vram_gib,
            },
            "condition_counts": dict(self.condition_counts),
            "condition_prompt_ids": {
                key: sorted(value) for key, value in self.condition_prompts.items()
            },
            "problem_counts": dict(self.problem_counts),
            "total_completions": self.total_completions,
            "total_generated_tokens": self.total_generated_tokens,
            "elapsed_seconds": max(0.0, now - self.started_at),
            "exceeded_reason": self.exceeded_reason,
        }
        assert_json_safe(payload)
        return payload


class GenerationBackend(Protocol):
    backward_count: int
    optimizer_steps: int
    training_steps: int
    checkpoint_writes: int
    model_writes: int
    eval_called: bool
    inference_mode_used: bool
    parameters_frozen: bool

    def prepare(self) -> None: ...

    def render(self, problem: MathProblem, prompt_version: str) -> tuple[str, str]: ...

    def generate(
        self,
        prompt: str,
        *,
        seed: int,
        sampling: dict[str, Any],
        max_new_tokens: int,
    ) -> GeneratedSequence: ...

    def peak_vram_gib(self) -> float | None: ...

    def close(self) -> dict[str, Any] | None: ...


class ArtifactLifecycle(Protocol):
    backed_up: bool

    def start(self, config: dict, problems: list[MathProblem], seed_map: list[dict]) -> None: ...

    def persist_jsonl(self, name: str, rows: list[dict]) -> None: ...

    def persist_csv(self, name: str, rows: list[dict]) -> None: ...

    def persist(self, name: str, payload: Any) -> None: ...

    def finalize(self, summary: dict) -> None: ...

    def backup_and_verify(self, *, failure: bool = False) -> None: ...

    def publish_git_safe(self) -> None: ...


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Guarded generation-only prompt A/B diagnostic")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--confirm-prompt-diagnostic", action="store_true")
    parser.add_argument("--confirm-single-update", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def validate_diagnostic_config(config: dict, config_path: Path) -> None:
    if config_path.resolve() != DIAGNOSTIC_CONFIG.resolve():
        raise DiagnosticAuthorizationError("only the fixed prompt A/B diagnostic config is allowed")
    contract = (
        config.get("experiment", {}).get("name"),
        config.get("experiment", {}).get("mode"),
        config.get("experiment", {}).get("seed"),
        config.get("model", {}).get("name_or_path"),
        config.get("model", {}).get("revision"),
        config.get("model", {}).get("local_files_only"),
        config.get("model", {}).get("dtype"),
        config.get("model", {}).get("base_model_only"),
        tuple((row.get("name"), row.get("prompt_version")) for row in config.get("conditions", [])),
        config.get("data", {}).get("problem_ids"),
        config.get("generation", {}).get("completions_per_prompt"),
        config.get("generation", {}).get("max_new_tokens"),
        config.get("generation", {}).get("do_sample"),
        config.get("generation", {}).get("temperature"),
        config.get("generation", {}).get("top_p"),
        config.get("generation", {}).get("top_k"),
        config.get("generation", {}).get("repetition_penalty"),
    )
    expected = (
        "qwen25-05b-prompt-ab-generation-diagnostic",
        "generation_only",
        42,
        PINNED_REPO_ID,
        PINNED_REVISION,
        True,
        "bfloat16",
        True,
        tuple(zip(CONDITION_ORDER, PROMPT_VERSIONS, strict=True)),
        ["countdown:train:0", "countdown:train:1"],
        4,
        128,
        True,
        0.8,
        0.95,
        None,
        1.0,
    )
    if contract != expected:
        raise DiagnosticAuthorizationError("prompt A/B diagnostic contract mismatch")
    budget = config.get("budget", {})
    if budget != {
        "max_conditions": 2,
        "unique_prompts_per_condition": 2,
        "completions_per_condition": 8,
        "max_total_completions": 16,
        "max_total_generated_tokens": 2048,
        "max_wall_time_seconds": 120,
        "max_peak_vram_gib": 3.5,
        "gpu_hour_price_cny": 8.88,
    }:
        raise DiagnosticAuthorizationError("prompt A/B budget mismatch")
    if config.get("safety") != {
        "training": False,
        "backward": False,
        "optimizer": False,
        "checkpoint": False,
        "adapter": False,
        "retry": False,
    }:
        raise DiagnosticAuthorizationError("prompt A/B safety contract mismatch")


def require_clean_git() -> dict[str, str]:
    branch = subprocess.run(
        ["git", "branch", "--show-current"], capture_output=True, text=True, check=True
    ).stdout.strip()
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain=v1"], capture_output=True, text=True, check=True
    ).stdout
    if branch != "pivot/math-rlvr" or dirty:
        raise DiagnosticAuthorizationError("prompt A/B requires clean pivot/math-rlvr worktree")
    return {"branch": branch, "commit": commit}


def require_offline_env() -> dict[str, str]:
    names = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    values = {name: os.environ.get(name) for name in names}
    if set(values.values()) != {"1"}:
        raise DiagnosticAuthorizationError("HF and Transformers offline mode are required")
    return values


def require_local_snapshot(
    *, cache_root: Path = DEFAULT_CACHE_ROOT, snapshot_resolver=None
) -> ValidatedModelSource:
    return ValidatedModelSource.resolve(
        PINNED_REPO_ID,
        PINNED_REVISION,
        cache_root=cache_root,
        snapshot_resolver=snapshot_resolver,
    )


def select_problems(config: dict) -> list[MathProblem]:
    requested = config["data"]["problem_ids"]
    manifest = load_manifest(Path(config["data"]["manifest"]))
    by_id = {problem.problem_id: problem for problem in manifest}
    try:
        problems = [by_id[problem_id] for problem_id in requested]
    except KeyError as exc:
        raise DiagnosticAuthorizationError("fixed diagnostic problem is absent") from exc
    if len(problems) != 2 or len({p.problem_id for p in problems}) != 2:
        raise DiagnosticAuthorizationError("exactly two unique diagnostic problems required")
    if any(p.source != "countdown" or p.split != "train" for p in problems):
        raise DiagnosticAuthorizationError("diagnostic problems must be countdown train records")
    return problems


def matched_seed_map(config: dict, problems: list[MathProblem]) -> list[dict[str, Any]]:
    base = config["experiment"]["seed"]
    generations = config["generation"]["completions_per_prompt"]
    rows = []
    for condition in CONDITION_ORDER:
        for problem_index, problem in enumerate(problems):
            for generation_index in range(generations):
                seed = base + problem_index * generations + generation_index
                rows.append(
                    {
                        "condition": condition,
                        "problem_id": problem.problem_id,
                        "generation_index": generation_index,
                        "seed": seed,
                        "matched_seed": seed,
                        "python_seed": seed,
                        "torch_cpu_seed": seed,
                        "torch_cuda_seed": seed,
                    }
                )
    return rows


def completion_fields(
    problem: MathProblem, text: str
) -> tuple[dict[str, Any], RewardResult, float]:
    parsed = parse_completion(text)
    result = MathVerifier()(problem, text)
    if result.status == RewardStatus.INFRA_ERROR:
        raise RuntimeError(f"infra_error: {result.detail}")
    scalar = DEFAULT_REWARD_POLICY.to_scalar(result)
    if not math.isfinite(scalar):
        raise FloatingPointError("non-finite reward")
    parsed_ok = isinstance(parsed, ParsedCompletion)
    answer_pair = text.count("<answer>") == text.count("</answer>") == 1
    reasoning_pair = text.count("<reasoning>") == text.count("</reasoning>") == 1
    prose_outside = False
    if parsed_ok:
        prefix, _, suffix = text.partition("<reasoning>")
        _, _, suffix = suffix.partition("</answer>")
        prose_outside = bool(prefix.strip() or suffix.strip())
    else:
        stripped = text
        for tag in ("<reasoning>", "</reasoning>", "<answer>", "</answer>"):
            stripped = stripped.replace(tag, "")
        prose_outside = bool(stripped.strip()) and not (answer_pair and text.startswith("<answer>"))
    return (
        {
            "parser_status": "parsed" if parsed_ok else parsed.status.value,
            "parser_detail": "" if parsed_ok else parsed.detail,
            "reward_status": result.status.value,
            "scalar_reward": scalar,
            "verifier_detail": result.detail,
            "format_valid": parsed_ok,
            "reasoning_open": text.count("<reasoning>") == 1,
            "reasoning_close": text.count("</reasoning>") == 1,
            "answer_open": text.count("<answer>") == 1,
            "answer_close": text.count("</answer>") == 1,
            "answer_only": answer_pair and not reasoning_pair,
            "prose_outside_envelope": prose_outside,
            "expression_valid": result.status
            not in {
                RewardStatus.FORMAT_ERROR,
                RewardStatus.PARSE_ERROR,
                RewardStatus.INVALID_EXPRESSION,
                RewardStatus.RESOURCE_LIMIT,
            },
            "number_usage_valid": result.status
            not in {
                RewardStatus.FORMAT_ERROR,
                RewardStatus.PARSE_ERROR,
                RewardStatus.INVALID_EXPRESSION,
                RewardStatus.INVALID_NUMBER_USAGE,
                RewardStatus.RESOURCE_LIMIT,
            },
            "final_answer_correct": result.status == RewardStatus.VERIFIED_PASS,
        },
        result,
        scalar,
    )


def _rate(rows: list[dict], key: str) -> float:
    return sum(bool(row[key]) for row in rows) / len(rows)


def condition_metrics(rows: list[dict]) -> dict[str, dict[str, Any]]:
    output = {}
    for condition in CONDITION_ORDER:
        selected = [row for row in rows if row["condition"] == condition]
        rewards = [row["scalar_reward"] for row in selected]
        lengths = [row["exact_completion_token_count"] for row in selected]
        grouped = defaultdict(list)
        for row in selected:
            grouped[row["problem_id"]].append(row["scalar_reward"])
        group_variances = {key: statistics.pvariance(value) for key, value in grouped.items()}
        pass_at_1 = sum(group[0] == 1.0 for group in grouped.values()) / len(grouped)
        pass_at_4 = sum(any(value == 1.0 for value in group) for group in grouped.values()) / len(
            grouped
        )
        output[condition] = {
            "completions": len(selected),
            "complete_envelope_rate": _rate(selected, "format_valid"),
            "reasoning_open_rate": _rate(selected, "reasoning_open"),
            "reasoning_close_rate": _rate(selected, "reasoning_close"),
            "answer_open_rate": _rate(selected, "answer_open"),
            "answer_close_rate": _rate(selected, "answer_close"),
            "answer_only_rate": _rate(selected, "answer_only"),
            "prose_outside_envelope_rate": _rate(selected, "prose_outside_envelope"),
            "truncation_rate": _rate(selected, "truncated_at_128"),
            "format_accuracy": _rate(selected, "format_valid"),
            "valid_expression_rate": _rate(selected, "expression_valid"),
            "number_usage_accuracy": _rate(selected, "number_usage_valid"),
            "pass_at_1": pass_at_1,
            "pass_at_4": pass_at_4,
            "reward_status_counts": dict(Counter(row["reward_status"] for row in selected)),
            "reward_mean": statistics.mean(rewards),
            "reward_std": statistics.pstdev(rewards),
            "reward_variance": statistics.pvariance(rewards),
            "group_reward_variance": group_variances,
            "nonzero_advantage_potential_groups": sum(v > 0 for v in group_variances.values()),
            "zero_advantage_group_count": sum(len(set(v)) == 1 for v in grouped.values()),
            "nonzero_variance_group_count": sum(len(set(v)) > 1 for v in grouped.values()),
            "completion_token_mean": statistics.mean(lengths),
            "completion_token_median": statistics.median(lengths),
            "completion_token_max": max(lengths),
        }
    return output


def candidate_qualification(metrics: dict[str, dict[str, Any]]) -> dict[str, Any]:
    v0, v1 = metrics["v0"], metrics["v1"]
    checks = {
        "complete_envelope_rate_higher_than_v0": v1["complete_envelope_rate"]
        > v0["complete_envelope_rate"],
        "at_least_one_complete_envelope": v1["complete_envelope_rate"] > 0,
        "truncation_rate_not_increased": v1["truncation_rate"] <= v0["truncation_rate"],
        "at_least_one_nonzero_variance_group": v1["nonzero_advantage_potential_groups"] >= 1,
    }
    wrong_only = set(v1["reward_status_counts"]) == {RewardStatus.WRONG_ANSWER.value}
    return {
        "v1_format_improved": checks["complete_envelope_rate_higher_than_v0"],
        "v1_has_complete_envelope": checks["at_least_one_complete_envelope"],
        "v1_no_truncation_regression": checks["truncation_rate_not_increased"],
        "v1_has_nonzero_group_reward_variance": checks[
            "at_least_one_nonzero_variance_group"
        ],
        "v1_eligible_for_grpo_review": all(checks.values()),
        "eligible_for_later_grpo_review": all(checks.values()),
        "checks": checks,
        "all_wrong_answer_warning": wrong_only,
        "auto_activate": False,
    }


def _assert_no_training_effects(backend: GenerationBackend):
    if not (backend.eval_called and backend.inference_mode_used and backend.parameters_frozen):
        raise RuntimeError("eval/inference/frozen-parameter contract violated")
    counters = {
        "backward_count": backend.backward_count,
        "optimizer_steps": backend.optimizer_steps,
        "global_training_steps": backend.training_steps,
        "checkpoint_writes": backend.checkpoint_writes,
        "model_or_adapter_writes": backend.model_writes,
    }
    if any(counters.values()):
        raise RuntimeError(f"generation-only safety counter violated: {counters}")
    return counters


def run_diagnostic(
    config: dict,
    backend: GenerationBackend,
    lifecycle: ArtifactLifecycle,
    *,
    clock: Callable[[], float] = time.monotonic,
    completion_analyzer=completion_fields,
    defer_parent_finalization: bool = False,
) -> dict[str, Any]:
    problems = select_problems(config)
    seeds = matched_seed_map(config, problems)
    start = clock()
    budget = config["budget"]
    guard = GenerationBudgetGuard(
        max_conditions=budget["max_conditions"],
        unique_prompts_per_condition=budget["unique_prompts_per_condition"],
        completions_per_prompt=config["generation"]["completions_per_prompt"],
        completions_per_condition=budget["completions_per_condition"],
        max_total_completions=budget["max_total_completions"],
        max_tokens_per_completion=config["generation"]["max_new_tokens"],
        max_total_generated_tokens=budget["max_total_generated_tokens"],
        deadline=start + budget["max_wall_time_seconds"],
        max_peak_vram_gib=budget["max_peak_vram_gib"],
        clock=clock,
        started_at=start,
    )
    rows: list[dict[str, Any]] = []
    result: dict[str, Any]
    backend_closed = False

    def close_backend() -> dict[str, Any]:
        nonlocal backend_closed
        if backend_closed:
            return {}
        backend_closed = True
        return backend.close() or {
            "available": False,
            "unavailable_reason": "backend did not expose allocator evidence",
        }

    try:
        lifecycle.start(config, problems, seeds)
        backend.prepare()
        sampling = {
            key: config["generation"][key]
            for key in ("do_sample", "temperature", "top_p", "top_k", "repetition_penalty")
        }
        for seed_row in seeds:
            condition = seed_row["condition"]
            problem = next(p for p in problems if p.problem_id == seed_row["problem_id"])
            prompt_version = dict(
                (row["name"], row["prompt_version"]) for row in config["conditions"]
            )[condition]
            prompt, rendered_hash = backend.render(problem, prompt_version)
            generated = backend.generate(
                prompt,
                seed=seed_row["seed"],
                sampling=sampling,
                max_new_tokens=config["generation"]["max_new_tokens"],
            )
            guard.record(
                condition,
                problem.problem_id,
                generated.completion_ids,
                backend.peak_vram_gib(),
            )
            forensic, _, _ = completion_analyzer(problem, generated.decoded_text)
            record = {
                **seed_row,
                "completion_index": len(rows),
                "condition_order": list(CONDITION_ORDER),
                "prompt_version": prompt_version,
                "prompt_hash": problem.content_hash,
                "rendered_prompt_sha256": rendered_hash,
                "input_token_count": generated.input_token_count,
                "completion_ids": generated.completion_ids,
                "exact_completion_token_count": len(generated.completion_ids),
                "sampling_parameters": sampling,
                "decoded_raw_text": generated.decoded_text,
                "truncated_at_128": len(generated.completion_ids)
                == config["generation"]["max_new_tokens"]
                and not generated.eos_reached,
                "eos_reached": generated.eos_reached,
                **forensic,
            }
            assert_json_safe(record)
            rows.append(record)
        guard.assert_success()
        counters = _assert_no_training_effects(backend)
        metrics = condition_metrics(rows)
        pairs = build_paired_comparison(rows)
        group_rewards = build_group_reward_evidence(rows)
        allocator = close_backend()
        summary = {
            "status": "pending_backup",
            "diagnostic_only": True,
            "training": False,
            "budget": guard.snapshot(),
            "safety_counters": counters,
            "condition_metrics": metrics,
            "per_problem_rewards": group_rewards,
            "paired_row_count": len(pairs),
            "candidate_qualification": candidate_qualification(metrics),
            "condition_order": list(CONDITION_ORDER),
            "seed_map": seeds,
            "completion_count": len(rows),
            "backed_up": False,
        }
        lifecycle.persist_jsonl("completions.jsonl", rows)
        lifecycle.persist("per_condition_metrics.json", metrics)
        lifecycle.persist("paired_comparison.json", pairs)
        lifecycle.persist_csv("paired_comparison.csv", pairs)
        lifecycle.persist("per_problem_rewards.json", group_rewards)
        lifecycle.persist("pytorch_allocator.json", allocator)
        lifecycle.persist("summary.json", summary)
        lifecycle.finalize(summary)
        if defer_parent_finalization:
            summary["status"] = "worker_complete"
            lifecycle.persist("summary.json", summary)
            lifecycle.finalize(summary)
            return summary
        lifecycle.backup_and_verify(failure=False)
        summary["backed_up"] = lifecycle.backed_up
        if not lifecycle.backed_up:
            raise RuntimeError("verified backup was not recorded")
        summary["status"] = "success"
        lifecycle.persist("summary.json", summary)
        lifecycle.finalize(summary)
        lifecycle.publish_git_safe()
        result = summary
    except Exception as exc:
        close_error = None
        try:
            allocator = close_backend()
        except Exception as cleanup_exc:
            close_error = f"{type(cleanup_exc).__name__}: {cleanup_exc}"
            allocator = {
                "available": False,
                "failure_phase": "backend_cleanup",
                "failure_reason": close_error,
            }
        result = {
            "status": "failure",
            "reason": f"{type(exc).__name__}: {exc}",
            "diagnostic_only": True,
            "training": False,
            "budget": guard.snapshot(),
            "completion_count": len(rows),
            "backed_up": False,
            "cleanup_error": close_error,
        }
        try:
            lifecycle.persist_jsonl("completions.jsonl", rows)
            lifecycle.persist("pytorch_allocator.json", allocator)
            lifecycle.persist("failure_report.json", result)
            lifecycle.persist("summary.json", result)
            lifecycle.finalize(result)
        except Exception as final_exc:
            result["finalization_error"] = f"{type(final_exc).__name__}: {final_exc}"
            try:
                lifecycle.persist(
                    "minimal_failure_record.json",
                    {
                        "status": "failure",
                        "failure_phase": "artifact_finalization",
                        "exception_type": type(final_exc).__name__,
                        "reason": str(final_exc),
                    },
                )
            except Exception:
                pass
        if defer_parent_finalization:
            return result
        try:
            lifecycle.backup_and_verify(failure=True)
            result["backed_up"] = lifecycle.backed_up
            lifecycle.persist("summary.json", result)
            lifecycle.finalize(result)
        except Exception as backup_exc:
            result["backup_error"] = f"{type(backup_exc).__name__}: {backup_exc}"
            result["backed_up"] = lifecycle.backed_up
    finally:
        if not backend_closed:
            close_backend()
    return result


def main(
    argv=None,
    *,
    execute_fn=None,
    git_probe=require_clean_git,
    snapshot_probe=require_local_snapshot,
    offline_probe=require_offline_env,
    capability_probe=lambda: load_capability_manifest(CAPABILITY_MANIFEST),
) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    validate_diagnostic_config(config, args.config)
    if args.confirm_single_update:
        raise DiagnosticAuthorizationError("training confirmation cannot authorize generation")
    if not args.generate_only and not args.confirm_prompt_diagnostic:
        print("Prompt A/B preflight passed; dry-run only, no generation performed.")
        return 0
    if not (args.generate_only and args.confirm_prompt_diagnostic):
        raise DiagnosticAuthorizationError("both generation-only confirmations are required")
    try:
        capability_probe()
    except EvidenceContractError as exc:
        raise DiagnosticAuthorizationError(str(exc)) from exc
    git_info = git_probe()
    offline_probe()
    source = snapshot_probe()
    if source is None:
        raise DiagnosticAuthorizationError("validated local snapshot is required")
    if execute_fn is None:
        from math_rlvr.evaluation.prompt_ab_supervisor import execute_supervised_diagnostic

        execute_fn = execute_supervised_diagnostic
    result = execute_fn(config=config, source=source, git_info=git_info)
    if result.get("status") != "success":
        raise RuntimeError(result.get("reason", "prompt A/B diagnostic failed"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
