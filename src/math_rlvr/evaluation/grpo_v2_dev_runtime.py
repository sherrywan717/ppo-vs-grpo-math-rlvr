"""CPU-safe contracts for the matched single-candidate GRPO-v2 dev evaluation."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from math_rlvr.contracts import formal_parser_verifier_metadata
from math_rlvr.parser import ParsedCompletion, parse_completion
from math_rlvr.prompt import PROMPT_RENDERER_VERSION, PROMPT_V2_FORMAL_MATH, PROMPT_V2_SHA256
from math_rlvr.rewards.formal import FORMAL_REWARD_SHA256, FORMAL_REWARD_VERSION
from math_rlvr.rewards.result import RewardStatus
from math_rlvr.training.model_source import FORMAL_REPO_ID, FORMAL_REVISION
from math_rlvr.training.warmstart_runtime import file_sha256, validate_checkpoint

ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = Path("configs/grpo_v2/dev_evaluation_seed42.json")
REGISTRY_PATH = ROOT / "configs/grpo_v2/runtime_registry.json"
DATA_REGISTRY_PATH = ROOT / "configs/grpo_v2/data_registry.json"
RUN_ROOT = Path("/root/autodl-tmp/runs/math_rlvr")
EXPECTED_BRANCH = "improve/grpo-v2"
DEV_SEED_NAMESPACE = "grpo_v2/dev_v2/single_candidate"
EXPECTED_WARMSTART_RUN_ID = "warmstart_grpo_v2_seed42_20260722T051218Z"
EXPECTED_CHECKPOINT_ARTIFACT_SHA256 = (
    "507749d393f38690915a76228b4c53a8b5c8927d40aada9f2768a90334d892f0"
)
EXPECTED_ADAPTER_SHA256 = "44066dd13d8cfa4f5c40f10cad705eea617c37ce2e2f85ff5407751fb5a972b9"


class DevEvaluationContractError(RuntimeError):
    """A frozen matched-dev identity, budget, or evidence invariant failed."""


def canonical_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _validate_runtime_registry(registry: dict[str, Any]) -> None:
    claim = registry.get("registry_sha256")
    body = dict(registry)
    body.pop("registry_sha256", None)
    if claim != canonical_sha256(body):
        raise DevEvaluationContractError("GRPO-v2 runtime registry SHA mismatch")


def dev_generation_seed(problem_id: str, content_hash: str, seed: int = 42) -> int:
    digest = canonical_sha256(
        {
            "namespace": DEV_SEED_NAMESPACE,
            "evaluation_seed": seed,
            "problem_id": problem_id,
            "content_hash": content_hash,
        }
    )
    return int(digest[:16], 16) & ((1 << 63) - 1)


def load_dev_contract(
    config_path: Path = CONFIG_PATH,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    if config_path != CONFIG_PATH or config_path.is_absolute() or ".." in config_path.parts:
        raise DevEvaluationContractError("dev evaluation requires the exact config path")
    target = ROOT / config_path
    if target.is_symlink() or target.resolve(strict=True) != target:
        raise DevEvaluationContractError("dev evaluation config must be canonical and non-symlink")
    registry = json.loads(REGISTRY_PATH.read_text())
    _validate_runtime_registry(registry)
    identity = registry.get("dev_evaluation")
    if not isinstance(identity, dict) or file_sha256(target) != identity.get("config_sha256"):
        raise DevEvaluationContractError("dev evaluation config SHA mismatch")
    config = json.loads(target.read_text())
    data_registry = json.loads(DATA_REGISTRY_PATH.read_text())
    data_claim = data_registry.get("registry_sha256")
    data_body = dict(data_registry)
    data_body.pop("registry_sha256", None)
    if data_claim != canonical_sha256(data_body):
        raise DevEvaluationContractError("GRPO-v2 data registry SHA mismatch")
    if data_claim != identity.get("data_registry_sha256"):
        raise DevEvaluationContractError("dev evaluation data registry identity mismatch")
    expected = {
        "schema_version": 1,
        "experiment": "grpo_v2_matched_dev_seed42",
        "seed": 42,
    }
    if any(config.get(key) != value for key, value in expected.items()):
        raise DevEvaluationContractError("dev evaluation top-level identity mismatch")
    if config.get("model") != {
        "repo": FORMAL_REPO_ID,
        "revision": FORMAL_REVISION,
        "local_files_only": True,
        "dtype": "bfloat16",
    }:
        raise DevEvaluationContractError("dev evaluation model identity mismatch")
    if config.get("prompt") != {
        "version": PROMPT_V2_FORMAL_MATH,
        "sha256": PROMPT_V2_SHA256,
        "renderer": PROMPT_RENDERER_VERSION,
        "max_prompt_length": 928,
        "max_completion_length": 256,
        "max_sequence_length": 1184,
    }:
        raise DevEvaluationContractError("dev evaluation prompt identity mismatch")
    contracts = formal_parser_verifier_metadata()
    if config.get("parser_sha256") != contracts["parser_contract"]["contract_sha256"]:
        raise DevEvaluationContractError("dev evaluation parser identity mismatch")
    if config.get("verifier_sha256") != contracts["verifier_contract"]["contract_sha256"]:
        raise DevEvaluationContractError("dev evaluation verifier identity mismatch")
    if config.get("reward") != {
        "policy": FORMAL_REWARD_VERSION,
        "sha256": FORMAL_REWARD_SHA256,
    }:
        raise DevEvaluationContractError("dev evaluation reward identity mismatch")
    if config.get("sampling") != {
        "candidate_index": 0,
        "do_sample": True,
        "per_problem_seed_namespace": DEV_SEED_NAMESPACE,
        "temperature": 0.8,
        "top_p": 0.95,
    }:
        raise DevEvaluationContractError("dev evaluation sampling identity mismatch")
    if config.get("budget") != {
        "automatic_retries": 0,
        "candidates_per_problem": 1,
        "max_generated_tokens": 32_768,
        "max_vram_gib": 24,
        "max_wall_time_seconds": 1_800,
        "total_completions": 128,
        "unique_problems": 128,
    }:
        raise DevEvaluationContractError("dev evaluation budget mismatch")
    dev = config.get("dev", {})
    public_path = ROOT / dev.get("manifest", "missing")
    trusted_path = Path(dev.get("trusted_manifest", "missing"))
    if (
        dev.get("manifest_sha256") != file_sha256(public_path)
        or dev.get("trusted_manifest_sha256") != file_sha256(trusted_path)
        or dev.get("manifest_sha256") != identity.get("dev_manifest_sha256")
        or dev.get("trusted_manifest_sha256") != identity.get("trusted_manifest_sha256")
    ):
        raise DevEvaluationContractError("dev manifest SHA mismatch")
    rows = _read_jsonl(public_path)
    trusted = {row["problem_id"]: row for row in _read_jsonl(trusted_path)}
    if len(rows) != 128 or len(trusted) != 128:
        raise DevEvaluationContractError("dev problem count mismatch")
    keys = [(row.get("problem_id"), row.get("content_hash")) for row in rows]
    if len(keys) != len(set(keys)) or {row["problem_id"] for row in rows} != set(trusted):
        raise DevEvaluationContractError("dev problem identity mismatch")
    for row in rows:
        gold = trusted[row["problem_id"]]
        if gold.get("content_hash") != row.get("content_hash"):
            raise DevEvaluationContractError("dev trusted/public hash mismatch")
    domains = Counter(row["source"] for row in rows)
    levels = Counter(str(row["difficulty"]) for row in rows if row["source"] == "math")
    if domains != Counter({"gsm8k": 64, "math": 64}) or levels != Counter(
        {"1": 16, "2": 24, "3": 24}
    ):
        raise DevEvaluationContractError("dev domain/level count mismatch")
    return config, identity, rows


def build_dev_plan(
    config: dict[str, Any], rows: list[dict[str, Any]], *, mode: str
) -> list[dict[str, Any]]:
    if mode not in {"base", "warmstart"}:
        raise DevEvaluationContractError("unknown dev evaluation mode")
    plan = []
    for position, row in enumerate(rows, start=1):
        plan.append(
            {
                "position": position,
                "problem_id": row["problem_id"],
                "content_hash": row["content_hash"],
                "dataset": row["source"],
                "math_level": str(row["difficulty"]) if row["source"] == "math" else None,
                "generation_seed": dev_generation_seed(row["problem_id"], row["content_hash"]),
                "candidate_index": 0,
            }
        )
    if len(plan) != 128 or len({row["problem_id"] for row in plan}) != 128:
        raise DevEvaluationContractError("dev plan count/order contract failed")
    return plan


def validate_matched_plans(base: list[dict[str, Any]], warmstart: list[dict[str, Any]]) -> None:
    if base != warmstart:
        raise DevEvaluationContractError("Base/warm-start dev plan identity mismatch")


def require_finite_logits(is_finite: bool) -> None:
    if not is_finite:
        raise DevEvaluationContractError("dev evaluation produced NaN/Inf logits")


def validate_inference_contract(
    *,
    model_training: bool,
    parameter_requires_grad: list[bool],
    inference_mode_used: bool,
    train_calls: int = 0,
    backward_calls: int = 0,
    optimizer_steps: int = 0,
    checkpoint_writes: int = 0,
) -> dict[str, Any]:
    if model_training or any(parameter_requires_grad) or not inference_mode_used:
        raise DevEvaluationContractError("dev model eval/frozen/inference contract failed")
    if (train_calls, backward_calls, optimizer_steps, checkpoint_writes) != (0, 0, 0, 0):
        raise DevEvaluationContractError(
            "dev evaluation attempted a forbidden training side effect"
        )
    return {
        "model_eval": True,
        "parameters_require_grad": 0,
        "inference_mode": True,
        "train_calls": 0,
        "backward_calls": 0,
        "optimizer_steps": 0,
        "checkpoint_writes": 0,
    }


@dataclass
class DevBudgetGuard:
    deadline: float
    clock: Callable[[], float] = time.monotonic
    completions: int = 0
    generated_tokens: int = 0
    seen_problem_ids: set[str] = field(default_factory=set)

    def record(self, row: dict[str, Any], *, peak_vram_gib: float | None = None) -> None:
        if self.clock() > self.deadline:
            raise DevEvaluationContractError("dev evaluation wall-time ceiling exceeded")
        problem_id = row.get("problem_id")
        if not isinstance(problem_id, str) or problem_id in self.seen_problem_ids:
            raise DevEvaluationContractError("duplicate/missing dev problem identity")
        if row.get("candidate_index") != 0:
            raise DevEvaluationContractError("dev candidate index must be zero")
        count = row.get("exact_token_count")
        if not isinstance(count, int) or isinstance(count, bool) or not 0 <= count <= 256:
            raise DevEvaluationContractError("dev completion token count invalid")
        if self.completions + 1 > 128 or self.generated_tokens + count > 32_768:
            raise DevEvaluationContractError("dev evaluation completion/token budget exceeded")
        if peak_vram_gib is not None and peak_vram_gib > 24:
            raise DevEvaluationContractError("dev evaluation peak VRAM stop gate exceeded")
        self.seen_problem_ids.add(problem_id)
        self.completions += 1
        self.generated_tokens += count

    def finalize(self) -> dict[str, int]:
        if self.completions != 128 or len(self.seen_problem_ids) != 128:
            raise DevEvaluationContractError("dev evaluation incomplete completion contract")
        return {
            "unique_problems": 128,
            "completions": self.completions,
            "generated_tokens": self.generated_tokens,
            "train_calls": 0,
            "backward_calls": 0,
            "optimizer_steps": 0,
            "checkpoint_writes": 0,
            "automatic_retries": 0,
        }

    def snapshot(self) -> dict[str, int]:
        return {
            "unique_problems": len(self.seen_problem_ids),
            "completions": self.completions,
            "generated_tokens": self.generated_tokens,
            "train_calls": 0,
            "backward_calls": 0,
            "optimizer_steps": 0,
            "checkpoint_writes": 0,
            "automatic_retries": 0,
        }


def completion_record(
    *,
    plan_row: dict[str, Any],
    prompt_hash: str,
    completion_ids: list[int],
    completion_mask: list[int],
    text: str,
    eos: bool,
    truncated: bool,
    evaluation: Any,
    mode: str,
    checkpoint_identity: dict[str, Any] | None,
) -> dict[str, Any]:
    status = evaluation.canonical_result.status
    reward = evaluation.to_dict()
    return {
        **plan_row,
        "mode": mode,
        "prompt_hash": prompt_hash,
        "completion_text": text,
        "completion_ids": completion_ids,
        "completion_mask": completion_mask,
        "attention_mask": completion_mask,
        "exact_token_count": sum(completion_mask),
        "eos": eos,
        "truncated": truncated,
        "format_valid": isinstance(parse_completion(text), ParsedCompletion),
        "valid_answer": status in {RewardStatus.WRONG_ANSWER, RewardStatus.VERIFIED_PASS},
        "parseable": status in {RewardStatus.WRONG_ANSWER, RewardStatus.VERIFIED_PASS},
        "canonical_correct": status is RewardStatus.VERIFIED_PASS,
        "verifier_status": reward["canonical_status"],
        "checkpoint_identity": checkpoint_identity,
        **reward,
    }


def validate_dev_rows(
    plan: list[dict[str, Any]], rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    if len(rows) != len(plan):
        raise DevEvaluationContractError("dev completion count mismatch")
    validated = []
    for expected, row in zip(plan, rows, strict=True):
        if any(row.get(key) != value for key, value in expected.items()):
            raise DevEvaluationContractError("dev problem/order/seed identity mismatch")
        ids = row.get("completion_ids")
        mask = row.get("completion_mask")
        if (
            not isinstance(ids, list)
            or not isinstance(mask, list)
            or len(ids) != len(mask)
            or any(not isinstance(value, int) or isinstance(value, bool) for value in ids)
            or any(value not in (0, 1) for value in mask)
            or row.get("exact_token_count") != sum(mask)
        ):
            raise DevEvaluationContractError("dev token/mask evidence mismatch")
        for key in (
            "eos",
            "truncated",
            "format_valid",
            "valid_answer",
            "parseable",
            "canonical_correct",
        ):
            if type(row.get(key)) is not bool:
                raise DevEvaluationContractError(f"dev {key} evidence must be boolean")
        scalar = row.get("scalar_reward")
        if (
            isinstance(scalar, bool)
            or not isinstance(scalar, (int, float))
            or not math.isfinite(scalar)
        ):
            raise DevEvaluationContractError("dev reward is non-finite")
        if row.get("verifier_status") == RewardStatus.INFRA_ERROR.value:
            raise DevEvaluationContractError("dev verifier infrastructure error")
        if row["canonical_correct"] != (row["verifier_status"] == RewardStatus.VERIFIED_PASS.value):
            raise DevEvaluationContractError("dev verifier/correctness contradiction")
        if not isinstance(row.get("completion_text"), str) or not isinstance(
            row.get("prompt_hash"), str
        ):
            raise DevEvaluationContractError("dev text/prompt evidence missing")
        json.dumps(row, allow_nan=False)
        validated.append(dict(row))
    return validated


def _rate(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    return {
        "numerator": sum(bool(row[key]) for row in rows),
        "denominator": len(rows),
        "value": sum(bool(row[key]) for row in rows) / len(rows) if rows else None,
        "available": bool(rows),
        "reason": None if rows else "zero_denominator",
    }


def aggregate_dev_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if len(rows) != 128:
        raise DevEvaluationContractError("dev aggregate requires 128 rows")
    lengths = [int(row["exact_token_count"]) for row in rows]
    parseable = sum(row["parseable"] for row in rows)
    correct = sum(row["canonical_correct"] for row in rows)
    slices: dict[str, Any] = {}
    selections = {
        "all": rows,
        "gsm8k": [row for row in rows if row["dataset"] == "gsm8k"],
        "math": [row for row in rows if row["dataset"] == "math"],
        **{
            f"math_level_{level}": [
                row for row in rows if row["dataset"] == "math" and row["math_level"] == level
            ]
            for level in ("1", "2", "3")
        },
    }
    for name, selected in selections.items():
        slices[name] = {
            "problems": len(selected),
            "candidate0_pass_at_1": _rate(selected, "canonical_correct"),
            "format_rate": _rate(selected, "format_valid"),
            "valid_answer_rate": _rate(selected, "valid_answer"),
            "parseable_rate": _rate(selected, "parseable"),
            "eos_rate": _rate(selected, "eos"),
            "truncation_rate": _rate(selected, "truncated"),
        }
    return {
        "completion_count": 128,
        "unique_problem_count": 128,
        "generated_tokens": sum(lengths),
        "candidate0_pass_at_1": _rate(rows, "canonical_correct"),
        "format_rate": _rate(rows, "format_valid"),
        "valid_answer_rate": _rate(rows, "valid_answer"),
        "parseable_rate": _rate(rows, "parseable"),
        "accuracy_given_parseable": {
            "numerator": correct,
            "denominator": parseable,
            "value": correct / parseable if parseable else None,
            "available": bool(parseable),
            "reason": None if parseable else "zero_denominator",
        },
        "eos_rate": _rate(rows, "eos"),
        "truncation_rate": _rate(rows, "truncated"),
        "completion_length": {
            "mean": statistics.fmean(lengths),
            "median": statistics.median(lengths),
            "std": statistics.stdev(lengths),
            "p95": sorted(lengths)[math.ceil(0.95 * len(lengths)) - 1],
        },
        "reward_status_counts": dict(
            sorted(Counter(row["verifier_status"] for row in rows).items())
        ),
        "pass_at_4": {
            "value": None,
            "available": False,
            "reason": "dev_protocol_one_candidate_per_problem",
        },
        "pass_at_10": {
            "value": None,
            "available": False,
            "reason": "dev_protocol_one_candidate_per_problem",
        },
        "slices": slices,
    }


def paired_dev_comparison(
    base_rows: list[dict[str, Any]], warmstart_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    if len(base_rows) != 128 or len(warmstart_rows) != 128:
        raise DevEvaluationContractError("paired dev comparison requires 128 rows per mode")
    keys = (
        "position",
        "problem_id",
        "content_hash",
        "generation_seed",
        "candidate_index",
        "prompt_hash",
    )
    transitions = Counter()
    pairs = []
    for base, warm in zip(base_rows, warmstart_rows, strict=True):
        if any(base.get(key) != warm.get(key) for key in keys):
            raise DevEvaluationContractError("paired dev evidence alignment mismatch")
        before, after = bool(base["canonical_correct"]), bool(warm["canonical_correct"])
        transition = (
            "improved"
            if not before and after
            else "regressed"
            if before and not after
            else "unchanged_correct"
            if before
            else "unchanged_wrong"
        )
        transitions[transition] += 1
        pairs.append(
            {
                "problem_id": base["problem_id"],
                "base_correct": before,
                "warmstart_correct": after,
                "transition": transition,
            }
        )
    return {
        "problem_count": 128,
        "transitions": dict(transitions),
        "base_pass_at_1": sum(row["canonical_correct"] for row in base_rows) / 128,
        "warmstart_pass_at_1": sum(row["canonical_correct"] for row in warmstart_rows) / 128,
        "delta_percentage_points": 100
        * (
            sum(row["canonical_correct"] for row in warmstart_rows)
            - sum(row["canonical_correct"] for row in base_rows)
        )
        / 128,
        "pairs": pairs,
    }


def validate_warmstart_selection(
    config: dict[str, Any], checkpoint: Path | None, *, mode: str
) -> dict[str, Any] | None:
    if mode == "base":
        if checkpoint is not None:
            raise DevEvaluationContractError("Base dev mode forbids an adapter checkpoint")
        return None
    if mode != "warmstart" or checkpoint is None:
        raise DevEvaluationContractError("warm-start dev mode requires the exact checkpoint")
    expected = config["warmstart_checkpoint"]
    if str(checkpoint) != expected["path"]:
        raise DevEvaluationContractError("warm-start checkpoint path mismatch")
    evidence = validate_checkpoint(
        checkpoint,
        expected_config_sha=expected["config_sha256"],
        expected_run_id=expected["run_id"],
    )
    if evidence["manifest"].get("artifact_sha256") != EXPECTED_CHECKPOINT_ARTIFACT_SHA256:
        raise DevEvaluationContractError("warm-start checkpoint artifact SHA mismatch")
    adapter = checkpoint / "adapter/adapter_model.safetensors"
    if file_sha256(adapter) != EXPECTED_ADAPTER_SHA256:
        raise DevEvaluationContractError("warm-start policy adapter SHA mismatch")
    adapter_config = json.loads((checkpoint / "adapter/adapter_config.json").read_text())
    expected_lora = expected["lora"]
    if (
        evidence["identity"].get("adapter_role") != "policy"
        or adapter_config.get("r") != expected_lora["rank"]
        or adapter_config.get("lora_alpha") != expected_lora["alpha"]
        or adapter_config.get("lora_dropout") != expected_lora["dropout"]
        or set(adapter_config.get("target_modules", [])) != set(expected_lora["target_modules"])
    ):
        raise DevEvaluationContractError("warm-start policy LoRA role mismatch")
    return {
        "checkpoint_path": str(checkpoint.resolve(strict=True)),
        "artifact_sha256": EXPECTED_CHECKPOINT_ARTIFACT_SHA256,
        "adapter_sha256": EXPECTED_ADAPTER_SHA256,
        "adapter_role": "policy",
        "source_run_id": EXPECTED_WARMSTART_RUN_ID,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
