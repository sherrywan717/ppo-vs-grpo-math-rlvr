"""CPU-safe contracts for the frozen four-model GRPO-v2 hidden evaluation."""

from __future__ import annotations

import json
import math
import statistics
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from math_rlvr.contracts import formal_parser_verifier_metadata
from math_rlvr.grpo_v2_contract import (
    canonical_json_sha256,
    pass_k_batch_seed,
    validate_contract_tree,
    validate_model_evaluation_ledger,
)
from math_rlvr.prompt import PROMPT_RENDERER_VERSION, PROMPT_V2_FORMAL_MATH, PROMPT_V2_SHA256
from math_rlvr.rewards.formal import FORMAL_REWARD_SHA256, FORMAL_REWARD_VERSION
from math_rlvr.training.model_source import FORMAL_REPO_ID, FORMAL_REVISION
from math_rlvr.training.warmstart_runtime import file_sha256

ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = Path("configs/grpo_v2/hidden_test_evaluation.json")
CONFIG_SHA256 = "ff588378a5a6bf1331d08ad95d7311648373eb6e28cae763447d9d67941b7d22"
RUN_ROOT = Path("/root/autodl-tmp/runs/math_rlvr")
EXPECTED_BRANCH = "improve/grpo-v2"
ROLES = ("base", "old_grpo_v1", "warmstart_only", "selected_grpo_v2")
MODEL_COMPLETIONS = 1_300
FOUR_MODEL_COMPLETIONS = 5_200


class HiddenEvaluationContractError(RuntimeError):
    """A frozen hidden-test identity, ledger, or role invariant failed."""


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _canonical_file(path: Path) -> Path:
    if path.is_symlink() or not path.is_file() or path.resolve(strict=True) != path:
        raise HiddenEvaluationContractError(f"non-canonical file: {path}")
    return path


def _validate_checkpoint_files(checkpoint: Path, expected: dict[str, Any]) -> dict[str, Any]:
    if (
        checkpoint.is_symlink()
        or checkpoint.parent.is_symlink()
        or checkpoint.resolve(strict=True) != checkpoint
        or str(checkpoint) != expected["checkpoint"]
        or checkpoint.name != f"checkpoint-{expected['checkpoint_step']}"
        or checkpoint.parent.name != expected["run_id"]
    ):
        raise HiddenEvaluationContractError("hidden evaluation checkpoint path/step mismatch")
    manifest_path = _canonical_file(checkpoint / "artifact_manifest.json")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("base_weights_included") is not False:
        raise HiddenEvaluationContractError("hidden evaluation checkpoint contains base weights")
    actual_manifest_sha = file_sha256(manifest_path)
    if expected.get("artifact_manifest_file_sha256"):
        if actual_manifest_sha != expected["artifact_manifest_file_sha256"]:
            raise HiddenEvaluationContractError("checkpoint manifest file SHA mismatch")
        if manifest.get("artifact_sha256") != expected["artifact_sha256"]:
            raise HiddenEvaluationContractError("checkpoint artifact SHA mismatch")
    elif actual_manifest_sha != expected["artifact_sha256"]:
        raise HiddenEvaluationContractError("checkpoint artifact SHA mismatch")
    adapter_dir = checkpoint / expected["adapter_subdir"]
    adapter = _canonical_file(adapter_dir / "adapter_model.safetensors")
    adapter_config = _canonical_file(adapter_dir / "adapter_config.json")
    if file_sha256(adapter) != expected["adapter_sha256"]:
        raise HiddenEvaluationContractError("checkpoint policy adapter SHA mismatch")
    config = json.loads(adapter_config.read_text())
    if (
        config.get("r") != 16
        or config.get("lora_alpha") != 32
        or float(config.get("lora_dropout", -1)) != 0
        or set(config.get("target_modules", [])) != {"q_proj", "k_proj", "v_proj", "o_proj"}
    ):
        raise HiddenEvaluationContractError("checkpoint policy LoRA identity mismatch")
    forbidden = {"model.safetensors", "pytorch_model.bin"}
    if any(path.name in forbidden for path in checkpoint.iterdir()):
        raise HiddenEvaluationContractError("full base-model file present in checkpoint")
    return {
        "role": "policy",
        "source_run_id": expected["run_id"],
        "checkpoint_step": expected["checkpoint_step"],
        "checkpoint_path": str(checkpoint),
        "artifact_sha256": expected["artifact_sha256"],
        "adapter_sha256": expected["adapter_sha256"],
        "adapter_path": str(adapter_dir),
        "base_weights_included": False,
        "optimizer_loaded_for_evaluation": False,
        "scheduler_loaded_for_evaluation": False,
        "rng_loaded_for_evaluation": False,
    }


def validate_role_selection(
    config: dict[str, Any], *, role: str, checkpoint: Path | None
) -> dict[str, Any] | None:
    if role not in ROLES:
        raise HiddenEvaluationContractError("unknown hidden-test model role")
    expected = config["roles"][role]
    if role == "base":
        if checkpoint is not None or any(
            expected.get(key) is not None
            for key in ("checkpoint", "artifact_sha256", "adapter_sha256")
        ):
            raise HiddenEvaluationContractError("Base hidden evaluation forbids adapter")
        return None
    if checkpoint is None:
        raise HiddenEvaluationContractError("adapter role requires exact checkpoint")
    return _validate_checkpoint_files(checkpoint, expected)


def load_hidden_contract(
    config_path: Path = CONFIG_PATH,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], set[str]]:
    if config_path != CONFIG_PATH or config_path.is_absolute() or ".." in config_path.parts:
        raise HiddenEvaluationContractError("hidden evaluation requires exact config path")
    path = _canonical_file(ROOT / config_path)
    if file_sha256(path) != CONFIG_SHA256:
        raise HiddenEvaluationContractError("hidden evaluation config SHA mismatch")
    config = json.loads(path.read_text())
    if config.get("schema_version") != 1 or config.get("seed") != 42:
        raise HiddenEvaluationContractError("hidden evaluation top-level identity mismatch")
    if config.get("model") != {
        "repo": FORMAL_REPO_ID,
        "revision": FORMAL_REVISION,
        "local_files_only": True,
        "dtype": "bfloat16",
    }:
        raise HiddenEvaluationContractError("hidden evaluation model identity mismatch")
    if config.get("prompt") != {
        "version": PROMPT_V2_FORMAL_MATH,
        "sha256": PROMPT_V2_SHA256,
        "renderer": PROMPT_RENDERER_VERSION,
        "max_prompt_length": 928,
        "max_completion_length": 256,
        "max_sequence_length": 1184,
    }:
        raise HiddenEvaluationContractError("hidden evaluation prompt identity mismatch")
    contracts = formal_parser_verifier_metadata()
    if (
        config.get("parser_sha256") != contracts["parser_contract"]["contract_sha256"]
        or config.get("verifier_sha256") != contracts["verifier_contract"]["contract_sha256"]
        or config.get("reward") != {"policy": FORMAL_REWARD_VERSION, "sha256": FORMAL_REWARD_SHA256}
    ):
        raise HiddenEvaluationContractError("hidden parser/verifier/reward identity mismatch")
    data = config["data"]
    registry_path = _canonical_file(ROOT / data["registry_path"])
    if file_sha256(registry_path) != data["registry_raw_sha256"]:
        raise HiddenEvaluationContractError("hidden data registry raw SHA mismatch")
    registry = json.loads(registry_path.read_text())
    claim = registry.pop("registry_sha256")
    if claim != canonical_json_sha256(registry) or claim != data["registry_canonical_sha256"]:
        raise HiddenEvaluationContractError("hidden data registry canonical SHA mismatch")
    public_path = _canonical_file(ROOT / data["public_manifest"])
    subset_path = _canonical_file(ROOT / data["shared_subset"])
    if (
        file_sha256(public_path) != data["public_manifest_sha256"]
        or file_sha256(subset_path) != data["shared_subset_sha256"]
    ):
        raise HiddenEvaluationContractError("hidden public manifest SHA mismatch")
    # Dry-run never opens trusted hidden gold; it binds only the preregistered path/SHA claim.
    trusted_claim = registry["trusted_runtime"]["files"]["test_v2_hidden_trusted.jsonl"]
    if (
        data["trusted_manifest"]
        != str(Path(registry["trusted_runtime"]["path"]) / "test_v2_hidden_trusted.jsonl")
        or data["trusted_manifest_sha256"] != trusted_claim["sha256"]
    ):
        raise HiddenEvaluationContractError("hidden trusted-manifest identity mismatch")
    for contract_field in ("evaluation_contract", "contract"):
        contract_path = _canonical_file(ROOT / config["pass_k"][contract_field])
        if file_sha256(contract_path) != config["pass_k"][f"{contract_field}_sha256"]:
            raise HiddenEvaluationContractError("hidden pass@k contract SHA mismatch")
    if validate_contract_tree(ROOT)["four_models_total"] != FOUR_MODEL_COMPLETIONS:
        raise HiddenEvaluationContractError("hidden frozen contract tree mismatch")
    rows = _read_jsonl(public_path)
    nested = json.loads(subset_path.read_text())["problems"]
    ids = [row["problem_id"] for row in rows]
    nested_ids = {row["problem_id"] for row in nested}
    if len(rows) != 400 or len(ids) != len(set(ids)) or len(nested_ids) != 100:
        raise HiddenEvaluationContractError("hidden problem counts/identity mismatch")
    if not nested_ids < set(ids):
        raise HiddenEvaluationContractError("hidden shared subset is not strict subset")
    domains = Counter(row["source"] for row in rows)
    levels = Counter(str(row["difficulty"]) for row in rows if row["source"] == "math")
    if domains != Counter({"gsm8k": 200, "math": 200}) or levels != Counter(
        {"1": 3, "2": 33, "3": 43, "4": 59, "5": 62}
    ):
        raise HiddenEvaluationContractError("hidden domain/level allocation mismatch")
    return (
        config,
        {"config_sha256": CONFIG_SHA256, "public_manifest_sha256": file_sha256(public_path)},
        rows,
        nested_ids,
    )


def build_hidden_plan(
    rows: list[dict[str, Any]], shared_problem_ids: set[str]
) -> list[dict[str, Any]]:
    plan: list[dict[str, Any]] = []
    for position, row in enumerate(rows, start=1):
        batch_seed = pass_k_batch_seed(
            problem_id=row["problem_id"], content_hash=row["content_hash"], seed=42
        )
        indices = range(10) if row["problem_id"] in shared_problem_ids else (0,)
        for candidate_index in indices:
            plan.append(
                {
                    "position": position,
                    "problem_id": row["problem_id"],
                    "content_hash": row["content_hash"],
                    "dataset": row["source"],
                    "math_level": (str(row["difficulty"]) if row["source"] == "math" else None),
                    "shared_n10": row["problem_id"] in shared_problem_ids,
                    "batch_seed": batch_seed,
                    "sampling_seed": batch_seed,
                    "candidate_index": candidate_index,
                }
            )
    if len(plan) != MODEL_COMPLETIONS:
        raise HiddenEvaluationContractError("hidden plan must contain exactly 1,300 keys")
    return plan


def validate_hidden_rows(
    plan: list[dict[str, Any]], rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    if len(rows) != 1_300 or len(plan) != 1_300:
        raise HiddenEvaluationContractError("hidden completion count mismatch")
    validated = []
    for expected, row in zip(plan, rows, strict=True):
        if any(row.get(key) != value for key, value in expected.items()):
            raise HiddenEvaluationContractError("hidden problem/candidate/seed identity mismatch")
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
            raise HiddenEvaluationContractError("hidden token/mask evidence mismatch")
        for key in (
            "eos",
            "truncated",
            "format_valid",
            "valid_answer",
            "parseable",
            "canonical_correct",
        ):
            if type(row.get(key)) is not bool:
                raise HiddenEvaluationContractError(f"hidden {key} evidence must be boolean")
        scalar = row.get("scalar_reward")
        if (
            isinstance(scalar, bool)
            or not isinstance(scalar, (int, float))
            or not math.isfinite(scalar)
        ):
            raise HiddenEvaluationContractError("hidden reward is non-finite")
        status = row.get("verifier_status")
        if status == "INFRA_ERROR":
            raise HiddenEvaluationContractError("hidden verifier infrastructure error")
        if row["canonical_correct"] != (status == "VERIFIED_PASS"):
            raise HiddenEvaluationContractError("hidden verifier/correctness contradiction")
        if not isinstance(row.get("completion_text"), str) or not isinstance(
            row.get("prompt_hash"), str
        ):
            raise HiddenEvaluationContractError("hidden text/prompt evidence missing")
        json.dumps(row, allow_nan=False)
        validated.append(dict(row))
    return validated


def validate_four_model_plans(plans: dict[str, list[dict[str, Any]]]) -> None:
    if set(plans) != set(ROLES):
        raise HiddenEvaluationContractError("four hidden model roles are required")
    reference = plans["base"]
    keys = [
        (row["problem_id"], row["content_hash"], row["candidate_index"], row["batch_seed"])
        for row in reference
    ]
    if len(keys) != MODEL_COMPLETIONS or len(keys) != len(set(keys)):
        raise HiddenEvaluationContractError("hidden reference candidate keys invalid")
    for role in ROLES[1:]:
        candidate_keys = [
            (row["problem_id"], row["content_hash"], row["candidate_index"], row["batch_seed"])
            for row in plans[role]
        ]
        if candidate_keys != keys:
            raise HiddenEvaluationContractError("four-model hidden candidate key drift")


def _hidden_rate(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    numerator = sum(bool(row[key]) for row in rows)
    return {
        "numerator": numerator,
        "denominator": len(rows),
        "value": numerator / len(rows) if rows else None,
        "available": bool(rows),
        "reason": None if rows else "zero_denominator",
    }


def aggregate_hidden_candidate0(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate the frozen 400-problem hidden candidate-0 universe."""
    if len(rows) != 400:
        raise HiddenEvaluationContractError("hidden candidate-0 aggregate requires 400 rows")
    if any(row.get("candidate_index") != 0 for row in rows):
        raise HiddenEvaluationContractError("hidden candidate-0 aggregate received nonzero index")
    problem_ids = [row.get("problem_id") for row in rows]
    if len(set(problem_ids)) != 400:
        raise HiddenEvaluationContractError("hidden candidate-0 problem IDs are not unique")
    domains = Counter(row.get("dataset") for row in rows)
    levels = Counter(str(row.get("math_level")) for row in rows if row.get("dataset") == "math")
    if domains != Counter({"gsm8k": 200, "math": 200}) or levels != Counter(
        {"1": 3, "2": 33, "3": 43, "4": 59, "5": 62}
    ):
        raise HiddenEvaluationContractError("hidden candidate-0 domain/level drift")
    lengths = [int(row["exact_token_count"]) for row in rows]
    parseable = sum(bool(row["parseable"]) for row in rows)
    correct = sum(bool(row["canonical_correct"]) for row in rows)
    selections = {
        "all": rows,
        "gsm8k": [row for row in rows if row["dataset"] == "gsm8k"],
        "math": [row for row in rows if row["dataset"] == "math"],
        **{
            f"math_level_{level}": [
                row
                for row in rows
                if row["dataset"] == "math" and str(row["math_level"]) == str(level)
            ]
            for level in range(1, 6)
        },
    }
    slices = {}
    for name, selected in selections.items():
        slices[name] = {
            "problems": len(selected),
            "candidate0_pass_at_1": _hidden_rate(selected, "canonical_correct"),
            "format_rate": _hidden_rate(selected, "format_valid"),
            "valid_answer_rate": _hidden_rate(selected, "valid_answer"),
            "parseable_rate": _hidden_rate(selected, "parseable"),
            "eos_rate": _hidden_rate(selected, "eos"),
            "truncation_rate": _hidden_rate(selected, "truncated"),
        }
    return {
        "completion_count": 400,
        "unique_problem_count": 400,
        "generated_tokens": sum(lengths),
        "candidate0_pass_at_1": _hidden_rate(rows, "canonical_correct"),
        "format_rate": _hidden_rate(rows, "format_valid"),
        "valid_answer_rate": _hidden_rate(rows, "valid_answer"),
        "parseable_rate": _hidden_rate(rows, "parseable"),
        "accuracy_given_parseable": {
            "numerator": correct,
            "denominator": parseable,
            "value": correct / parseable if parseable else None,
            "available": bool(parseable),
            "reason": None if parseable else "zero_denominator",
        },
        "eos_rate": _hidden_rate(rows, "eos"),
        "truncation_rate": _hidden_rate(rows, "truncated"),
        "completion_length": {
            "mean": statistics.fmean(lengths),
            "median": statistics.median(lengths),
            "std": statistics.stdev(lengths),
            "p95": sorted(lengths)[math.ceil(0.95 * len(lengths)) - 1],
        },
        "reward_status_counts": dict(
            sorted(Counter(row["verifier_status"] for row in rows).items())
        ),
        "slices": slices,
    }


@dataclass
class HiddenBudgetGuard:
    deadline: float
    clock: Any = time.monotonic
    completions: int = 0
    generated_tokens: int = 0
    keys: set[tuple[str, int]] = field(default_factory=set)

    def record(self, row: dict[str, Any], *, peak_vram_gib: float | None = None) -> None:
        if self.clock() > self.deadline:
            raise HiddenEvaluationContractError("hidden evaluation wall-time ceiling exceeded")
        key = (row.get("problem_id"), row.get("candidate_index"))
        count = row.get("exact_token_count")
        if (
            not isinstance(key[0], str)
            or key in self.keys
            or not isinstance(key[1], int)
            or not isinstance(count, int)
            or isinstance(count, bool)
            or not 0 <= count <= 256
        ):
            raise HiddenEvaluationContractError("hidden completion evidence key/count invalid")
        if self.completions >= MODEL_COMPLETIONS or self.generated_tokens + count > 332_800:
            raise HiddenEvaluationContractError("hidden completion/token budget exceeded")
        if peak_vram_gib is not None and peak_vram_gib > 24:
            raise HiddenEvaluationContractError("hidden peak VRAM stop gate exceeded")
        self.keys.add(key)
        self.completions += 1
        self.generated_tokens += count

    def finalize(self, all_problem_ids: set[str], shared_problem_ids: set[str], rows: list[dict]):
        ledger = validate_model_evaluation_ledger(
            rows, all_problem_ids=all_problem_ids, shared_problem_ids=shared_problem_ids
        )
        if self.completions != MODEL_COMPLETIONS or len(self.keys) != MODEL_COMPLETIONS:
            raise HiddenEvaluationContractError("hidden completion prefix incomplete")
        return {**ledger, "generated_tokens": self.generated_tokens, "automatic_retries": 0}


def artifact_schema() -> dict[str, list[str]]:
    return {
        "per_model": [
            "completions.jsonl",
            "per_problem.csv",
            "candidate0_metrics.json",
            "candidate0_metrics.csv",
            "pass_k_per_problem.csv",
            "pass_k_summary.json",
            "pass_k_summary.csv",
            "status_distribution.csv",
            "truncation_analysis.csv",
            "resource_metrics.csv",
            "resource_summary.json",
            "summary.json",
            "report.md",
            "figures/",
            "checksums.sha256",
        ],
        "aggregate": [
            "four_model_summary.csv",
            "paired_candidate0_comparisons.csv",
            "paired_pass_k_comparisons.csv",
            "per_level_results.csv",
            "cost_quality_tradeoff.csv",
            "error_analysis.md",
            "case_studies.md",
            "final_comparison.md",
            "figures/",
        ],
    }
