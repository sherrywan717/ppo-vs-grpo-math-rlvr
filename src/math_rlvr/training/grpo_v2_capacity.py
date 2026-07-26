"""Pinned-tokenizer prompt-capacity preflight for GRPO-v2 training and dev."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from math_rlvr.dataset import MathProblem
from math_rlvr.prompt import (
    PROMPT_V2_FORMAL_MATH,
    ExperimentScope,
    render_prompt_version,
    render_training_prompt,
)
from math_rlvr.training.model_source import FORMAL_REVISION

ROOT = Path(__file__).resolve().parents[3]
OLD_PROMPT_CAP = 832
MINIMUM_AMENDED_PROMPT_CAP = 928
PROMPT_CAP_ALIGNMENT = 32


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _problem(public: dict[str, Any], trusted: dict[str, Any]) -> MathProblem:
    if public["content_hash"] != trusted["content_hash"]:
        raise ValueError("trusted/public prompt content hash mismatch")
    return MathProblem(
        **{
            key: public[key]
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
        gold_answer=trusted["gold_answer"],
    )


def deterministic_prompt_cap(max_prompt_tokens: int) -> int:
    if not isinstance(max_prompt_tokens, int) or max_prompt_tokens <= 0:
        raise ValueError("max prompt token count must be a positive integer")
    aligned = math.ceil(max_prompt_tokens / PROMPT_CAP_ALIGNMENT) * PROMPT_CAP_ALIGNMENT
    return max(MINIMUM_AMENDED_PROMPT_CAP, aligned)


def _percentile(values: list[int], fraction: float) -> int:
    ordered = sorted(values)
    return ordered[math.ceil(fraction * len(ordered)) - 1]


def length_statistics(values: list[int]) -> dict[str, int | float]:
    if not values:
        raise ValueError("prompt length statistics require rows")
    return {
        "count": len(values),
        "min": min(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p90": _percentile(values, 0.90),
        "p95": _percentile(values, 0.95),
        "p99": _percentile(values, 0.99),
        "max": max(values),
    }


def validate_capacity_rows(
    rows: Sequence[dict[str, Any]],
    *,
    expected_training: Sequence[tuple[str, str]],
    expected_dev: Sequence[tuple[str, str]],
    prompt_cap: int,
    completion_cap: int,
    sequence_ceiling: int,
) -> None:
    training = [row for row in rows if row.get("phase") == "train"]
    dev = [row for row in rows if row.get("phase") == "dev"]
    if len(training) != 512 or len(dev) != 128:
        raise ValueError("capacity preflight requires exactly 512 train and 128 dev prompts")
    if [(row["problem_id"], row["content_hash"]) for row in training] != list(expected_training):
        raise ValueError("capacity preflight training problem order/hash drift")
    if [(row["problem_id"], row["content_hash"]) for row in dev] != list(expected_dev):
        raise ValueError("capacity preflight dev problem order/hash drift")
    for index, row in enumerate(training):
        if (
            row.get("position") != index + 1
            or row.get("update") != index // 4 + 1
            or row.get("slot") != index % 4
        ):
            raise ValueError("capacity preflight curriculum position drift")
    for index, row in enumerate(dev):
        if row.get("position") != index + 1:
            raise ValueError("capacity preflight dev position drift")
    for row in rows:
        prompt_tokens = row.get("prompt_tokens")
        if not isinstance(prompt_tokens, int) or prompt_tokens <= 0:
            raise ValueError("capacity preflight prompt token count is invalid")
        if row.get("truncation") is not False:
            raise ValueError("capacity preflight detected tokenizer truncation")
        if prompt_tokens > prompt_cap:
            raise ValueError("prompt exceeds frozen GRPO-v2 prompt cap")
        if row.get("combined_potential") != prompt_tokens + completion_cap:
            raise ValueError("capacity preflight combined-length accounting mismatch")
        if prompt_tokens + completion_cap > sequence_ceiling:
            raise ValueError("prompt plus completion exceeds frozen sequence ceiling")
        prompt_hash = row.get("prompt_hash")
        if not isinstance(prompt_hash, str) or len(prompt_hash) != 64:
            raise ValueError("capacity preflight prompt hash missing")


def audit_prompt_capacity(
    tokenizer,
    *,
    design: dict[str, Any],
    identity: dict[str, Any],
    contract,
    model_source,
) -> dict[str, Any]:
    """Render and tokenize the exact 512 training and 128 dev prompts without truncation."""
    from math_rlvr.evaluation.grpo_v2_dev_runtime import load_dev_contract
    from math_rlvr.training.grpo_v2_runtime import normalized_training_config

    if model_source.revision != FORMAL_REVISION or not model_source.local_files_only:
        raise ValueError("capacity preflight model source identity mismatch")
    prompt_cap = design["prompt"]["max_prompt_length"]
    completion_cap = design["prompt"]["max_completion_length"]
    sequence_ceiling = design["prompt"]["max_sequence_length"]
    normalized = normalized_training_config(design, contract)

    public_train = {
        row["problem_id"]: row for row in _read_jsonl(ROOT / design["data"]["manifest"])
    }
    trusted_train = {
        row["problem_id"]: row for row in _read_jsonl(Path(identity["trusted_train_manifest_path"]))
    }
    curriculum = json.loads((ROOT / design["data"]["curriculum"]).read_text())["positions"]
    rows: list[dict[str, Any]] = []
    for curriculum_row in curriculum:
        problem = _problem(
            public_train[curriculum_row["problem_id"]],
            trusted_train[curriculum_row["problem_id"]],
        )
        rendered = render_training_prompt(
            tokenizer, problem, normalized, scope=ExperimentScope.MAIN_FORMAL
        )
        prompt_ids = tokenizer(rendered, add_special_tokens=False, truncation=False)["input_ids"]
        rows.append(
            {
                "phase": "train",
                "position": curriculum_row["position"],
                "update": curriculum_row["update"],
                "slot": curriculum_row["slot"],
                "problem_id": problem.problem_id,
                "dataset": problem.source,
                "math_level": str(problem.difficulty) if problem.source == "math" else None,
                "content_hash": problem.content_hash,
                "prompt_hash": hashlib.sha256(rendered.encode()).hexdigest(),
                "prompt_tokens": len(prompt_ids),
                "truncation": False,
            }
        )

    dev_config, _, public_dev = load_dev_contract()
    trusted_dev = {
        row["problem_id"]: row for row in _read_jsonl(Path(dev_config["dev"]["trusted_manifest"]))
    }
    for position, source in enumerate(public_dev, start=1):
        problem = _problem(source, trusted_dev[source["problem_id"]])
        rendered = render_prompt_version(tokenizer, problem, PROMPT_V2_FORMAL_MATH)
        prompt_ids = tokenizer(rendered, add_special_tokens=False, truncation=False)["input_ids"]
        rows.append(
            {
                "phase": "dev",
                "position": position,
                "update": None,
                "slot": None,
                "problem_id": problem.problem_id,
                "dataset": problem.source,
                "math_level": str(problem.difficulty) if problem.source == "math" else None,
                "content_hash": problem.content_hash,
                "prompt_hash": hashlib.sha256(rendered.encode()).hexdigest(),
                "prompt_tokens": len(prompt_ids),
                "truncation": False,
            }
        )

    for row in rows:
        row.update(
            {
                "old_prompt_cap": OLD_PROMPT_CAP,
                "new_prompt_cap": prompt_cap,
                "old_cap_overflow": row["prompt_tokens"] > OLD_PROMPT_CAP,
                "cap_928_overflow": row["prompt_tokens"] > MINIMUM_AMENDED_PROMPT_CAP,
                "new_cap_overflow": row["prompt_tokens"] > prompt_cap,
                "max_completion_length": completion_cap,
                "combined_potential": row["prompt_tokens"] + completion_cap,
                "sequence_ceiling": sequence_ceiling,
                "tokenizer_revision": model_source.revision,
                "prompt_identity": contract.prompt_sha256,
            }
        )

    expected_training = tuple((row["problem_id"], row["content_hash"]) for row in curriculum)
    expected_dev = tuple((row["problem_id"], row["content_hash"]) for row in public_dev)
    validate_capacity_rows(
        rows,
        expected_training=expected_training,
        expected_dev=expected_dev,
        prompt_cap=prompt_cap,
        completion_cap=completion_cap,
        sequence_ceiling=sequence_ceiling,
    )
    model_config = json.loads((model_source.snapshot_path / "config.json").read_text())
    context_window = int(model_config["max_position_embeddings"])
    if sequence_ceiling > context_window:
        raise ValueError("GRPO-v2 sequence ceiling exceeds the model context window")
    lengths = [row["prompt_tokens"] for row in rows]
    observed_cap = deterministic_prompt_cap(max(lengths))
    if prompt_cap != observed_cap or sequence_ceiling != prompt_cap + completion_cap:
        raise ValueError("GRPO-v2 capacity fields do not match the deterministic amendment")
    summary = {
        "schema_version": 1,
        "status": "passed",
        "training": length_statistics(
            [row["prompt_tokens"] for row in rows if row["phase"] == "train"]
        ),
        "dev": length_statistics([row["prompt_tokens"] for row in rows if row["phase"] == "dev"]),
        "overall": length_statistics(lengths),
        "old_prompt_cap": OLD_PROMPT_CAP,
        "minimum_amended_prompt_cap": MINIMUM_AMENDED_PROMPT_CAP,
        "new_prompt_cap": prompt_cap,
        "max_completion_length": completion_cap,
        "sequence_ceiling": sequence_ceiling,
        "old_cap_overflows": sum(row["old_cap_overflow"] for row in rows),
        "cap_928_overflows": sum(row["cap_928_overflow"] for row in rows),
        "new_cap_overflows": sum(row["new_cap_overflow"] for row in rows),
        "truncation_count": sum(row["truncation"] is not False for row in rows),
        "max_combined_potential": max(row["combined_potential"] for row in rows),
        "model_context_window": context_window,
        "model_context_margin": context_window - sequence_ceiling,
        "model_revision": model_source.revision,
        "prompt_identity": contract.prompt_sha256,
        "hidden_test_accesses": 0,
        "model_weight_loads": 0,
        "generation_calls": 0,
        "trainer_constructions": 0,
        "optimizer_initializations": 0,
    }
    summary["audit_identity_sha256"] = hashlib.sha256(
        json.dumps(
            {"summary": summary, "rows": rows},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode()
    ).hexdigest()
    return {"summary": summary, "rows": rows}


def run_pre_model_capacity_preflight(
    *,
    design: dict[str, Any],
    identity: dict[str, Any],
    contract,
    model_source,
    tokenizer=None,
) -> dict[str, Any]:
    if tokenizer is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            model_source.snapshot_path,
            local_files_only=True,
            trust_remote_code=False,
        )
    result = audit_prompt_capacity(
        tokenizer,
        design=design,
        identity=identity,
        contract=contract,
        model_source=model_source,
    )
    expected = identity.get("capacity_audit_identity_sha256")
    if expected is not None and result["summary"]["audit_identity_sha256"] != expected:
        raise ValueError("GRPO-v2 frozen capacity audit identity mismatch")
    try:
        import torch

        if torch.cuda.is_initialized():
            raise ValueError("capacity preflight initialized CUDA")
    except ImportError:
        pass
    return result
