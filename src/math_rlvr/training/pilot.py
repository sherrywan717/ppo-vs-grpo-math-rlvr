"""CPU-safe matched 0.5B pilot contracts and exact config authorization."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from math_rlvr.contracts import parser_verifier_metadata
from math_rlvr.dataset import MathProblem, load_manifest
from math_rlvr.prompt import (
    PROMPT_RENDERER_VERSION,
    PROMPT_V1_SHA256,
    PROMPT_V1_STRICT_CONCISE,
    format_problem_version,
)
from math_rlvr.rewards.staged import (
    STAGED_REWARD_SHA256,
    STAGED_REWARD_VERSION,
)

PILOT_FAMILY = "matched_0p5b_v1"
PILOT_DISCLAIMER = "Matched 0.5B pilot - not the final benchmark"
PILOT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
PILOT_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
PILOT_SEEDS = (42, 123, 2026)
PILOT_PROBLEM_IDS = tuple(f"countdown:train:{index}" for index in range(4))
PILOT_RESPONSES_PER_PROMPT = 4
PILOT_COMPLETIONS = 16
PILOT_TOKEN_CAP = 2048
PILOT_CONFIG_ROOT = Path("configs/pilot")
PILOT_MANIFEST_PATH = PILOT_CONFIG_ROOT / "matched_0p5b_manifest.json"
PILOT_REGISTRY_PATH = PILOT_CONFIG_ROOT / "resolved_config_sha256.json"
SOURCE_MANIFEST_PATH = Path("/root/autodl-tmp/datasets/math_rlvr/manifests/countdown_train.json")
SOURCE_MANIFEST_SHA256 = "f7b3138c4fd29063ee05b568462c9cc5c2f8697ee63b8b208949b1b3998ce196"
POLICY_LORA = {
    "rank": 16,
    "alpha": 32,
    "dropout": 0,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
}
RUN_ORDER = (
    (42, "ppo"),
    (42, "grpo"),
    (123, "grpo"),
    (123, "ppo"),
    (2026, "ppo"),
    (2026, "grpo"),
)


def _canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def canonical_json_sha256(payload: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def rendered_prompt_payload_sha256(problem: MathProblem, version: str) -> str:
    """Hash the frozen tokenizer-free chat payload without changing prompt.py."""
    return canonical_json_sha256(
        {
            "renderer_version": PROMPT_RENDERER_VERSION,
            "messages": format_problem_version(problem, version),
            "add_generation_prompt": True,
        }
    )


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def problem_contract_sha256(problem: MathProblem) -> str:
    """Hash the selected source record without copying gold/construction into pilot files."""
    metadata = {
        key: problem.metadata[key]
        for key in (
            "dataset_id",
            "revision",
            "source_split",
            "source_index",
            "numbers",
            "target",
        )
    }
    return canonical_json_sha256(
        {
            "problem_id": problem.problem_id,
            "source": problem.source,
            "prompt": problem.prompt,
            "category": problem.category,
            "difficulty": problem.difficulty,
            "split": problem.split,
            "source_index": problem.source_index,
            "content_hash": problem.content_hash,
            "metadata": metadata,
        }
    )


def pilot_manifest_sha256(payload: dict[str, Any]) -> str:
    body = dict(payload)
    body.pop("manifest_sha256", None)
    return canonical_json_sha256(body)


def _source_problem_map() -> dict[str, MathProblem]:
    if file_sha256(SOURCE_MANIFEST_PATH) != SOURCE_MANIFEST_SHA256:
        raise ValueError("frozen Countdown source manifest SHA256 mismatch")
    problems = load_manifest(SOURCE_MANIFEST_PATH)
    selected = problems[:4]
    if tuple(problem.problem_id for problem in selected) != PILOT_PROBLEM_IDS:
        raise ValueError("pilot problems are not the first four source records")
    return {problem.problem_id: problem for problem in selected}


def validate_pilot_manifest(path: Path = PILOT_MANIFEST_PATH) -> dict[str, Any]:
    if path.resolve() != PILOT_MANIFEST_PATH.resolve():
        raise ValueError("unexpected pilot manifest path")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("manifest_sha256") != pilot_manifest_sha256(payload):
        raise ValueError("pilot manifest SHA256 mismatch")
    expected_identity = {
        "model": {
            "name_or_path": PILOT_MODEL,
            "revision": PILOT_REVISION,
            "local_files_only": True,
            "dtype": "bfloat16",
        },
        "prompt": {
            "version": PROMPT_V1_STRICT_CONCISE,
            "sha256": PROMPT_V1_SHA256,
            "renderer": PROMPT_RENDERER_VERSION,
        },
        "reward": {
            "version": STAGED_REWARD_VERSION,
            "sha256": STAGED_REWARD_SHA256,
        },
        **parser_verifier_metadata(),
    }
    if payload.get("identity") != expected_identity:
        raise ValueError("pilot manifest identity mismatch")
    if payload.get("source_manifest") != {
        "path": str(SOURCE_MANIFEST_PATH),
        "sha256": SOURCE_MANIFEST_SHA256,
        "selection_rule": "first_four_records_in_original_order",
    }:
        raise ValueError("pilot source manifest identity mismatch")
    records = payload.get("problems")
    if not isinstance(records, list) or len(records) != 4:
        raise ValueError("pilot manifest must contain four problem records")
    if tuple(row.get("problem_id") for row in records) != PILOT_PROBLEM_IDS:
        raise ValueError("pilot problem order mismatch")
    source = _source_problem_map()
    for row in records:
        if "gold_answer" in row or "construction" in row:
            raise ValueError("gold or construction leaked into pilot manifest")
        problem = source[row["problem_id"]]
        expected = {
            "ordinal": problem.source_index,
            "problem_id": problem.problem_id,
            "problem_sha256": problem_contract_sha256(problem),
            "source_prompt_content_sha256": problem.content_hash,
            "rendered_prompt_sha256": rendered_prompt_payload_sha256(
                problem, PROMPT_V1_STRICT_CONCISE
            ),
            "difficulty": {
                "category": problem.category,
                "label": problem.difficulty,
                "source_index": problem.source_index,
            },
        }
        if row != expected:
            raise ValueError(f"pilot problem contract mismatch: {problem.problem_id}")
    return payload


def pilot_pair_keys() -> list[str]:
    return [
        f"{problem_id}::generation:{generation_index}"
        for problem_id in PILOT_PROBLEM_IDS
        for generation_index in range(PILOT_RESPONSES_PER_PROMPT)
    ]


def pilot_episode_records(algorithm: str, seed: int) -> list[dict[str, Any]]:
    if algorithm not in {"ppo", "grpo"}:
        raise ValueError("pilot algorithm must be ppo or grpo")
    if seed not in PILOT_SEEDS:
        raise ValueError("pilot episode seed is not approved")
    manifest = validate_pilot_manifest()
    problem_rows = {row["problem_id"]: row for row in manifest["problems"]}
    return [
        {
            "episode_position": index,
            "problem_id": pair_key.split("::", 1)[0],
            "generation_index": int(pair_key.rsplit(":", 1)[1]),
            "pair_key": pair_key,
            "problem_hash": problem_rows[pair_key.split("::", 1)[0]]["problem_sha256"],
            "rendered_prompt_hash": problem_rows[pair_key.split("::", 1)[0]][
                "rendered_prompt_sha256"
            ],
            "seed": seed,
            "algorithm": algorithm,
        }
        for index, pair_key in enumerate(pilot_pair_keys())
    ]


def _common_contract(config: dict[str, Any], algorithm: str) -> None:
    seed = config.get("experiment", {}).get("seed")
    expected_name = f"pilot-{algorithm}-qwen-0.5b-matched-seed-{seed}"
    if seed not in PILOT_SEEDS or config["experiment"].get("name") != expected_name:
        raise ValueError("pilot seed/name is not frozen")
    pilot = config.get("pilot", {})
    if pilot != {
        "family": PILOT_FAMILY,
        "disclaimer": PILOT_DISCLAIMER,
        "template_only": False,
        "approved_seeds": list(PILOT_SEEDS),
        "automatic_retries": 0,
        "pairing_key": "problem_id::generation:{0..3}",
        "execution_order": "prompt_major_then_generation_index",
    }:
        raise ValueError("pilot family metadata mismatch")
    if config.get("model") != {
        "name_or_path": PILOT_MODEL,
        "revision": PILOT_REVISION,
        "local_files_only": True,
        "dtype": "bfloat16",
        "use_qlora": False,
        "gradient_checkpointing": True,
    }:
        raise ValueError("pilot model identity mismatch")
    if config.get("prompt") != {"version": PROMPT_V1_STRICT_CONCISE}:
        raise ValueError("pilot prompt selector mismatch")
    if (
        config.get("prompt_version") != PROMPT_V1_STRICT_CONCISE
        or config.get("prompt_sha256") != PROMPT_V1_SHA256
        or config.get("renderer_version") != PROMPT_RENDERER_VERSION
    ):
        raise ValueError("pilot resolved prompt identity mismatch")
    if (
        config.get("reward") != {"policy": STAGED_REWARD_VERSION}
        or config.get("reward_policy_version") != STAGED_REWARD_VERSION
        or config.get("reward_policy_sha256") != STAGED_REWARD_SHA256
    ):
        raise ValueError("pilot resolved reward identity mismatch")
    metadata = parser_verifier_metadata()
    if any(config.get(key) != value for key, value in metadata.items()):
        raise ValueError("pilot parser/verifier identity mismatch")
    if config.get("lora") != POLICY_LORA:
        raise ValueError("pilot policy LoRA mismatch")
    manifest = validate_pilot_manifest()
    data = config.get("data", {})
    expected_data = {
        "pilot_manifest": str(PILOT_MANIFEST_PATH),
        "pilot_manifest_sha256": manifest["manifest_sha256"],
        "source_manifest": str(SOURCE_MANIFEST_PATH),
        "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
        "ordered_problem_ids": list(PILOT_PROBLEM_IDS),
        "unique_prompts": 4,
        "responses_per_prompt": 4,
        "max_train_samples": 16 if algorithm == "ppo" else 4,
    }
    if data != expected_data:
        raise ValueError("pilot data identity mismatch")
    generation = config.get("generation", {})
    common_generation = {
        "max_prompt_length": 512,
        "max_completion_length": 128,
        "temperature": 0.8,
        "top_p": 0.95,
    }
    if any(generation.get(key) != value for key, value in common_generation.items()):
        raise ValueError("pilot sampling identity mismatch")
    budget = config.get("budget", {})
    if any(
        budget.get(key) != value
        for key, value in {
            "max_completions": 16,
            "max_generated_tokens": 2048,
            "max_outer_updates": 1,
            "max_optimizer_steps": 1,
            "max_global_steps": 1,
            "max_checkpoints": 1,
            "gpu_hour_price_cny": 8.88,
        }.items()
    ):
        raise ValueError("pilot shared hard budget mismatch")
    artifacts = config.get("artifacts", {})
    if artifacts != {
        "independent_run_id": True,
        "independent_checkpoint": True,
        "inherit_checkpoint": False,
        "independent_full_backup": True,
        "checkpoint_count": 1,
    }:
        raise ValueError("pilot artifact isolation mismatch")
    reporting = config.get("reporting", {})
    if (
        reporting.get("disclaimer") != PILOT_DISCLAIMER
        or reporting.get("completion_matched_metrics") is not True
        or reporting.get("generated_token_normalized_metrics") is not True
    ):
        raise ValueError("pilot reporting contract mismatch")


def resolve_ppo_pilot_contract(config: dict[str, Any]) -> dict[str, Any]:
    training = config["training"]
    generation = config["generation"]
    per_device = training["per_device_train_batch_size"]
    accumulation = training["gradient_accumulation_steps"]
    local_batch = per_device * accumulation
    minibatches = training["num_mini_batches"]
    local_minibatch = local_batch // minibatches
    microbatches = local_minibatch // per_device
    outer_updates = (training["total_episodes"] + local_batch - 1) // local_batch
    optimizer_steps = (
        outer_updates * training["num_ppo_epochs"] * minibatches * microbatches // accumulation
    )
    return {
        "unique_prompts": config["data"]["unique_prompts"],
        "responses_per_prompt": config["data"]["responses_per_prompt"],
        "selected_dataset_records": config["data"]["max_train_samples"],
        "total_episodes": training["total_episodes"],
        "rollout_batch_size": local_batch,
        "micro_batch_size": per_device,
        "gradient_accumulation_steps": accumulation,
        "local_rollout_forward_batch_size": training["local_rollout_forward_batch_size"],
        "num_ppo_epochs": training["num_ppo_epochs"],
        "num_mini_batches": minibatches,
        "microbatches_per_minibatch": microbatches,
        "outer_updates": outer_updates,
        "optimizer_steps": optimizer_steps,
        "global_steps": outer_updates,
        "total_completions": training["total_episodes"],
        "max_completion_length": generation["max_new_tokens"],
        "total_generated_tokens": training["total_episodes"] * generation["max_new_tokens"],
        "num_generations_in_ppo_config": False,
        "authoritative_checkpoints": training["save_total_limit"],
        "episode_pair_keys": pilot_pair_keys(),
    }


def resolve_grpo_pilot_contract(config: dict[str, Any]) -> dict[str, Any]:
    generation = config["generation"]
    training = config["training"]
    generation_batch = generation["generation_batch_size"]
    per_device = training["per_device_train_batch_size"]
    return {
        "unique_prompts": generation_batch // generation["num_generations"],
        "responses_per_prompt": generation["num_generations"],
        "selected_dataset_records": config["data"]["max_train_samples"],
        "generation_batch_size": generation_batch,
        "micro_batch_size": per_device,
        "gradient_accumulation_steps": training["gradient_accumulation_steps"],
        "steps_per_generation": generation_batch // per_device,
        "num_iterations": training["num_iterations"],
        "outer_updates": training["max_steps"],
        "optimizer_steps": training["max_steps"],
        "global_steps": training["max_steps"],
        "total_completions": generation_batch,
        "max_completion_length": generation["max_completion_length"],
        "total_generated_tokens": generation_batch * generation["max_completion_length"],
        "authoritative_checkpoints": training["save_total_limit"],
        "completion_pair_keys": pilot_pair_keys(),
    }


def validate_pilot_config_content(config: dict[str, Any], algorithm: str) -> dict[str, Any]:
    if (
        algorithm not in {"ppo", "grpo"}
        or config.get("experiment", {}).get("algorithm") != algorithm
    ):
        raise ValueError("pilot algorithm mismatch")
    _common_contract(config, algorithm)
    if algorithm == "ppo":
        if "num_generations" in config["generation"]:
            raise ValueError("num_generations must not enter PPO pilot config")
        expected_training = {
            "max_steps": 1,
            "per_device_train_batch_size": 4,
            "gradient_accumulation_steps": 4,
            "total_episodes": 16,
            "num_ppo_epochs": 1,
            "num_mini_batches": 1,
            "local_rollout_forward_batch_size": 4,
            "save_total_limit": 1,
            "save_strategy": "steps",
            "save_steps": 1,
            "save_only_model": True,
            "push_to_hub": False,
            "report_to": [],
        }
        if config["generation"].get("max_new_tokens") != 128:
            raise ValueError("PPO pilot response length mismatch")
        if config["training"] != expected_training:
            raise ValueError("PPO pilot training contract mismatch")
        for key, value in {
            "max_ppo_epochs": 1,
            "max_minibatches": 1,
            "max_wall_time_seconds": 78,
            "max_vram_gib": 14,
            "max_gpu_hours": 0.0216666667,
            "max_estimated_cost_cny": 0.1924,
        }.items():
            if config["budget"].get(key) != value:
                raise ValueError("PPO pilot resource contract mismatch")
        contract = resolve_ppo_pilot_contract(config)
    else:
        expected_training = {
            "max_steps": 1,
            "per_device_train_batch_size": 4,
            "gradient_accumulation_steps": 4,
            "num_iterations": 1,
            "save_total_limit": 1,
            "save_strategy": "steps",
            "save_steps": 1,
            "save_only_model": True,
            "push_to_hub": False,
            "report_to": [],
        }
        if (
            config["generation"].get("num_generations") != 4
            or config["generation"].get("generation_batch_size") != 16
        ):
            raise ValueError("GRPO pilot generation contract mismatch")
        if config["training"] != expected_training:
            raise ValueError("GRPO pilot training contract mismatch")
        for key, value in {
            "max_wall_time_seconds": 40,
            "max_vram_gib": 7,
            "max_gpu_hours": 0.0111111111,
            "max_estimated_cost_cny": 0.0986666667,
        }.items():
            if config["budget"].get(key) != value:
                raise ValueError("GRPO pilot resource contract mismatch")
        contract = resolve_grpo_pilot_contract(config)
    expected = {
        "unique_prompts": 4,
        "responses_per_prompt": 4,
        "outer_updates": 1,
        "optimizer_steps": 1,
        "global_steps": 1,
        "total_completions": 16,
        "total_generated_tokens": 2048,
        "authoritative_checkpoints": 1,
    }
    if any(contract.get(key) != value for key, value in expected.items()):
        raise ValueError("matched pilot derived budget mismatch")
    return contract


def _load_registry() -> dict[str, str]:
    payload = json.loads(PILOT_REGISTRY_PATH.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or set(payload.get("configs", {})) != {
        f"configs/pilot/resolved/{algorithm}_seed_{seed}.json"
        for algorithm in ("ppo", "grpo")
        for seed in PILOT_SEEDS
    }:
        raise ValueError("pilot resolved config registry mismatch")
    return payload["configs"]


def validate_pilot_config_file(path: Path, algorithm: str) -> tuple[dict[str, Any], dict[str, Any]]:
    resolved = path.resolve()
    try:
        relative = str(resolved.relative_to(Path.cwd().resolve()))
    except ValueError as exc:
        raise ValueError("pilot config must be inside the repository") from exc
    registry = _load_registry()
    if relative not in registry or not relative.startswith(
        f"configs/pilot/resolved/{algorithm}_seed_"
    ):
        raise ValueError("pilot runner accepts only frozen resolved config files")
    if file_sha256(resolved) != registry[relative]:
        raise ValueError("pilot resolved config SHA256 mismatch")
    config = json.loads(resolved.read_text(encoding="utf-8"))
    contract = validate_pilot_config_content(config, algorithm)
    return config, contract


def enrich_pilot_config(
    config: dict[str, Any], contract: dict[str, Any], config_path: Path
) -> dict[str, Any]:
    from math_rlvr.config import resolve_training_config

    resolved = resolve_training_config(config)
    resolved["resolved_pilot_contract"] = contract
    resolved["resolved_config_path"] = str(config_path)
    resolved["resolved_config_sha256"] = file_sha256(config_path)
    return resolved


def validate_pilot_execution_authorization(
    config: dict[str, Any], config_path: Path, algorithm: str
) -> dict[str, Any]:
    frozen, contract = validate_pilot_config_file(config_path, algorithm)
    expected = enrich_pilot_config(frozen, contract, config_path)
    if config != expected:
        raise ValueError("resolved pilot execution config differs from frozen file")
    return contract


def pilot_run_order() -> list[dict[str, Any]]:
    return [
        {
            "sequence": index,
            "seed": seed,
            "algorithm": algorithm,
            "config": f"configs/pilot/resolved/{algorithm}_seed_{seed}.json",
        }
        for index, (seed, algorithm) in enumerate(RUN_ORDER, start=1)
    ]
