"""CPU-safe formal 1.5B training contracts and resolved-config authorization."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import yaml

FORMAL_FAMILY = "formal_1p5b_v1"
FORMAL_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
FORMAL_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
FORMAL_PROMPT = "prompt_v2_formal_math"
FORMAL_REWARD = "shaped_v3_domain"
FORMAL_SEEDS = (42, 123, 2026)
FORMAL_ACTIVE_SEEDS = (42, 123)
FORMAL_RESERVED_SEEDS = (2026,)
FORMAL_UPDATES = 32
FORMAL_UNIQUE_PROMPTS = 128
FORMAL_RESPONSES_PER_PROMPT = 4
FORMAL_COMPLETIONS = 512
FORMAL_MAX_COMPLETION_LENGTH = 256
FORMAL_TOKEN_CAP = 131_072
FORMAL_SCHEDULE_VERSION = "deterministic_domain_interleave_2_gsm8k_2_math"
FORMAL_SCHEDULE_SHA256 = "a4b3745e0359757df8441dffa1250c89397457cb5aaf77165d26701e9488b6ee"
FORMAL_ORDERED_PROBLEM_IDS_SHA256 = (
    "fc15d338287a6212b71608f1e066991c162698a9c9196055a3eace9e46123454"
)
FORMAL_DATA_REGISTRY_SHA256 = "d7c53f6180187711da780a3a1f81f8b45e6164ddc9f115eac2fb6ae3e1fe7393"
FORMAL_CHECKPOINT_STEPS = (8, 16, 24, 32)
FORMAL_CONFIG_ROOT = Path("configs/formal_1p5b")
FORMAL_REGISTRY_PATH = FORMAL_CONFIG_ROOT / "resolved_config_sha256.json"
FORMAL_ACTIVE_SUITE_PATH = FORMAL_CONFIG_ROOT / "active_suite.json"
TRAIN_MANIFEST = Path("/root/autodl-tmp/datasets/math_rlvr/manifests/train_core_128.json")
VALIDATION_MANIFEST = Path("/root/autodl-tmp/datasets/math_rlvr/manifests/validation_64.json")
TRAIN_MANIFEST_SHA256 = "553939ce40ef20af86f5eabe987bff42814f07e9d40ddf1c4cde1208dcc96dd0"
VALIDATION_MANIFEST_SHA256 = "83eee5f6191f003c3c5d8f273adb2a5631c848d1f196bfe34891efeca658e70d"
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
)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest_rows(path: Path, digest: str, expected_count: int) -> list[dict[str, Any]]:
    if not path.is_file() or file_sha256(path) != digest:
        raise ValueError(f"formal manifest identity mismatch: {path.name}")
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or len(rows) != expected_count:
        raise ValueError(f"formal manifest row count mismatch: {path.name}")
    ids = [row.get("problem_id") for row in rows]
    hashes = [row.get("content_hash") for row in rows]
    if any(not isinstance(value, str) or not value for value in ids + hashes):
        raise ValueError(f"formal manifest identity fields missing: {path.name}")
    if len(ids) != len(set(ids)) or len(hashes) != len(set(hashes)):
        raise ValueError(f"formal manifest contains duplicate identities: {path.name}")
    return rows


def validate_formal_manifests() -> dict[str, Any]:
    train = _manifest_rows(TRAIN_MANIFEST, TRAIN_MANIFEST_SHA256, 128)
    validation = _manifest_rows(VALIDATION_MANIFEST, VALIDATION_MANIFEST_SHA256, 64)
    train_hashes = {row["content_hash"] for row in train}
    validation_hashes = {row["content_hash"] for row in validation}
    if train_hashes & validation_hashes:
        raise ValueError("formal train/validation manifests overlap")
    return {
        "train_rows": len(train),
        "validation_rows": len(validation),
        "train_manifest_sha256": TRAIN_MANIFEST_SHA256,
        "validation_manifest_sha256": VALIDATION_MANIFEST_SHA256,
    }


def formal_training_schedule() -> dict[str, Any]:
    rows = _manifest_rows(TRAIN_MANIFEST, TRAIN_MANIFEST_SHA256, 128)
    gsm8k = [row for row in rows if row["source"] == "gsm8k"]
    math_rows = [row for row in rows if row["source"] == "math"]
    if len(gsm8k) != 64 or len(math_rows) != 64:
        raise ValueError("formal train manifest must contain 64 GSM8K and 64 MATH rows")
    ordered = []
    for update in range(FORMAL_UPDATES):
        ordered.extend(
            (
                gsm8k[2 * update],
                gsm8k[2 * update + 1],
                math_rows[2 * update],
                math_rows[2 * update + 1],
            )
        )
    payload = {
        "schedule_version": FORMAL_SCHEDULE_VERSION,
        "updates": FORMAL_UPDATES,
        "prompts_per_update": 4,
        "domain_pattern": ["gsm8k", "gsm8k", "math", "math"],
        "ordered_problem_ids": [row["problem_id"] for row in ordered],
    }
    groups = [payload["ordered_problem_ids"][index : index + 4] for index in range(0, 128, 4)]
    encoded = json.dumps(groups, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    if hashlib.sha256(encoded).hexdigest() != FORMAL_SCHEDULE_SHA256:
        raise ValueError("formal training schedule SHA256 mismatch")
    return payload


def formal_pair_keys() -> list[str]:
    return [
        f"{problem_id}::generation:{generation_index}"
        for problem_id in formal_training_schedule()["ordered_problem_ids"]
        for generation_index in range(FORMAL_RESPONSES_PER_PROMPT)
    ]


def resolve_ppo_formal_contract(config: dict[str, Any]) -> dict[str, Any]:
    training = config["training"]
    generation = config["generation"]
    per_device = training["per_device_train_batch_size"]
    accumulation = training["gradient_accumulation_steps"]
    rollout_batch = per_device * accumulation
    local_minibatch = rollout_batch // training["num_mini_batches"]
    outer_updates = math.ceil(training["total_episodes"] / rollout_batch)
    return {
        "unique_prompts": FORMAL_UNIQUE_PROMPTS,
        "responses_per_prompt": FORMAL_RESPONSES_PER_PROMPT,
        "selected_dataset_records": FORMAL_UNIQUE_PROMPTS,
        "total_episodes": training["total_episodes"],
        "rollout_batch_size": rollout_batch,
        "micro_batch_size": per_device,
        "gradient_accumulation_steps": accumulation,
        "local_rollout_forward_batch_size": training["local_rollout_forward_batch_size"],
        "num_ppo_epochs": training["num_ppo_epochs"],
        "num_mini_batches": training["num_mini_batches"],
        "microbatches_per_minibatch": local_minibatch // per_device,
        "outer_updates": outer_updates,
        "optimizer_steps": outer_updates
        * training["num_ppo_epochs"]
        * training["num_mini_batches"],
        "global_steps": outer_updates,
        "total_completions": training["total_episodes"],
        "max_completion_length": generation["max_new_tokens"],
        "total_generated_tokens": training["total_episodes"] * generation["max_new_tokens"],
        "num_generations_in_ppo_config": False,
        "checkpoint_steps": list(FORMAL_CHECKPOINT_STEPS),
        "authoritative_checkpoints": len(FORMAL_CHECKPOINT_STEPS),
    }


def resolve_grpo_formal_contract(config: dict[str, Any]) -> dict[str, Any]:
    generation = config["generation"]
    training = config["training"]
    generation_batch = generation["generation_batch_size"]
    steps_per_generation = generation_batch // training["per_device_train_batch_size"]
    return {
        "unique_prompts": FORMAL_UNIQUE_PROMPTS,
        "responses_per_prompt": generation["num_generations"],
        "selected_dataset_records": FORMAL_UNIQUE_PROMPTS,
        "generation_batch_size": generation_batch,
        "micro_batch_size": training["per_device_train_batch_size"],
        "gradient_accumulation_steps": training["gradient_accumulation_steps"],
        "steps_per_generation": steps_per_generation,
        "num_iterations": training["num_iterations"],
        "outer_updates": training["max_steps"],
        "optimizer_steps": training["max_steps"],
        "global_steps": training["max_steps"],
        "training_microsteps": training["max_steps"] * training["gradient_accumulation_steps"],
        "total_completions": generation_batch * training["max_steps"],
        "max_completion_length": generation["max_completion_length"],
        "total_generated_tokens": generation_batch
        * training["max_steps"]
        * generation["max_completion_length"],
        "checkpoint_steps": list(FORMAL_CHECKPOINT_STEPS),
        "authoritative_checkpoints": len(FORMAL_CHECKPOINT_STEPS),
    }


def validate_formal_config_content(config: dict[str, Any], algorithm: str) -> dict[str, Any]:
    if (
        algorithm not in {"ppo", "grpo"}
        or config.get("experiment", {}).get("algorithm") != algorithm
    ):
        raise ValueError("formal algorithm mismatch")
    seed = config["experiment"].get("seed")
    if seed not in FORMAL_SEEDS or config["experiment"].get("name") != (
        f"formal-{algorithm}-qwen-1.5b-seed-{seed}"
    ):
        raise ValueError("formal seed/name mismatch")
    formal = config.get("formal", {})
    if formal != {
        "family": FORMAL_FAMILY,
        "template_only": False,
        "approved_seeds": list(FORMAL_SEEDS),
        "automatic_retries": 0,
        "execution_order": FORMAL_SCHEDULE_VERSION,
        "checkpoint_steps": list(FORMAL_CHECKPOINT_STEPS),
        "validation_steps": list(FORMAL_CHECKPOINT_STEPS),
        "final_test_checkpoint_step": 32,
    }:
        raise ValueError("formal family metadata mismatch")
    if config.get("model") != {
        "name_or_path": FORMAL_MODEL,
        "revision": FORMAL_REVISION,
        "local_files_only": True,
        "dtype": "bfloat16",
        "use_qlora": False,
        "gradient_checkpointing": True,
    }:
        raise ValueError("formal model identity mismatch")
    if config.get("prompt") != {"version": FORMAL_PROMPT}:
        raise ValueError("formal prompt selector mismatch")
    if config.get("reward") != {"policy": FORMAL_REWARD}:
        raise ValueError("formal reward selector mismatch")
    if config.get("lora") != POLICY_LORA:
        raise ValueError("formal policy LoRA mismatch")
    manifests = validate_formal_manifests()
    if config.get("data") != {
        "manifest": str(TRAIN_MANIFEST),
        "manifest_sha256": manifests["train_manifest_sha256"],
        "validation_manifest": str(VALIDATION_MANIFEST),
        "validation_manifest_sha256": manifests["validation_manifest_sha256"],
        "max_train_samples": 128,
        "validation_samples": 64,
        "unique_prompts": 128,
        "responses_per_prompt": 4,
        "ordering": FORMAL_SCHEDULE_VERSION,
        "schedule_sha256": FORMAL_SCHEDULE_SHA256,
        "ordered_problem_ids_sha256": FORMAL_ORDERED_PROBLEM_IDS_SHA256,
        "data_registry": "configs/formal_1p5b/data_registry.json",
        "data_registry_sha256": FORMAL_DATA_REGISTRY_SHA256,
    }:
        raise ValueError("formal data identity mismatch")
    generation = config.get("generation", {})
    common_generation = {
        "max_prompt_length": 832,
        "max_completion_length": 256,
        "temperature": 0.8,
        "top_p": 0.95,
    }
    if any(generation.get(key) != value for key, value in common_generation.items()):
        raise ValueError("formal generation identity mismatch")
    common_training = {
        "max_steps": 32,
        "per_device_train_batch_size": 4,
        "gradient_accumulation_steps": 4,
        "learning_rate": 1e-5,
        "logging_steps": 1,
        "save_total_limit": 4,
        "save_strategy": "steps",
        "save_steps": 8,
        "save_only_model": True,
        "push_to_hub": False,
        "report_to": [],
    }
    if any(config.get("training", {}).get(key) != value for key, value in common_training.items()):
        raise ValueError("formal common training contract mismatch")
    if algorithm == "ppo":
        if "num_generations" in generation or generation.get("max_new_tokens") != 256:
            raise ValueError("formal PPO generation contract mismatch")
        expected = {
            **common_training,
            "total_episodes": 512,
            "num_ppo_epochs": 1,
            "num_mini_batches": 1,
            "local_rollout_forward_batch_size": 4,
        }
        if config["training"] != expected:
            raise ValueError("formal PPO training contract mismatch")
        contract = resolve_ppo_formal_contract(config)
    else:
        if generation.get("num_generations") != 4 or generation.get("generation_batch_size") != 16:
            raise ValueError("formal GRPO generation contract mismatch")
        expected = {
            **common_training,
            "num_iterations": 1,
            "shuffle_dataset": False,
            "dataloader_drop_last": True,
            "dataloader_num_workers": 0,
        }
        if config["training"] != expected:
            raise ValueError("formal GRPO training contract mismatch")
        contract = resolve_grpo_formal_contract(config)
    for key, value in {
        "unique_prompts": 128,
        "responses_per_prompt": 4,
        "outer_updates": 32,
        "optimizer_steps": 32,
        "global_steps": 32,
        "total_completions": 512,
        "total_generated_tokens": 131_072,
        "authoritative_checkpoints": 4,
    }.items():
        if contract.get(key) != value:
            raise ValueError(f"formal derived budget mismatch: {key}")
    budget = config.get("budget", {})
    for key, value in {
        "max_completions": 512,
        "max_generated_tokens": 131_072,
        "max_outer_updates": 32,
        "max_optimizer_steps": 32,
        "max_global_steps": 32,
        "max_checkpoints": 4,
        "gpu_hour_price_cny": 8.88,
    }.items():
        if budget.get(key) != value:
            raise ValueError(f"formal hard budget mismatch: {key}")
    from math_rlvr.contracts import formal_parser_verifier_metadata
    from math_rlvr.prompt import prompt_metadata
    from math_rlvr.rewards.staged import reward_metadata_from_config

    for key, value in prompt_metadata(FORMAL_PROMPT).items():
        if key in config and config[key] != value:
            raise ValueError(f"formal prompt identity mismatch: {key}")
    reward_metadata_from_config(config)
    for key, value in formal_parser_verifier_metadata().items():
        if key in config and config[key] != value:
            raise ValueError(f"formal parser/verifier identity mismatch: {key}")
    return contract


def _registry() -> dict[str, str]:
    payload = json.loads(FORMAL_REGISTRY_PATH.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("formal config registry schema mismatch")
    return payload.get("configs", {})


def validate_formal_config_file(
    path: Path, algorithm: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    resolved = path.resolve()
    relative = resolved.relative_to(Path.cwd().resolve()).as_posix()
    registry = _registry()
    if relative not in registry or not relative.startswith(
        f"configs/formal_1p5b/resolved/{algorithm}_seed_"
    ):
        raise ValueError("formal runner accepts only frozen resolved config descriptors")
    if file_sha256(resolved) != registry[relative]:
        raise ValueError("formal resolved config SHA256 mismatch")
    descriptor = json.loads(resolved.read_text(encoding="utf-8"))
    seed = descriptor.get("seed")
    template_path = Path(descriptor.get("template", ""))
    if (
        descriptor
        != {
            "schema_version": 1,
            "family": FORMAL_FAMILY,
            "algorithm": algorithm,
            "seed": seed,
            "template": f"configs/formal_1p5b/{algorithm}_1p5b.yaml",
            "template_sha256": registry.get(f"configs/formal_1p5b/{algorithm}_1p5b.yaml"),
            "training_schedule": {
                "strategy": FORMAL_SCHEDULE_VERSION,
                "schedule_sha256": FORMAL_SCHEDULE_SHA256,
                "ordered_problem_ids_sha256": FORMAL_ORDERED_PROBLEM_IDS_SHA256,
            },
        }
        or seed not in FORMAL_SEEDS
    ):
        raise ValueError("formal resolved descriptor mismatch")
    if file_sha256(template_path) != descriptor["template_sha256"]:
        raise ValueError("formal template SHA256 mismatch")
    config = yaml.safe_load(template_path.read_text(encoding="utf-8"))
    config["experiment"] = {
        "name": f"formal-{algorithm}-qwen-1.5b-seed-{seed}",
        "algorithm": algorithm,
        "seed": seed,
    }
    config["formal"]["template_only"] = False
    contract = validate_formal_config_content(config, algorithm)
    config["resolved_config_path"] = relative
    config["resolved_config_sha256"] = registry[relative]
    config["resolved_template_path"] = descriptor["template"]
    config["resolved_template_sha256"] = descriptor["template_sha256"]
    config["resolved_formal_contract"] = contract
    from math_rlvr.contracts import formal_parser_verifier_metadata
    from math_rlvr.prompt import prompt_metadata
    from math_rlvr.rewards.staged import reward_metadata_from_config

    config.update(prompt_metadata(FORMAL_PROMPT))
    config.update(reward_metadata_from_config(config))
    config.update(formal_parser_verifier_metadata())
    config["model_identity_path"] = "configs/formal_1p5b/model_identity.json"
    config["data_registry_path"] = "configs/formal_1p5b/data_registry.json"
    from math_rlvr.training.execution_contract import validated_experiment_scope

    scope = validated_experiment_scope(Path(relative), algorithm)
    config["validated_experiment_scope"] = scope.to_dict()
    return config, contract


def validate_active_suite(path: Path = FORMAL_ACTIVE_SUITE_PATH) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected_sha256 = payload.get("active_suite_sha256")
    body = dict(payload)
    body.pop("active_suite_sha256", None)
    actual_sha256 = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError("formal active-suite SHA256 mismatch")
    active = payload.get("active_training_runs")
    expected = [
        {
            "position": index,
            "algorithm": algorithm,
            "seed": seed,
            "config": f"configs/formal_1p5b/resolved/{algorithm}_seed_{seed}.json",
            "config_sha256": _registry()[
                f"configs/formal_1p5b/resolved/{algorithm}_seed_{seed}.json"
            ],
        }
        for index, (seed, algorithm) in enumerate(RUN_ORDER, start=1)
    ]
    if active != expected or payload.get("active_seeds") != list(FORMAL_ACTIVE_SEEDS):
        raise ValueError("formal active-suite run order mismatch")
    reserved = payload.get("reserved_configs")
    expected_reserved = [
        {
            "algorithm": algorithm,
            "seed": seed,
            "config": f"configs/formal_1p5b/resolved/{algorithm}_seed_{seed}.json",
            "config_sha256": _registry()[
                f"configs/formal_1p5b/resolved/{algorithm}_seed_{seed}.json"
            ],
            "status": "reserved_not_scheduled",
        }
        for seed in FORMAL_RESERVED_SEEDS
        for algorithm in ("ppo", "grpo")
    ]
    if reserved != expected_reserved:
        raise ValueError("formal reserved config status mismatch")
    return payload


def formal_run_order() -> list[dict[str, Any]]:
    suite = validate_active_suite()
    return [
        {
            "sequence": row["position"],
            "seed": row["seed"],
            "algorithm": row["algorithm"],
            "config": row["config"],
            "config_sha256": row["config_sha256"],
            "automatic_retries": suite["automatic_retries"],
        }
        for row in suite["active_training_runs"]
    ]


def formal_reserved_configs() -> list[dict[str, Any]]:
    return [
        {
            "seed": seed,
            "algorithm": algorithm,
            "config": f"configs/formal_1p5b/resolved/{algorithm}_seed_{seed}.json",
            "status": "reserved_not_scheduled",
            "automatic_retries": 0,
        }
        for seed in FORMAL_RESERVED_SEEDS
        for algorithm in ("ppo", "grpo")
    ]
