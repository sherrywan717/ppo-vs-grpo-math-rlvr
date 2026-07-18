"""Exact, hash-authorized execution evidence profiles for Stage D and the pilot."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from math_rlvr.contracts import parser_verifier_metadata
from math_rlvr.prompt import (
    PROMPT_RENDERER_VERSION,
    PROMPT_V1_SHA256,
    PROMPT_V1_STRICT_CONCISE,
    PROMPT_V2_FORMAL_MATH,
    PROMPT_V2_SHA256,
    ExperimentScope,
)
from math_rlvr.rewards.formal import FORMAL_REWARD_SHA256, FORMAL_REWARD_VERSION
from math_rlvr.rewards.staged import STAGED_REWARD_SHA256, STAGED_REWARD_VERSION

REPO_ROOT = Path(__file__).resolve().parents[3]
SOURCE_MANIFEST_SHA256 = "f7b3138c4fd29063ee05b568462c9cc5c2f8697ee63b8b208949b1b3998ce196"
PILOT_MANIFEST_SHA256 = "0235210e038bc27ebf2e7218691f36f09c8e11f0bbc743f46a5318a279f6bc1f"
MODEL_REPO = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"

_MAIN_FORMAL_CONFIG_SHA256 = {
    "configs/main/ppo.yaml": (
        "ppo", "1ced44a672fa3a5dcf9871bd8c1893a3bdad641d756dcf9de226b20440d1ad74"
    ),
    "configs/main/grpo.yaml": (
        "grpo", "fc1b0c73de431d81e9e827107d8491aba4d54b92f7e04fd4678b6fd828b6f675"
    ),
}

_CONFIG_SHA256 = {
    "configs/smoke/ppo.yaml": "547e67360fd73385c688f6d1b3b10d95cf191b70456d1b893870540b6de9f668",
    "configs/smoke/grpo.yaml": "068ff8d742849ffa0d43ccf6f4e74898e08c5f031c0f837c18ac8e5b183d8979",
    "configs/pilot/resolved/ppo_seed_42.json": (
        "1daeba7e6cd5e0af43c7f7cb9db87b46d44608adf9fdf432dc7b2c34ea059fdd"
    ),
    "configs/pilot/resolved/ppo_seed_123.json": (
        "9da0ad35e943cdeda2da410c20eec73e6d105f0ef66f7f67b1be22950a0e43c5"
    ),
    "configs/pilot/resolved/ppo_seed_2026.json": (
        "d3255ddb849224a4d87a069d981fcacf85cb98a7afa986ff0a9fb284b7698044"
    ),
    "configs/pilot/resolved/grpo_seed_42.json": (
        "83992a9c312b3ea6ab87f33dce1d4e9572a9647bbdb72bd67a6e98e90c182ac8"
    ),
    "configs/pilot/resolved/grpo_seed_123.json": (
        "edec9ce1265dfaec8c712b2c65046fe860cbd3e10aab52cf31b6d5e0350c2a28"
    ),
    "configs/pilot/resolved/grpo_seed_2026.json": (
        "1d558da6ea57cfa074fee30868f1772c76c617920e4e93a4897be2e2b48d6b00"
    ),
}

_FORMAL_CONFIG_SHA256 = {
    "configs/formal_1p5b/resolved/ppo_seed_42.json": (
        "ppo", 42, "1093e87a8363a0a2a6ab640a6f723c04cb6cfb22edef2e38a8c3a0062693ec43"
    ),
    "configs/formal_1p5b/resolved/grpo_seed_42.json": (
        "grpo", 42, "3371d23166d01834c67830eb8dfd51a02d4af483687b0b29e941194174099199"
    ),
    "configs/formal_1p5b/resolved/grpo_seed_123.json": (
        "grpo", 123, "cc95138f50f37fafa76766d3a08b0995ffd5e0bf87cd7b9050acedb5e0bbc75e"
    ),
    "configs/formal_1p5b/resolved/ppo_seed_123.json": (
        "ppo", 123, "3d6cc1f30f7b72bfadb5191613298ac3f64a1ba3c699cc8d1e30ce147218c15e"
    ),
    "configs/formal_1p5b/resolved/ppo_seed_2026.json": (
        "ppo", 2026, "6d041a4bd14be7ca3f5024c0415f56ea5e5b8c20f81d0b40d59323835389f25b"
    ),
    "configs/formal_1p5b/resolved/grpo_seed_2026.json": (
        "grpo", 2026, "168044574b2ef6b293d214560dd4032bb603ac0c25b2efe8589af423d75178f4"
    ),
}


@dataclass(frozen=True)
class ValidatedExperimentScope:
    """Scope evidence selected only by an exact repository path and raw SHA256."""

    scope: ExperimentScope
    algorithm: str
    config_path: str
    config_sha256: str
    expected_run_profile: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "scope": self.scope.value,
            "algorithm": self.algorithm,
            "config_path": self.config_path,
            "config_sha256": self.config_sha256,
            "expected_run_profile": self.expected_run_profile,
        }


@dataclass(frozen=True)
class ExpectedRunContract:
    """Immutable evidence contract selected only by an exact path/SHA allowlist."""

    profile: str
    experiment_scope: ExperimentScope
    algorithm: str
    config_path: str
    config_sha256: str
    manifest_sha256: str
    model_repo: str
    model_revision: str
    local_files_only: bool
    dtype: str
    policy_lora_rank: int
    policy_lora_alpha: int
    policy_lora_dropout: float
    policy_lora_targets: tuple[str, ...]
    temperature: float
    top_p: float
    max_completion_length: int
    prompt_version: str
    prompt_sha256: str
    renderer_version: str
    reward_version: str
    reward_sha256: str
    parser_version: str
    parser_sha256: str
    verifier_version: str
    verifier_sha256: str
    expected_prompt_count: int
    responses_per_prompt: int
    expected_completions: int
    generated_token_cap: int
    expected_updates: int
    expected_optimizer_steps: int
    expected_global_steps: int
    expected_microsteps: int
    expected_ppo_epochs: int
    expected_minibatches: int
    problem_ids: tuple[str, ...]

    @property
    def pair_keys(self) -> tuple[str, ...]:
        return tuple(
            f"{problem_id}::generation:{generation_index}"
            for problem_id in self.problem_ids
            for generation_index in range(self.responses_per_prompt)
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["experiment_scope"] = self.experiment_scope.value
        payload["problem_ids"] = list(self.problem_ids)
        payload["policy_lora_targets"] = list(self.policy_lora_targets)
        payload["pair_keys"] = list(self.pair_keys)
        return payload


def _relative_config_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError as exc:
        raise ValueError("execution config must be inside the repository") from exc


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _profile_shape(relative: str) -> tuple[str, str, int, int, int, int, int, int, int]:
    if relative == "configs/smoke/ppo.yaml":
        return "ppo_stage_d_smoke", "ppo", 4, 1, 4, 512, 0, 1, 1
    if relative == "configs/smoke/grpo.yaml":
        return "grpo_stage_d_smoke", "grpo", 2, 4, 8, 1024, 4, 0, 0
    if relative.startswith("configs/pilot/resolved/ppo_seed_"):
        return "ppo_matched_pilot", "ppo", 4, 4, 16, 2048, 0, 1, 1
    if relative.startswith("configs/pilot/resolved/grpo_seed_"):
        return "grpo_matched_pilot", "grpo", 4, 4, 16, 2048, 4, 0, 0
    if relative.startswith("configs/formal_1p5b/resolved/ppo_seed_"):
        return "ppo_formal_1p5b", "ppo", 128, 4, 512, 131_072, 0, 1, 1
    if relative.startswith("configs/formal_1p5b/resolved/grpo_seed_"):
        return "grpo_formal_1p5b", "grpo", 128, 4, 512, 131_072, 128, 0, 0
    raise ValueError("config path has no protected execution evidence profile")


def _scope_for_profile(profile: str) -> ExperimentScope:
    if profile.endswith("_stage_d_smoke"):
        return ExperimentScope.STAGE_D_SMOKE
    if profile.endswith("_matched_pilot"):
        return ExperimentScope.MATCHED_0P5B_PILOT
    if profile.endswith("_formal_1p5b"):
        return ExperimentScope.MAIN_FORMAL
    raise ValueError("execution profile has no validated experiment scope")


def expected_run_contract(config_path: Path, algorithm: str) -> ExpectedRunContract:
    """Resolve an exact profile; caller-provided numeric limits are never accepted."""
    relative = _relative_config_path(config_path)
    formal_entry = _FORMAL_CONFIG_SHA256.get(relative)
    if formal_entry is not None:
        _, formal_seed, expected_sha = formal_entry
        if formal_seed == 2026:
            raise ValueError("formal seed 2026 is reserved_not_scheduled")
    else:
        expected_sha = _CONFIG_SHA256.get(relative)
        if expected_sha is None:
            raise ValueError("config path has no protected execution evidence profile")
    actual_sha = _file_sha256(config_path.resolve())
    if actual_sha != expected_sha:
        raise ValueError("protected execution config SHA256 mismatch")
    (
        profile,
        expected_algorithm,
        prompts,
        responses,
        completions,
        token_cap,
        microsteps,
        ppo_epochs,
        minibatches,
    ) = _profile_shape(relative)
    if algorithm != expected_algorithm:
        raise ValueError("execution profile algorithm mismatch")
    is_formal = profile.endswith("_formal_1p5b")
    if is_formal:
        from math_rlvr.contracts import formal_parser_verifier_metadata
        from math_rlvr.training.formal import (
            FORMAL_MODEL,
            FORMAL_REVISION,
            formal_training_schedule,
        )

        metadata = formal_parser_verifier_metadata()
        problem_ids = tuple(formal_training_schedule()["ordered_problem_ids"])
        manifest_sha = "553939ce40ef20af86f5eabe987bff42814f07e9d40ddf1c4cde1208dcc96dd0"
        model_repo = FORMAL_MODEL
        model_revision = FORMAL_REVISION
        max_completion_length = 256
        prompt_version = PROMPT_V2_FORMAL_MATH
        prompt_sha = PROMPT_V2_SHA256
        reward_version = FORMAL_REWARD_VERSION
        reward_sha = FORMAL_REWARD_SHA256
        updates = 32
    else:
        metadata = parser_verifier_metadata()
        problem_ids = tuple(f"countdown:train:{index}" for index in range(prompts))
        manifest_sha = (
            PILOT_MANIFEST_SHA256 if "matched_pilot" in profile else SOURCE_MANIFEST_SHA256
        )
        model_repo = MODEL_REPO
        model_revision = MODEL_REVISION
        max_completion_length = 128
        prompt_version = PROMPT_V1_STRICT_CONCISE
        prompt_sha = PROMPT_V1_SHA256
        reward_version = STAGED_REWARD_VERSION
        reward_sha = STAGED_REWARD_SHA256
        updates = 1
    parser = metadata["parser_contract"]
    verifier = metadata["verifier_contract"]
    return ExpectedRunContract(
        profile=profile,
        experiment_scope=_scope_for_profile(profile),
        algorithm=algorithm,
        config_path=relative,
        config_sha256=expected_sha,
        manifest_sha256=manifest_sha,
        model_repo=model_repo,
        model_revision=model_revision,
        local_files_only=True,
        dtype="bfloat16",
        policy_lora_rank=16,
        policy_lora_alpha=32,
        policy_lora_dropout=0.0,
        policy_lora_targets=("q_proj", "k_proj", "v_proj", "o_proj"),
        temperature=0.8,
        top_p=0.95,
        max_completion_length=max_completion_length,
        prompt_version=prompt_version,
        prompt_sha256=prompt_sha,
        renderer_version=PROMPT_RENDERER_VERSION,
        reward_version=reward_version,
        reward_sha256=reward_sha,
        parser_version=parser["contract_version"],
        parser_sha256=parser["contract_sha256"],
        verifier_version=verifier["contract_version"],
        verifier_sha256=verifier["contract_sha256"],
        expected_prompt_count=prompts,
        responses_per_prompt=responses,
        expected_completions=completions,
        generated_token_cap=token_cap,
        expected_updates=updates,
        expected_optimizer_steps=updates,
        expected_global_steps=updates,
        expected_microsteps=microsteps,
        expected_ppo_epochs=ppo_epochs,
        expected_minibatches=minibatches,
        problem_ids=problem_ids,
    )


def validated_experiment_scope(
    config_path: Path, algorithm: str
) -> ValidatedExperimentScope:
    """Resolve scope from the sole exact path/SHA allowlist; names are irrelevant."""
    relative = _relative_config_path(config_path)
    if relative in _CONFIG_SHA256:
        contract = expected_run_contract(config_path, algorithm)
        return ValidatedExperimentScope(
            scope=contract.experiment_scope,
            algorithm=algorithm,
            config_path=relative,
            config_sha256=contract.config_sha256,
            expected_run_profile=contract.profile,
        )
    formal = _FORMAL_CONFIG_SHA256.get(relative)
    if formal is not None:
        expected_algorithm, seed, expected_sha = formal
        if algorithm != expected_algorithm:
            raise ValueError("validated formal scope algorithm mismatch")
        if _file_sha256(config_path.resolve()) != expected_sha:
            raise ValueError("validated formal scope config SHA256 mismatch")
        profile = None
        if seed != 2026:
            profile = f"{algorithm}_formal_1p5b"
        return ValidatedExperimentScope(
            scope=ExperimentScope.MAIN_FORMAL,
            algorithm=algorithm,
            config_path=relative,
            config_sha256=expected_sha,
            expected_run_profile=profile,
        )
    main = _MAIN_FORMAL_CONFIG_SHA256.get(relative)
    if main is None:
        raise ValueError("config path has no validated experiment scope")
    expected_algorithm, expected_sha = main
    if algorithm != expected_algorithm:
        raise ValueError("validated experiment scope algorithm mismatch")
    if _file_sha256(config_path.resolve()) != expected_sha:
        raise ValueError("validated experiment scope config SHA256 mismatch")
    return ValidatedExperimentScope(
        scope=ExperimentScope.MAIN_FORMAL,
        algorithm=algorithm,
        config_path=relative,
        config_sha256=expected_sha,
        expected_run_profile=None,
    )


def validated_scope_from_config(
    config: dict[str, Any], algorithm: str
) -> ValidatedExperimentScope:
    """Revalidate serialized scope evidence against the same exact allowlist."""
    payload = config.get("validated_experiment_scope")
    if not isinstance(payload, dict):
        raise ValueError("resolved config is missing validated experiment scope")
    path = payload.get("config_path")
    if not isinstance(path, str):
        raise ValueError("validated experiment scope path is missing")
    validated = validated_experiment_scope(Path(path), algorithm)
    if payload != validated.to_dict():
        raise ValueError("serialized experiment scope differs from path/SHA validation")
    if config.get("resolved_config_path") != validated.config_path:
        raise ValueError("resolved config path differs from validated experiment scope")
    if config.get("resolved_config_sha256") != validated.config_sha256:
        raise ValueError("resolved config SHA256 differs from validated experiment scope")
    return validated


def expected_run_contract_for_config(config: dict[str, Any], algorithm: str) -> ExpectedRunContract:
    path = config.get("resolved_config_path")
    if path is None:
        path = f"configs/smoke/{algorithm}.yaml"
    contract = expected_run_contract(Path(path), algorithm)
    if config.get("resolved_config_sha256") not in (None, contract.config_sha256):
        raise ValueError("resolved config SHA256 differs from protected execution profile")
    if config.get("validated_experiment_scope") is None:
        scope = validated_experiment_scope(Path(path), algorithm)
    else:
        scope = validated_scope_from_config(config, algorithm)
    if (
        scope.scope is not contract.experiment_scope
        or scope.expected_run_profile != contract.profile
    ):
        raise ValueError("ExpectedRunContract scope differs from resolved experiment scope")
    identity_config = config
    if config.get("prompt_sha256") is None or config.get("reward_policy_sha256") is None:
        from math_rlvr.config import resolve_training_config

        identity_config = resolve_training_config(config, scope)
    identity = {
        "algorithm": identity_config.get("experiment", {}).get("algorithm"),
        "model_repo": identity_config.get("model", {}).get("name_or_path"),
        "model_revision": identity_config.get("model", {}).get("revision"),
        "local_files_only": identity_config.get("model", {}).get("local_files_only"),
        "dtype": identity_config.get("model", {}).get("dtype"),
        "policy_lora_rank": identity_config.get("lora", {}).get("rank"),
        "policy_lora_alpha": identity_config.get("lora", {}).get("alpha"),
        "policy_lora_dropout": float(identity_config.get("lora", {}).get("dropout")),
        "policy_lora_targets": tuple(identity_config.get("lora", {}).get("target_modules", [])),
        "temperature": identity_config.get("generation", {}).get("temperature"),
        "top_p": identity_config.get("generation", {}).get("top_p"),
        "max_completion_length": identity_config.get("generation", {}).get("max_completion_length"),
        "prompt_version": identity_config.get("prompt_version"),
        "prompt_sha256": identity_config.get("prompt_sha256"),
        "renderer_version": identity_config.get("renderer_version"),
        "reward_version": identity_config.get("reward_policy_version"),
        "reward_sha256": identity_config.get("reward_policy_sha256"),
    }
    expected = {
        "algorithm": contract.algorithm,
        "model_repo": contract.model_repo,
        "model_revision": contract.model_revision,
        "local_files_only": contract.local_files_only,
        "dtype": contract.dtype,
        "policy_lora_rank": contract.policy_lora_rank,
        "policy_lora_alpha": contract.policy_lora_alpha,
        "policy_lora_dropout": contract.policy_lora_dropout,
        "policy_lora_targets": contract.policy_lora_targets,
        "temperature": contract.temperature,
        "top_p": contract.top_p,
        "max_completion_length": contract.max_completion_length,
        "prompt_version": contract.prompt_version,
        "prompt_sha256": contract.prompt_sha256,
        "renderer_version": contract.renderer_version,
        "reward_version": contract.reward_version,
        "reward_sha256": contract.reward_sha256,
    }
    if identity != expected:
        raise ValueError("resolved config identity differs from protected execution profile")
    if "matched_pilot" in contract.profile or "formal_1p5b" in contract.profile:
        expected_parser = {
            "contract_version": contract.parser_version,
            "contract_sha256": contract.parser_sha256,
        }
        expected_verifier = {
            "contract_version": contract.verifier_version,
            "contract_sha256": contract.verifier_sha256,
        }
        if config.get("parser_contract") != expected_parser or config.get(
            "verifier_contract"
        ) != expected_verifier:
            raise ValueError("parser/verifier differs from protected execution profile")
    budget = config.get("budget", {})
    if (
        budget.get("max_completions") != contract.expected_completions
        or budget.get("max_generated_tokens") != contract.generated_token_cap
    ):
        raise ValueError("resolved completion/token budget differs from protected profile")
    data = config.get("data", {})
    if "matched_pilot" in contract.profile:
        if data.get("pilot_manifest_sha256") != contract.manifest_sha256:
            raise ValueError("pilot manifest differs from protected execution profile")
    else:
        manifest_path = Path(data.get("manifest", ""))
        if not manifest_path.is_file() or _file_sha256(manifest_path) != contract.manifest_sha256:
            raise ValueError("Stage D manifest differs from protected execution profile")
    return contract


def protected_execution_profiles() -> tuple[ExpectedRunContract, ...]:
    """Return the four semantic profiles, choosing seed 42 for pilot path evidence."""
    paths = (
        (Path("configs/smoke/ppo.yaml"), "ppo"),
        (Path("configs/smoke/grpo.yaml"), "grpo"),
        (Path("configs/pilot/resolved/ppo_seed_42.json"), "ppo"),
        (Path("configs/pilot/resolved/grpo_seed_42.json"), "grpo"),
    )
    return tuple(expected_run_contract(path, algorithm) for path, algorithm in paths)
