"""CPU-only validation of the frozen formal evaluation protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from math_rlvr.contracts import formal_parser_verifier_metadata
from math_rlvr.prompt import PROMPT_V2_FORMAL_MATH, PROMPT_V2_SHA256
from math_rlvr.rewards.formal import FORMAL_REWARD_SHA256, FORMAL_REWARD_VERSION
from math_rlvr.training.formal import FORMAL_MODEL, FORMAL_REVISION, FORMAL_SEEDS
from math_rlvr.training.formal_data import validate_formal_data_registry

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = REPOSITORY_ROOT / "configs/formal_1p5b/evaluation.json"
FORMAL_EVALUATION_LABEL = "Formal 1.5B evaluation - frozen before model execution"


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_evaluation_config(path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    expected = config.get("config_sha256")
    body = dict(config)
    body.pop("config_sha256", None)
    if expected != canonical_sha256(body):
        raise ValueError("formal evaluation config SHA256 mismatch")
    return config


def validate_evaluation_config(
    config: dict[str, Any], phase: str, *, algorithm: str | None = None, seed: int | None = None
) -> dict[str, Any]:
    if phase not in {"baseline", "validation", "final"}:
        raise ValueError("unknown formal evaluation phase")
    if config.get("label") != FORMAL_EVALUATION_LABEL:
        raise ValueError("formal evaluation label mismatch")
    if config.get("model") != {
        "repo_id": FORMAL_MODEL,
        "revision": FORMAL_REVISION,
        "local_files_only": True,
        "dtype": "bfloat16",
    }:
        raise ValueError("formal evaluation model identity mismatch")
    if config.get("prompt") != {
        "version": PROMPT_V2_FORMAL_MATH,
        "sha256": PROMPT_V2_SHA256,
        "renderer": "math_rlvr.prompt.chat_template.v1",
    }:
        raise ValueError("formal evaluation prompt identity mismatch")
    if config.get("reward") != {
        "version": FORMAL_REWARD_VERSION,
        "sha256": FORMAL_REWARD_SHA256,
        "canonical_metrics_ignore_scalar_shaping": True,
    }:
        raise ValueError("formal evaluation reward identity mismatch")
    contracts = formal_parser_verifier_metadata()
    parser_contract = contracts["parser_contract"]
    verifier_contract = contracts["verifier_contract"]
    domain_contracts = contracts["domain_verifier_contracts"]
    if config.get("parser") != {
        "version": parser_contract["contract_version"],
        "sha256": parser_contract["contract_sha256"],
    }:
        raise ValueError("formal evaluation parser identity mismatch")
    if config.get("verifier_bundle") != {
        "version": verifier_contract["contract_version"],
        "sha256": verifier_contract["contract_sha256"],
        "gsm8k_sha256": domain_contracts["gsm8k"]["contract_sha256"],
        "math_sha256": domain_contracts["math"]["contract_sha256"],
    }:
        raise ValueError("formal evaluation verifier identity mismatch")
    sampling = config.get("sampling", {})
    if sampling != {
        "temperature": 0.8,
        "top_p": 0.95,
        "max_prompt_length": 512,
        "max_completion_length": 256,
    }:
        raise ValueError("formal evaluation sampling mismatch")
    if tuple(config.get("seeds", ())) != FORMAL_SEEDS:
        raise ValueError("formal evaluation seed mismatch")
    if seed is not None and seed not in FORMAL_SEEDS:
        raise ValueError("unapproved formal evaluation seed")
    if phase == "baseline" and algorithm is not None:
        raise ValueError("baseline is shared and algorithm-neutral")
    if phase in {"validation", "final"} and algorithm not in {"ppo", "grpo"}:
        raise ValueError("checkpoint evaluation requires PPO or GRPO")
    protocol = config.get("protocol", {})
    expected_counts = {
        "validation": {"unique_problems": 64, "responses_per_problem": 1, "completions": 64},
        "baseline": {
            "unique_problems": 400,
            "pass1_completions": 400,
            "pass4_subset_problems": 100,
            "pass4_completions": 400,
            "completions_per_seed": 800,
        },
        "final": {
            "unique_problems": 400,
            "pass1_completions": 400,
            "pass4_subset_problems": 100,
            "pass4_completions": 400,
            "completions_per_checkpoint_seed": 800,
        },
    }
    if protocol != expected_counts:
        raise ValueError("formal evaluation completion protocol mismatch")
    if config.get("selection_policy") != {
        "test_use": "baseline_and_final_only",
        "checkpoint_selection": "fixed_step_32_no_test_tuning",
        "validation_steps": [8, 16, 24, 32],
        "test_driven_tuning_forbidden": True,
    }:
        raise ValueError("formal test-selection policy mismatch")
    data = validate_formal_data_registry()
    return {
        "phase": phase,
        "algorithm": algorithm,
        "seed": seed,
        "cuda_initialized": False,
        "model_or_tokenizer_loads": 0,
        "generation_calls": 0,
        "trainer_calls": 0,
        "data_registry_sha256": data["registry_sha256"],
        "completion_contract": expected_counts[phase],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--phase", choices=("baseline", "validation", "final"), required=True)
    parser.add_argument("--algorithm", choices=("ppo", "grpo"))
    parser.add_argument("--seed", type=int)
    args = parser.parse_args(argv)
    result = validate_evaluation_config(
        load_evaluation_config(args.config), args.phase, algorithm=args.algorithm, seed=args.seed
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
