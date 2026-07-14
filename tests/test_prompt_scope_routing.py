import copy
import hashlib
import inspect
import json
from pathlib import Path

import pytest
import torch

from math_rlvr.config import load_config
from math_rlvr.dataset import load_manifest
from math_rlvr.prompt import (
    PROMPT_V0_GRPO_SMOKE,
    PROMPT_V0_SHA256,
    PROMPT_V1_SHA256,
    PROMPT_V1_STRICT_CONCISE,
    ExperimentScope,
    format_problem,
    format_training_problem,
    prompt_spec_sha256,
    prompt_version_from_config,
)
from math_rlvr.training.common import preflight
from math_rlvr.training.execution_contract import (
    expected_run_contract_for_config,
    validated_experiment_scope,
    validated_scope_from_config,
)
from math_rlvr.training.grpo_runtime import (
    RealBackend,
    build_grpo_runtime_dataset_rows,
    execute_real_grpo,
)
from math_rlvr.training.guarded_grpo import select_grpo_execution_problems
from math_rlvr.training.guarded_ppo import ppo_execution_problems_and_episodes
from math_rlvr.training.pilot import PILOT_SEEDS, pilot_pair_keys
from math_rlvr.training.ppo_runtime import (
    RealPPOBackend,
    build_ppo_runtime_dataset_rows,
    execute_real_ppo,
)
from math_rlvr.training.runtime_prompt_scope import prepare_runtime_prompt_preflight


class FakeTokenizer:
    pad_token_id = 0
    pad_token = "<pad>"
    eos_token = "<eos>"
    padding_side = "right"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        assert tokenize is False
        assert add_generation_prompt is True
        return json.dumps(messages, ensure_ascii=False, sort_keys=True) + "<assistant>"

    def __call__(self, text, add_special_tokens=False, truncation=False):
        assert add_special_tokens is False
        assert truncation is False
        return {"input_ids": [ord(char) % 251 for char in text[::4]]}


@pytest.mark.parametrize("seed", PILOT_SEEDS)
def test_ppo_delayed_runtime_builder_uses_validated_pilot_scope(seed, monkeypatch):
    import math_rlvr.training.builders as builders

    model_loads = []
    monkeypatch.setattr(
        builders,
        "load_policy_and_tokenizer",
        lambda *args, **kwargs: model_loads.append((args, kwargs)),
    )
    config = preflight(Path(f"configs/pilot/resolved/ppo_seed_{seed}.json"), "ppo")
    evidence = prepare_runtime_prompt_preflight(config, "ppo")
    assert {
        evidence["cpu_resolved_scope"],
        evidence["expected_run_contract_scope"],
        evidence["delayed_runtime_scope"],
        evidence["prompt_selector_scope"],
    } == {ExperimentScope.MATCHED_0P5B_PILOT.value}
    assert evidence["rendered_row_count"] == 16
    assert evidence["comparison_keys"] == pilot_pair_keys()

    scope = validated_scope_from_config(config, "ppo")
    contract = expected_run_contract_for_config(config, "ppo")
    problems, episodes = ppo_execution_problems_and_episodes(config, contract)
    prompt_lookup, rows = build_ppo_runtime_dataset_rows(
        config, FakeTokenizer(), problems, episodes, scope
    )
    assert len(rows) == 16
    assert len(prompt_lookup) == 4
    assert [row["pair_key"] for row in rows] == pilot_pair_keys()
    assert [row["episode_position"] for row in rows] == list(range(16))
    for problem_index in range(4):
        group = rows[problem_index * 4 : (problem_index + 1) * 4]
        assert {row["problem_id"] for row in group} == {f"countdown:train:{problem_index}"}
        assert [row["generation_index"] for row in group] == [0, 1, 2, 3]
        assert len({row["rendered_prompt_hash"] for row in group}) == 1
    assert model_loads == []
    assert torch.cuda.is_initialized() is False


def test_ppo_seed42_exact_failed_path_no_longer_hits_main_formal_rejection():
    config = preflight(Path("configs/pilot/resolved/ppo_seed_42.json"), "ppo")
    scope = validated_scope_from_config(config, "ppo")
    contract = expected_run_contract_for_config(config, "ppo")
    problems, episodes = ppo_execution_problems_and_episodes(config, contract)
    try:
        _lookup, rows = build_ppo_runtime_dataset_rows(
            config, FakeTokenizer(), problems, episodes, scope
        )
    except ValueError as exc:
        assert "main/formal configs must not activate a smoke prompt" not in str(exc)
        raise
    assert len(rows) == 16


@pytest.mark.parametrize("seed", PILOT_SEEDS)
def test_grpo_delayed_runtime_builder_uses_validated_pilot_scope(seed, monkeypatch):
    import math_rlvr.training.builders as builders

    model_loads = []
    monkeypatch.setattr(
        builders,
        "load_policy_and_tokenizer",
        lambda *args, **kwargs: model_loads.append((args, kwargs)),
    )
    config = preflight(Path(f"configs/pilot/resolved/grpo_seed_{seed}.json"), "grpo")
    evidence = prepare_runtime_prompt_preflight(config, "grpo")
    assert {
        evidence["cpu_resolved_scope"],
        evidence["expected_run_contract_scope"],
        evidence["delayed_runtime_scope"],
        evidence["prompt_selector_scope"],
    } == {ExperimentScope.MATCHED_0P5B_PILOT.value}
    assert evidence["rendered_row_count"] == 4
    assert evidence["comparison_keys"] == pilot_pair_keys()

    scope = validated_scope_from_config(config, "grpo")
    contract = expected_run_contract_for_config(config, "grpo")
    problems = select_grpo_execution_problems(config, contract)
    rows = build_grpo_runtime_dataset_rows(config, problems, scope)
    assert len(rows) == 4
    assert [row["problem_id"] for row in rows] == [
        f"countdown:train:{index}" for index in range(4)
    ]
    assert [row["rendered_prompt_hash"] for row in rows] == [
        evidence["rows"][index]["rendered_prompt_hash"] for index in range(4)
    ]
    assert len(contract.pair_keys) == len(set(contract.pair_keys)) == 16
    assert model_loads == []
    assert torch.cuda.is_initialized() is False


def test_model_load_is_ordered_after_cpu_prompt_preflight_in_both_runtimes():
    ppo_execute = inspect.getsource(execute_real_ppo)
    grpo_execute = inspect.getsource(execute_real_grpo)
    assert ppo_execute.index("prepare_runtime_prompt_preflight") < ppo_execute.index(
        "require_local_snapshot"
    )
    assert grpo_execute.index("prepare_runtime_prompt_preflight") < grpo_execute.index(
        "require_local_snapshot"
    )
    ppo_backend = inspect.getsource(RealPPOBackend.run)
    grpo_backend = inspect.getsource(RealBackend.run)
    assert ppo_backend.index("scope = validate_runtime_prompt_preflight") < ppo_backend.index(
        "policy, tokenizer = load_policy_and_tokenizer"
    )
    assert grpo_backend.index("scope = validate_runtime_prompt_preflight") < grpo_backend.index(
        "model, tokenizer = load_policy_and_tokenizer"
    )


@pytest.mark.parametrize(
    ("path", "algorithm"),
    [
        ("configs/smoke/ppo.yaml", "ppo"),
        ("configs/smoke/grpo.yaml", "grpo"),
    ],
)
def test_stage_d_scope_still_allows_v1(path, algorithm):
    config = preflight(Path(path), algorithm)
    scope = validated_scope_from_config(config, algorithm)
    assert scope.scope is ExperimentScope.STAGE_D_SMOKE
    assert prompt_version_from_config(config, scope.scope) == PROMPT_V1_STRICT_CONCISE
    problem = load_manifest(Path(config["data"]["manifest"]))[0]
    assert format_training_problem(problem, config, scope=scope.scope)


@pytest.mark.parametrize(
    ("path", "algorithm"),
    [
        ("configs/main/ppo.yaml", "ppo"),
        ("configs/main/grpo.yaml", "grpo"),
    ],
)
def test_main_formal_name_spoof_cannot_activate_smoke_v1(path, algorithm):
    config = preflight(Path(path), algorithm)
    scope = validated_scope_from_config(config, algorithm)
    assert scope.scope is ExperimentScope.MAIN_FORMAL
    spoof = copy.deepcopy(config)
    spoof["experiment"]["name"] = "pilot-forged-name"
    spoof["prompt"] = {"version": PROMPT_V1_STRICT_CONCISE}
    with pytest.raises(ValueError, match="main/formal configs"):
        prompt_version_from_config(spoof, scope.scope)


def test_unknown_path_hash_scope_and_plain_string_scope_fail_closed(tmp_path, monkeypatch):
    forged = tmp_path / "ppo_seed_42.json"
    forged.write_bytes(Path("configs/pilot/resolved/ppo_seed_42.json").read_bytes())
    with pytest.raises(ValueError, match="inside the repository|no validated"):
        validated_experiment_scope(forged, "ppo")

    config = load_config("configs/main/ppo.yaml")
    config["experiment"]["name"] = "pilot-forged-name"
    config["prompt"] = {"version": PROMPT_V1_STRICT_CONCISE}
    main_scope = validated_experiment_scope(Path("configs/main/ppo.yaml"), "ppo")
    with pytest.raises(ValueError, match="main/formal"):
        prompt_version_from_config(config, main_scope.scope)
    with pytest.raises(ValueError, match="validated experiment scope"):
        prompt_version_from_config(config, "matched_0p5b_pilot")

    import math_rlvr.training.execution_contract as module

    algorithm, digest = module._MAIN_FORMAL_CONFIG_SHA256["configs/main/ppo.yaml"]
    monkeypatch.setitem(
        module._MAIN_FORMAL_CONFIG_SHA256,
        "configs/main/ppo.yaml",
        (algorithm, "0" * 64),
    )
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        validated_experiment_scope(Path("configs/main/ppo.yaml"), "ppo")
    assert digest == "1ced44a672fa3a5dcf9871bd8c1893a3bdad641d756dcf9de226b20440d1ad74"


def test_serialized_scope_tampering_is_rejected():
    config = preflight(Path("configs/pilot/resolved/ppo_seed_42.json"), "ppo")
    forged = copy.deepcopy(config)
    forged["validated_experiment_scope"]["scope"] = ExperimentScope.MAIN_FORMAL.value
    with pytest.raises(ValueError, match="serialized experiment scope"):
        validated_scope_from_config(forged, "ppo")


def test_historical_v0_replay_and_prompt_identity_are_unchanged():
    assert prompt_spec_sha256(PROMPT_V0_GRPO_SMOKE) == PROMPT_V0_SHA256
    assert PROMPT_V0_SHA256 == "20b54a2ae00ebc762a1a90a3221f5c2409c7e64d2b35fcf2c6dfaaff48a9ef4f"
    assert PROMPT_V1_SHA256 == "6842002e4591630a4105a4ca8fdf4cab91676b3902708ec0cb7f7b458864ecd7"
    problem = load_manifest(
        Path("/root/autodl-tmp/datasets/math_rlvr/manifests/countdown_train.json")
    )[0]
    assert format_problem(problem)


def test_failed_seed42_run_and_git_safe_evidence_remain_immutable():
    expected = {
        Path(
            "reports/runs/ppo_matched_0p5b_seed42_20260714T073357Z/failure_report.json"
        ): "9240b4d13b649ded6e27360965870705763cea09d8098743dd113c27b2e6d4d2",
        Path(
            "reports/runs/ppo_matched_0p5b_seed42_20260714T073357Z/summary.json"
        ): "80cb79dd1e6341e11b6bb06aceff376c400cdf197ccbe4094bae0c53e20a63d7",
        Path(
            "reports/runs/ppo_matched_0p5b_seed42_20260714T073357Z/launcher_output_capture.txt"
        ): "05a4546894fc55cd6fd430c8f21d1b9df1e041db735301a3afff92446655bd44",
        Path(
            "reports/runs/ppo_matched_0p5b_seed42_20260714T073357Z/assessment.md"
        ): "c73ff974c57d60d88c558a5458d597ddcbfeedddabd96a6b9778c28c60953298",
        Path(
            "/root/autodl-tmp/runs/math_rlvr/"
            "ppo_matched_0p5b_seed42_20260714T073357Z/final_summary.json"
        ): "c729f58634e97bd90f3c00f48f8366a55fe21cb13e7b17c73544db0d79241607",
    }
    for path, digest in expected.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    failure = json.loads(
        Path(
            "reports/runs/ppo_matched_0p5b_seed42_20260714T073357Z/failure_report.json"
        ).read_text()
    )
    assert failure["status"] == "failure"
    assert failure["counters"]["completions"] == failure["counters"]["generated_tokens"] == 0
    assert (
        failure["counters"]["updates"]
        == failure["counters"]["optimizer_steps"]
        == failure["counters"]["global_step"]
        == 0
    )
