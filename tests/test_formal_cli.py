from pathlib import Path

import pytest
from formal_checkpoint_helpers import write_fake_trusted_checkpoint

from math_rlvr.evaluation.formal import main as evaluation_main
from math_rlvr.evaluation.formal_cli import validate_formal_evaluation_selection
from math_rlvr.training.execution_contract import expected_run_contract_for_config
from math_rlvr.training.formal import validate_formal_config_file
from math_rlvr.training.formal_cli import (
    FormalAuthorizationError,
    require_formal_local_snapshot,
    validate_formal_training_authorization,
)
from math_rlvr.training.formal_model_runtime import audit_grpo_parameter_roles
from math_rlvr.training.formal_runtime import (
    FormalOnlineGuard,
    FormalRuntimeError,
    formal_run_contract,
    prepare_formal_runtime_prompt_preflight,
)
from math_rlvr.training.grpo import main as grpo_main
from math_rlvr.training.model_source import SnapshotValidationError
from math_rlvr.training.ppo import main as ppo_main

ACTIVE = (
    ("ppo", 42),
    ("grpo", 42),
    ("grpo", 123),
    ("ppo", 123),
)


def config_path(algorithm, seed):
    return Path(f"configs/formal_1p5b/resolved/{algorithm}_seed_{seed}.json")


@pytest.mark.parametrize(("algorithm", "seed"), ACTIVE)
def test_active_formal_scope_expected_contract_and_prompt_preflight_agree(algorithm, seed):
    path = config_path(algorithm, seed)
    config = validate_formal_config_file(path, algorithm)[0]
    authorization = validate_formal_training_authorization(config, path, algorithm)
    expected = expected_run_contract_for_config(config, algorithm)
    runtime = formal_run_contract(config)
    prompt = prepare_formal_runtime_prompt_preflight(config, algorithm)
    assert authorization["expected_run_profile"] == expected.profile == runtime.profile
    assert authorization["active_suite_sha256"] == runtime.active_suite_sha256
    assert expected.expected_updates == 32
    assert expected.expected_completions == 512
    assert expected.generated_token_cap == 131_072
    assert prompt["rendered_row_count"] == (512 if algorithm == "ppo" else 128)
    assert prompt["cpu_resolved_scope"] == prompt["prompt_selector_scope"] == "main_formal"
    assert len(prompt["comparison_keys"]) == 512


@pytest.mark.parametrize(("main", "algorithm"), ((ppo_main, "ppo"), (grpo_main, "grpo")))
def test_formal_dry_run_and_missing_confirmation_touch_no_paid_boundary(main, algorithm):
    path = config_path(algorithm, 42)
    calls = []

    def forbidden():
        calls.append("called")
        raise AssertionError("paid boundary was touched")

    assert main(["--config", str(path)], git_probe=forbidden, snapshot_probe=forbidden) == 0
    with pytest.raises(RuntimeError, match="requires --execute"):
        main(
            ["--config", str(path), "--execute"],
            git_probe=forbidden,
            snapshot_probe=forbidden,
        )
    assert calls == []




@pytest.mark.parametrize(("main", "algorithm"), ((ppo_main, "ppo"), (grpo_main, "grpo")))
def test_missing_formal_resume_checkpoint_fails_before_paid_boundary(main, algorithm, tmp_path):
    confirmation = "--confirm-formal-ppo" if algorithm == "ppo" else "--confirm-formal-grpo"
    calls = []

    def forbidden():
        calls.append("called")
        raise AssertionError("paid boundary was touched")

    with pytest.raises(FormalAuthorizationError, match="does not exist"):
        main(
            [
                "--config",
                str(config_path(algorithm, 42)),
                "--execute",
                confirmation,
                "--resume-checkpoint",
                str(tmp_path / "checkpoint-8"),
            ],
            git_probe=forbidden,
            offline_probe=forbidden,
            snapshot_probe=forbidden,
        )
    assert calls == []
@pytest.mark.parametrize(("algorithm", "seed"), ACTIVE)
def test_each_active_training_cli_dispatches_only_after_dual_confirmation(algorithm, seed):
    path = config_path(algorithm, seed)
    captured = {}

    def execute(config, **kwargs):
        captured.update(kwargs)
        captured["config"] = config
        return {"status": "success"}

    main = ppo_main if algorithm == "ppo" else grpo_main
    confirmation = "--confirm-formal-ppo" if algorithm == "ppo" else "--confirm-formal-grpo"
    assert (
        main(
            ["--config", str(path), "--execute", confirmation],
            execute_fn=execute,
            git_probe=lambda: {"clean": True},
            offline_probe=lambda: {"offline": True},
            snapshot_probe=lambda: "fake-validated-source",
        )
        == 0
    )
    assert captured["authorization"]["algorithm"] == algorithm
    assert captured["authorization"]["seed"] == seed
    assert captured["model_source"] == "fake-validated-source"
    assert captured["prompt_preflight"]["expected_run_profile"] == f"{algorithm}_formal_1p5b"


@pytest.mark.parametrize("algorithm", ("ppo", "grpo"))
def test_reserved_2026_and_absolute_alias_are_rejected_before_snapshot(algorithm):
    path = config_path(algorithm, 2026)
    main = ppo_main if algorithm == "ppo" else grpo_main
    confirmation = "--confirm-formal-ppo" if algorithm == "ppo" else "--confirm-formal-grpo"
    snapshot_calls = []
    with pytest.raises(FormalAuthorizationError, match="reserved_not_scheduled"):
        main(
            ["--config", str(path), "--execute", confirmation],
            git_probe=lambda: {},
            offline_probe=lambda: {},
            snapshot_probe=lambda: snapshot_calls.append(True),
        )
    active = config_path(algorithm, 42).resolve()
    with pytest.raises(FormalAuthorizationError, match="repository-relative"):
        main(
            ["--config", str(active), "--execute", confirmation],
            git_probe=lambda: {},
            offline_probe=lambda: {},
            snapshot_probe=lambda: snapshot_calls.append(True),
        )
    assert snapshot_calls == []


def test_missing_formal_snapshot_fails_before_execute(tmp_path):
    cache = tmp_path / "missing-cache"
    cache.mkdir()
    with pytest.raises(SnapshotValidationError, match="does not exist"):
        require_formal_local_snapshot(cache_root=cache)

    executed = []
    with pytest.raises(SnapshotValidationError, match="does not exist"):
        ppo_main(
            [
                "--config",
                str(config_path("ppo", 42)),
                "--execute",
                "--confirm-formal-ppo",
            ],
            execute_fn=lambda *_args, **_kwargs: executed.append(True),
            git_probe=lambda: {},
            offline_probe=lambda: {},
            snapshot_probe=lambda: require_formal_local_snapshot(cache_root=cache),
        )
    assert executed == []


def test_online_token_cap_accepts_exact_and_rejects_one_more():
    config = validate_formal_config_file(config_path("grpo", 42), "grpo")[0]
    contract = formal_run_contract(config)
    exact = FormalOnlineGuard(contract)
    for update in range(32):
        exact.record_generation(16, 4096)
        for _ in range(16):
            exact.record_reward(None, 0.1, {})
        for _ in range(4):
            exact.record_microstep()
        exact.record_optimizer_step()
        exact.record_global_step(update + 1)
        exact.record_update()
    assert exact.assert_complete()["generated_tokens"] == 131_072

    overflow = FormalOnlineGuard(contract)
    with pytest.raises(FormalRuntimeError, match="token cap"):
        overflow.record_generation(16, 131_073)


class FakeParameter:
    def __init__(self, size=1, requires_grad=True):
        self.requires_grad = requires_grad
        self.size = size

    def numel(self):
        return self.size


class FakeModel:
    def __init__(self, named):
        self.named = named

    def named_parameters(self):
        return list(self.named)

    def parameters(self):
        return [parameter for _, parameter in self.named]


class FakeOptimizer:
    def __init__(self, parameters):
        self.param_groups = [{"params": list(parameters)}]


def test_grpo_optimizer_role_is_policy_lora_only():
    lora = FakeParameter(7)
    frozen = FakeParameter(11, requires_grad=False)
    policy = FakeModel((("base.q_proj.lora_A.default.weight", lora), ("base.weight", frozen)))
    evidence = audit_grpo_parameter_roles(policy, optimizer=FakeOptimizer([lora]))
    assert evidence["optimizer_exact_role_match"] is True
    assert evidence["policy_trainable_parameters"] == 7
    with pytest.raises(FormalRuntimeError, match="exact policy-LoRA"):
        audit_grpo_parameter_roles(policy, optimizer=FakeOptimizer([lora, frozen]))


def write_checkpoint(root, algorithm, seed, step=32):
    config = validate_formal_config_file(config_path(algorithm, seed), algorithm)[0]
    contract = formal_run_contract(config)
    write_fake_trusted_checkpoint(root, contract, step)


def test_evaluation_base_and_policy_only_modes(tmp_path):
    base = validate_formal_evaluation_selection(
        config_path=Path("configs/formal_1p5b/evaluation.json"),
        mode="base",
        phase="baseline",
        seed=42,
        checkpoint_step=None,
        checkpoint=None,
    )
    assert base.policy_adapter is None
    for algorithm in ("ppo", "grpo"):
        checkpoint = tmp_path / algorithm / "checkpoint-32"
        write_checkpoint(checkpoint, algorithm, 42)
        selected = validate_formal_evaluation_selection(
            config_path=Path("configs/formal_1p5b/evaluation.json"),
            mode=algorithm,
            phase="final",
            seed=42,
            checkpoint_step=32,
            checkpoint=checkpoint,
        )
        assert selected.policy_adapter == (checkpoint / "policy_adapter").resolve()
        evidence = selected.to_dict()
        assert evidence["value_adapter_loaded_for_generation"] is False
        assert evidence["value_head_loaded_for_generation"] is False


def test_evaluation_confirmation_precedes_snapshot_and_base_dispatches():
    calls = []
    argv = [
        "--config",
        "configs/formal_1p5b/evaluation.json",
        "--phase",
        "baseline",
        "--seed",
        "42",
        "--mode",
        "base",
        "--execute",
    ]
    with pytest.raises(RuntimeError, match="confirm-formal-evaluation"):
        evaluation_main(argv, snapshot_probe=lambda: calls.append("snapshot"))
    assert calls == []

    captured = {}
    assert (
        evaluation_main(
            [*argv, "--confirm-formal-evaluation"],
            snapshot_probe=lambda: "fake-source",
            offline_probe=lambda: {},
            execute_fn=lambda **kwargs: captured.update(kwargs) or {"status": "success"},
            git_probe=lambda: {},
        )
        == 0
    )
    assert captured["selection"].mode == "base"
    assert captured["selection"].policy_adapter is None
    assert captured["model_source"] == "fake-source"
