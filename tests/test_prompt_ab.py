import hashlib
import json
import sys
from pathlib import Path

import pytest
import torch

from math_rlvr.config import load_config
from math_rlvr.evaluation.prompt_ab import (
    CONDITION_ORDER,
    DiagnosticAuthorizationError,
    GeneratedSequence,
    GenerationBudgetExceededError,
    GenerationBudgetGuard,
    completion_fields,
    main,
    matched_seed_map,
    run_diagnostic,
    select_problems,
    split_completion_ids,
)
from math_rlvr.rewards.result import RewardStatus

CONFIG = Path("configs/diagnostics/prompt_ab.yaml")


class FakeParameter:
    def __init__(self):
        self.requires_grad = True


class FakeModel:
    def __init__(self):
        self.training = True
        self.eval_calls = 0
        self.generate_calls = 0
        self.parameters = [FakeParameter(), FakeParameter()]

    def eval(self):
        self.training = False
        self.eval_calls += 1

    def generate(self, token_count):
        self.generate_calls += 1
        return list(range(token_count))


class FakeTokenizer:
    @staticmethod
    def decode(problem):
        return f"<reasoning>fake evidence only</reasoning><answer>{problem.gold_answer}</answer>"


class FakeBackend:
    backward_count = 0
    optimizer_steps = 0
    training_steps = 0
    checkpoint_writes = 0
    model_writes = 0

    def __init__(self, *, token_count=8, eos=True, safety_violation=None):
        self.model = FakeModel()
        self.tokenizer = FakeTokenizer()
        self.token_count = token_count
        self.eos = eos
        self.prepare_calls = 0
        self.generate_calls = 0
        self.closed = False
        self.eval_called = False
        self.inference_mode_used = False
        self.parameters_frozen = False
        self.current_problem = None
        self.seeds = []
        if safety_violation:
            setattr(self, safety_violation, 1)

    def prepare(self):
        self.prepare_calls += 1
        self.model.eval()
        for parameter in self.model.parameters:
            parameter.requires_grad = False
        self.eval_called = not self.model.training
        self.parameters_frozen = not any(p.requires_grad for p in self.model.parameters)

    def render(self, problem, prompt_version):
        self.current_problem = problem
        text = f"{prompt_version}:{problem.prompt}"
        return text, hashlib.sha256(text.encode()).hexdigest()

    def generate(self, prompt, *, seed, sampling, max_new_tokens):
        del prompt, sampling, max_new_tokens
        self.generate_calls += 1
        self.inference_mode_used = True
        self.seeds.append(seed)
        completion_ids = self.model.generate(self.token_count)
        text = self.tokenizer.decode(self.current_problem)
        return GeneratedSequence(
            input_token_count=5,
            completion_ids=completion_ids,
            decoded_text=text,
            eos_reached=self.eos,
        )

    def peak_vram_gib(self):
        return 0.0

    def close(self):
        self.closed = True
        return {
            "available": True,
            "device_index": 0,
            "memory_allocated": {"bytes": 0, "mib": 0.0},
            "memory_reserved": {"bytes": 0, "mib": 0.0},
            "max_memory_allocated": {"bytes": 1024, "mib": 0.0009765625},
            "max_memory_reserved": {"bytes": 2048, "mib": 0.001953125},
            "lifecycle": ["not_started", "active", "finalized"],
        }


class NonzeroCleanupFakeBackend(FakeBackend):
    def close(self):
        evidence = super().close()
        evidence["memory_allocated"] = {"bytes": 1024, "mib": 0.0009765625}
        evidence["memory_reserved"] = {"bytes": 2048, "mib": 0.001953125}
        evidence["worker_cleanup"] = {
            "current_allocated_bytes": 1024,
            "current_reserved_bytes": 2048,
            "warning": "worker_allocator_nonzero_before_process_exit",
        }
        return evidence


class FakeLifecycle:
    def __init__(self, tmp_path, *, fail_write=False, fail_backup=False):
        self.root = tmp_path
        self.root.mkdir(parents=True)
        self.fail_write = fail_write
        self.fail_backup = fail_backup
        self.backed_up = False
        self.finalized = False

    def start(self, config, problems, seed_map):
        self.persist("run_manifest.json", {"ids": [p.problem_id for p in problems]})
        self.persist("seed_map.json", seed_map)

    def persist_jsonl(self, name, rows):
        (self.root / name).write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
        )

    def persist_csv(self, name, rows):
        if not rows:
            raise RuntimeError("empty CSV")
        import csv

        with (self.root / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def persist(self, name, payload):
        if self.fail_write and name == "summary.json":
            raise OSError("artifact write failed")
        (self.root / name).write_text(json.dumps(payload, ensure_ascii=False))

    def finalize(self, summary):
        self.finalized = True
        (self.root / "checksums.sha256").write_text(summary["status"])

    def backup_and_verify(self, *, failure=False):
        self.failure_backup = failure
        if self.fail_backup:
            raise OSError("backup failed")
        self.backed_up = True

    def publish_git_safe(self):
        self.persist("published.json", {"published": True})


@pytest.fixture
def config():
    assert "math_rlvr.evaluation.prompt_ab_runtime" not in sys.modules
    return load_config(CONFIG)


def execute_fake(tmp_path, config=None, backend=None, lifecycle=None, **kwargs):
    config = config or load_config(CONFIG)
    backend = backend or FakeBackend()
    lifecycle = lifecycle or FakeLifecycle(tmp_path / "run")
    return run_diagnostic(config, backend, lifecycle, **kwargs), backend, lifecycle


def test_dry_run_and_partial_confirmations_never_generate():
    calls = []
    assert main(["--config", str(CONFIG)], execute_fn=lambda **kwargs: calls.append(kwargs)) == 0
    for args in (["--generate-only"], ["--confirm-prompt-diagnostic"]):
        with pytest.raises(DiagnosticAuthorizationError, match="both"):
            main(
                ["--config", str(CONFIG), *args],
                execute_fn=lambda **kwargs: calls.append(kwargs),
            )
    assert calls == []


def test_training_confirmation_and_training_configs_cannot_enter():
    with pytest.raises(DiagnosticAuthorizationError, match="training confirmation"):
        main(["--config", str(CONFIG), "--confirm-single-update"])
    for path in ("configs/smoke/grpo.yaml", "configs/smoke/ppo.yaml", "configs/main/grpo.yaml"):
        with pytest.raises(DiagnosticAuthorizationError):
            main(["--config", path])


def test_dual_confirmation_enters_only_injected_fake():
    calls = []

    def fake_execute(**kwargs):
        calls.append(kwargs)
        return {"status": "success"}

    assert (
        main(
            [
                "--config",
                str(CONFIG),
                "--generate-only",
                "--confirm-prompt-diagnostic",
            ],
            execute_fn=fake_execute,
            git_probe=lambda: {"branch": "pivot/math-rlvr", "commit": "fake"},
            snapshot_probe=lambda: object(),
            offline_probe=lambda: {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
        )
        == 0
    )
    assert len(calls) == 1


def test_real_confirmation_requires_offline_before_snapshot(monkeypatch):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    snapshots = []
    with pytest.raises(DiagnosticAuthorizationError, match="offline"):
        main(
            [
                "--config",
                str(CONFIG),
                "--generate-only",
                "--confirm-prompt-diagnostic",
            ],
            execute_fn=lambda **kwargs: {"status": "success"},
            git_probe=lambda: {"branch": "pivot/math-rlvr", "commit": "fake"},
            snapshot_probe=lambda: snapshots.append(True),
        )
    assert snapshots == []


def test_fake_success_exact_ab_counts_and_no_training(tmp_path, config):
    result, backend, lifecycle = execute_fake(tmp_path, config)
    assert result["status"] == "success"
    assert result["completion_count"] == 16
    assert result["budget"]["condition_counts"] == {"v0": 8, "v1": 8}
    assert result["budget"]["total_generated_tokens"] == 128
    assert result["safety_counters"] == {
        "backward_count": 0,
        "optimizer_steps": 0,
        "global_training_steps": 0,
        "checkpoint_writes": 0,
        "model_or_adapter_writes": 0,
    }
    assert backend.prepare_calls == 1 and backend.generate_calls == 16
    assert backend.model.eval_calls == 1 and backend.model.generate_calls == 16
    assert backend.eval_called and backend.inference_mode_used and backend.parameters_frozen
    assert backend.closed and lifecycle.backed_up and lifecycle.finalized


def make_guard(**overrides):
    values = {
        "max_conditions": 2,
        "unique_prompts_per_condition": 2,
        "completions_per_prompt": 4,
        "completions_per_condition": 8,
        "max_total_completions": 16,
        "max_tokens_per_completion": 128,
        "max_total_generated_tokens": 2048,
        "deadline": 120,
        "max_peak_vram_gib": 3.5,
        "clock": lambda: 0,
        "started_at": 0,
    }
    values.update(overrides)
    return GenerationBudgetGuard(**values)


def test_completion_and_token_caps_fail_closed():
    guard = make_guard()
    for index in range(16):
        guard.record(CONDITION_ORDER[index // 8], f"p{(index % 8) // 4}", [1])
    with pytest.raises(GenerationBudgetExceededError, match="completion"):
        guard.record("v1", "p1", [1])

    with pytest.raises(GenerationBudgetExceededError, match="per-completion"):
        make_guard().record("v0", "p0", [1] * 129)

    token_guard = make_guard(max_tokens_per_completion=2048)
    token_guard.record("v0", "p0", [1] * 2048)
    with pytest.raises(GenerationBudgetExceededError, match="generated-token"):
        token_guard.record("v0", "p1", [1])


def test_deadline_and_vram_gate_fail_closed():
    times = iter((121,))
    guard = make_guard(clock=lambda: next(times))
    with pytest.raises(GenerationBudgetExceededError, match="deadline"):
        guard.record("v0", "p0", [1])
    with pytest.raises(GenerationBudgetExceededError, match="VRAM"):
        make_guard().record("v0", "p0", [1], peak_vram_gib=3.51)


def test_matched_seed_map_is_identical_between_conditions(config):
    problems = select_problems(config)
    rows = matched_seed_map(config, problems)
    assert [row["condition"] for row in rows] == ["v0"] * 8 + ["v1"] * 8
    a = {(row["problem_id"], row["generation_index"]): row["seed"] for row in rows[:8]}
    b = {(row["problem_id"], row["generation_index"]): row["seed"] for row in rows[8:]}
    assert a == b
    assert list(a.values()) == list(range(42, 50))


def test_v0_rng_cannot_pollute_v1(tmp_path, config):
    result, backend, _ = execute_fake(tmp_path, config)
    assert result["status"] == "success"
    assert backend.seeds[:8] == backend.seeds[8:] == list(range(42, 50))


def test_left_padding_split_and_eos_are_exact():
    with_eos = split_completion_ids(
        [0, 0, 11, 12, 21, 22, 2, 0],
        padded_input_width=4,
        input_attention_mask=[0, 0, 1, 1],
        eos_token_id=2,
        pad_token_id=0,
    )
    assert with_eos.input_token_count == 2
    assert with_eos.completion_ids == [21, 22, 2]
    assert with_eos.eos_reached is True

    without_eos = split_completion_ids(
        [0, 11, 12, 21, 22, 0],
        padded_input_width=3,
        input_attention_mask=[0, 1, 1],
        eos_token_id=2,
        pad_token_id=0,
    )
    assert without_eos.completion_ids == [21, 22]
    assert without_eos.eos_reached is False


def test_eos_and_128_truncation_evidence(tmp_path, config):
    result, _, lifecycle = execute_fake(
        tmp_path, config, backend=FakeBackend(token_count=128, eos=False)
    )
    assert result["status"] == "success"
    rows = [
        json.loads(line) for line in (lifecycle.root / "completions.jsonl").read_text().splitlines()
    ]
    assert all(row["truncated_at_128"] and not row["eos_reached"] for row in rows)


def test_worker_allocator_nonzero_is_warning_not_runtime_failure(tmp_path, config):
    result, backend, lifecycle = execute_fake(
        tmp_path, config, backend=NonzeroCleanupFakeBackend()
    )
    assert result["status"] == "success"
    assert result["warnings"] == ["worker_allocator_nonzero_before_process_exit"]
    assert backend.closed
    allocator = json.loads((lifecycle.root / "pytorch_allocator.json").read_text())
    assert allocator["worker_cleanup"]["current_allocated_bytes"] == 1024
    assert allocator["worker_cleanup"]["current_reserved_bytes"] == 2048


@pytest.mark.parametrize(
    "counter",
    ["backward_count", "optimizer_steps", "training_steps", "checkpoint_writes", "model_writes"],
)
def test_any_training_or_checkpoint_side_effect_fails(tmp_path, counter):
    result, backend, _ = execute_fake(tmp_path, backend=FakeBackend(safety_violation=counter))
    assert result["status"] == "failure"
    assert "safety counter" in result["reason"]
    assert backend.closed


def test_eval_inference_and_frozen_contract_fail_closed(tmp_path):
    backend = FakeBackend()
    backend.prepare = lambda: None
    result, _, _ = execute_fake(tmp_path, backend=backend)
    assert result["status"] == "failure"
    assert "eval/inference/frozen" in result["reason"]


def test_parser_verifier_order_and_completion_order(tmp_path, config):
    result, _, lifecycle = execute_fake(tmp_path, config)
    rows = [
        json.loads(line) for line in (lifecycle.root / "completions.jsonl").read_text().splitlines()
    ]
    assert result["status"] == "success"
    assert [(row["condition"], row["generation_index"]) for row in rows[:4]] == [
        ("v0", 0),
        ("v0", 1),
        ("v0", 2),
        ("v0", 3),
    ]
    assert all(row["parser_status"] == "parsed" for row in rows)
    assert all(row["reward_status"] != RewardStatus.INFRA_ERROR.value for row in rows)


def test_infra_error_and_nan_reward_abort(tmp_path):
    def infra(problem, text):
        del problem, text
        raise RuntimeError("infra_error: injected")

    result, _, _ = execute_fake(tmp_path / "infra", completion_analyzer=infra)
    assert result["status"] == "failure" and "infra_error" in result["reason"]

    def nan(problem, text):
        fields, reward, scalar = completion_fields(problem, text)
        fields["scalar_reward"] = float("nan")
        return fields, reward, scalar

    result, _, _ = execute_fake(tmp_path / "nan", completion_analyzer=nan)
    assert result["status"] == "failure" and "non-finite" in result["reason"]


def test_artifact_and_backup_failure_never_succeed(tmp_path):
    for fail_write, fail_backup in ((True, False), (False, True)):
        lifecycle = FakeLifecycle(
            tmp_path / f"run-{fail_write}-{fail_backup}",
            fail_write=fail_write,
            fail_backup=fail_backup,
        )
        result, backend, _ = execute_fake(tmp_path, lifecycle=lifecycle)
        assert result["status"] == "failure"
        assert result["backed_up"] == (not fail_backup)
        assert backend.closed


def test_candidate_qualification_never_auto_activates(tmp_path):
    result, _, _ = execute_fake(tmp_path)
    assert result["status"] == "success"
    assert result["candidate_qualification"]["auto_activate"] is False


def test_frozen_yaml_and_history_hashes_unchanged():
    expected = {
        "configs/smoke/grpo.yaml": "068ff8d742849ffa0d43ccf6f4e74898e08c5f031c0f837c18ac8e5b183d8979",  # noqa: E501
        "configs/smoke/ppo.yaml": "1496c65309befbcf4c5143b5d19e963013a9c869ff4af4e82b838abc317a0379",  # noqa: E501
        "reports/runs/grpo_single_update_qwen25_05b_20260713T063829Z/summary.json": "39c40a0f87ebd069a7f0757e7bba3cac8ae58e93549f41bd681c7ba02e1b6e09",  # noqa: E501
        "reports/runs/grpo_single_update_qwen25_05b_20260713T063829Z/completions.jsonl": "bd0df2cfb0ed0ad85f75393213eef632c3b65508b35e32f4657102a56bff25bd",  # noqa: E501
    }
    for name, digest in expected.items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest
    assert torch.cuda.is_initialized() is False
