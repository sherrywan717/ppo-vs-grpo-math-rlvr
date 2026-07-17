import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from formal_checkpoint_helpers import write_fake_trusted_checkpoint

from math_rlvr.training.formal import validate_formal_config_file
from math_rlvr.training.formal_model_runtime import _restore_trusted_training_state
from math_rlvr.training.formal_runtime import (
    FormalRuntimeError,
    formal_checkpoint_inventory,
    formal_run_contract,
    validate_formal_resume_checkpoint,
)


def _contract(algorithm: str):
    path = Path(f"configs/formal_1p5b/resolved/{algorithm}_seed_42.json")
    return formal_run_contract(validate_formal_config_file(path, algorithm)[0])


def _checkpoint(tmp_path: Path, algorithm: str, step: int = 16):
    run_dir = tmp_path / f"{algorithm}_formal_1p5b_seed42_fake"
    root = run_dir / f"checkpoint-{step}"
    contract = _contract(algorithm)
    write_fake_trusted_checkpoint(root, contract, step)
    return contract, run_dir, root


@pytest.mark.parametrize("algorithm", ("ppo", "grpo"))
def test_resume_checkpoint_accepts_only_same_run_and_allowed_steps(tmp_path, algorithm):
    contract, run_dir, root = _checkpoint(tmp_path, algorithm)
    validated = validate_formal_resume_checkpoint(root, contract, run_dir=run_dir)
    assert validated.step == 16
    assert validated.manifest["run_id"] == run_dir.name

    other_run = tmp_path / "other-run"
    other_run.mkdir()
    with pytest.raises(FormalRuntimeError, match="same run"):
        validate_formal_resume_checkpoint(root, contract, run_dir=other_run)

    final = run_dir / "checkpoint-32"
    write_fake_trusted_checkpoint(final, contract, 32)
    with pytest.raises(FormalRuntimeError, match="8/16/24"):
        validate_formal_resume_checkpoint(final, contract, run_dir=run_dir)


@pytest.mark.parametrize("algorithm", ("ppo", "grpo"))
def test_resume_checkpoint_rejects_sha_tamper_before_state_load(tmp_path, algorithm):
    contract, _run_dir, root = _checkpoint(tmp_path, algorithm)
    with (root / "optimizer.pt").open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(FormalRuntimeError, match="SHA256 inventory mismatch"):
        formal_checkpoint_inventory(root, contract, 16)


@pytest.mark.parametrize("algorithm", ("ppo", "grpo"))
def test_resume_checkpoint_rejects_full_base_weight_and_unknown_files(tmp_path, algorithm):
    contract, _run_dir, root = _checkpoint(tmp_path, algorithm)
    (root / "model.safetensors").write_bytes(b"forbidden full base-model placeholder")
    with pytest.raises(FormalRuntimeError, match="unexpected formal checkpoint file"):
        formal_checkpoint_inventory(root, contract, 16)


def test_resume_checkpoint_rejects_identity_tamper(tmp_path):
    contract, _run_dir, root = _checkpoint(tmp_path, "ppo")
    path = root / "artifact_manifest.json"
    artifact = json.loads(path.read_text(encoding="utf-8"))
    artifact["checkpoint_identity"]["config_sha256"] = "0" * 64
    path.write_text(json.dumps(artifact) + "\n", encoding="utf-8")
    with pytest.raises(FormalRuntimeError, match="manifest identity mismatch"):
        formal_checkpoint_inventory(root, contract, 16)


def _advance_fake_state(state, first_step: int, last_step: int):
    import torch

    parameter = state["parameter"].clone()
    momentum = state["momentum"].clone()
    for step in range(first_step, last_step + 1):
        gradient = torch.tensor((step % 7 + 1) / 100, dtype=torch.float64)
        momentum = momentum * 0.875 + gradient
        parameter = parameter - momentum * 0.03125
    return {"parameter": parameter, "momentum": momentum}


class _StateTarget:
    def __init__(self):
        self.state = None

    def load_state_dict(self, state):
        self.state = state


@pytest.mark.parametrize("algorithm", ("ppo", "grpo"))
def test_fake_continuous_32_equals_save_16_resume_to_32(tmp_path, algorithm):
    """CPU float64 is bit-exact here; real GPU kernels may require documented tolerance."""
    import torch

    initial = {
        "parameter": torch.tensor([0.5, -0.25], dtype=torch.float64),
        "momentum": torch.tensor([0.0, 0.0], dtype=torch.float64),
    }
    continuous = _advance_fake_state(initial, 1, 32)
    interrupted = _advance_fake_state(initial, 1, 16)
    contract = _contract(algorithm)
    run_dir = tmp_path / f"{algorithm}_formal_1p5b_seed42_fake"
    root = run_dir / "checkpoint-16"
    write_fake_trusted_checkpoint(
        root,
        contract,
        16,
        optimizer_state=interrupted,
        scheduler_state={"last_epoch": 16},
    )
    validated = validate_formal_resume_checkpoint(root, contract, run_dir=run_dir)
    trainer = SimpleNamespace(optimizer=_StateTarget(), lr_scheduler=_StateTarget())
    _restore_trusted_training_state(validated, trainer)
    assert trainer.lr_scheduler.state == {"last_epoch": 16}
    resumed = _advance_fake_state(trainer.optimizer.state, 17, 32)
    assert torch.equal(resumed["parameter"], continuous["parameter"])
    assert torch.equal(resumed["momentum"], continuous["momentum"])
