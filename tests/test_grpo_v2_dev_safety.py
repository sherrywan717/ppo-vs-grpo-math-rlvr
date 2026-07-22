import hashlib
from pathlib import Path

import pytest

from math_rlvr.evaluation.grpo_v2_dev_runtime import (
    DevEvaluationContractError,
    require_finite_logits,
    validate_inference_contract,
)
from math_rlvr.evaluation.grpo_v2_dev_supervisor import (
    REQUIRED_SUCCESS_FILES,
    validate_backup_result,
    validate_success_artifacts,
)
from math_rlvr.evaluation.prompt_ab_supervisor import verify_post_worker_exit


def test_inference_role_and_forbidden_side_effect_contract():
    evidence = validate_inference_contract(
        model_training=False,
        parameter_requires_grad=[False, False],
        inference_mode_used=True,
    )
    assert evidence["parameters_require_grad"] == 0
    assert evidence["train_calls"] == evidence["optimizer_steps"] == 0
    cases = (
        {"model_training": True},
        {"parameter_requires_grad": [True]},
        {"inference_mode_used": False},
        {"train_calls": 1},
        {"backward_calls": 1},
        {"optimizer_steps": 1},
        {"checkpoint_writes": 1},
    )
    for update in cases:
        values = {
            "model_training": False,
            "parameter_requires_grad": [False],
            "inference_mode_used": True,
            **update,
        }
        with pytest.raises(DevEvaluationContractError):
            validate_inference_contract(**values)


def test_nonfinite_logits_fail_closed():
    require_finite_logits(True)
    with pytest.raises(DevEvaluationContractError, match="NaN/Inf logits"):
        require_finite_logits(False)


def test_success_artifact_and_backup_fail_closed(tmp_path: Path):
    with pytest.raises(RuntimeError, match="artifacts missing"):
        validate_success_artifacts(tmp_path)
    for name in REQUIRED_SUCCESS_FILES:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("evidence\n")
    validate_success_artifacts(tmp_path)
    archive = tmp_path / "run.tar.gz"
    archive.write_bytes(b"archive")
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    validate_backup_result({"archive": str(archive), "sha256": digest})
    with pytest.raises(RuntimeError, match="backup verification"):
        validate_backup_result({"archive": str(archive), "sha256": "0" * 64})


def test_post_worker_gpu_release_contract():
    evidence = verify_post_worker_exit(
        worker_pid=123,
        baseline={"memory_used_mib": {"0": 0}, "compute_pids": []},
        current={"memory_used_mib": {"0": 0}, "compute_pids": []},
        pid_exists=lambda pid: False,
    )
    assert evidence["gpu_memory_restored_to_baseline"] is True
    with pytest.raises(Exception, match="post-worker GPU verification failed"):
        verify_post_worker_exit(
            worker_pid=123,
            baseline={"memory_used_mib": {"0": 0}, "compute_pids": []},
            current={"memory_used_mib": {"0": 1}, "compute_pids": [123]},
            pid_exists=lambda pid: True,
        )
