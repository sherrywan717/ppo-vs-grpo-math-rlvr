from pathlib import Path

import pytest

from code_rlvr.config import load_config, validate_runtime_path, validate_training_config


@pytest.mark.parametrize("path,algorithm", [
    ("configs/smoke/ppo.yaml", "ppo"),
    ("configs/smoke/grpo.yaml", "grpo"),
    ("configs/main/ppo.yaml", "ppo"),
    ("configs/main/grpo.yaml", "grpo"),
])
def test_training_configs_are_bounded(path: str, algorithm: str) -> None:
    validate_training_config(load_config(path), algorithm)


def test_runtime_paths_stay_on_temp_disk() -> None:
    assert validate_runtime_path("/root/autodl-tmp/code-rlvr-outputs").is_absolute()
    with pytest.raises(ValueError):
        validate_runtime_path(Path.home() / ".cache" / "models")

