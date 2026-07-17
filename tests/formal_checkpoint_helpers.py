import json
from pathlib import Path
from typing import Any

from math_rlvr.training.formal_runtime import (
    FORMAL_RESUME_SCHEMA,
    write_formal_checkpoint_artifact_manifest,
)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, allow_nan=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def write_fake_trusted_checkpoint(
    root: Path,
    contract,
    step: int,
    *,
    completion_prefix: list[dict[str, Any]] | None = None,
    metric_prefix: list[dict[str, Any]] | None = None,
    optimizer_state: dict[str, Any] | None = None,
    scheduler_state: dict[str, Any] | None = None,
) -> None:
    expected_count = step * contract.completions_per_update
    if completion_prefix is None:
        completion_prefix = [
            {
                "update": index // contract.completions_per_update + 1,
                "problem_id": pair_key.rsplit("::generation:", 1)[0],
                "generation_index": int(pair_key.rsplit(":", 1)[1]),
                "pair_key": pair_key,
                "completion_ids": [7],
                "completion_mask": [1],
                "exact_token_count": 1,
                "raw_completion": "<reasoning>fake</reasoning><answer>0</answer>",
                "scalar_reward": 0.1,
            }
            for index, pair_key in enumerate(contract.pair_keys[:expected_count])
        ]
    if metric_prefix is None:
        metric_prefix = [{"update": update, "loss": 0.1} for update in range(1, step + 1)]
    tokens = sum(int(row["exact_token_count"]) for row in completion_prefix)
    root.mkdir(parents=True)
    role_files = [
        "policy_adapter/adapter_model.safetensors",
        "policy_adapter/adapter_config.json",
    ]
    if contract.algorithm == "ppo":
        role_files.extend(
            [
                "value_adapter/adapter_model.safetensors",
                "value_adapter/adapter_config.json",
                "value_head/value_head.safetensors",
                "value_head/config.json",
            ]
        )
    for relative in role_files:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fake role state\n", encoding="utf-8")
    if optimizer_state is None and scheduler_state is None:
        (root / "optimizer.pt").write_bytes(b"project-created fake optimizer state\n")
        (root / "scheduler.pt").write_bytes(b"project-created fake scheduler state\n")
    else:
        import torch

        torch.save(optimizer_state or {}, root / "optimizer.pt")
        torch.save(scheduler_state or {}, root / "scheduler.pt")
    (root / "torch_rng.safetensors").write_bytes(b"project-created fake RNG state\n")
    (root / "rng_state.json").write_text(
        json.dumps({"python": {}, "numpy": {}, "torch_cuda_rng_device_count": 0}) + "\n",
        encoding="utf-8",
    )
    (root / "trainer_state.json").write_text(
        json.dumps({"global_step": step}) + "\n", encoding="utf-8"
    )
    _write_jsonl(root / "trainer_completion_prefix.jsonl", completion_prefix)
    _write_jsonl(root / "metrics_prefix.jsonl", metric_prefix)
    online_counters = {
        "completions": expected_count,
        "generated_tokens": tokens,
        "rewards": expected_count,
        "microsteps": step * 4 if contract.algorithm == "grpo" else 0,
        "optimizer_steps": step,
        "global_steps": step,
        "updates": step,
        "loop_positions": (
            [[index, 0, 0] for index in range(step)] if contract.algorithm == "ppo" else []
        ),
    }
    resume = {
        **contract.checkpoint_identity(run_id=root.parent.name, step=step),
        "schema": FORMAL_RESUME_SCHEMA,
        "project_created": True,
        "updates": step,
        "sampler_position": {
            "comparison_key_count": expected_count,
            "ppo_episode_rows": expected_count if contract.algorithm == "ppo" else None,
            "grpo_prompt_rows": step * 4 if contract.algorithm == "grpo" else None,
        },
        "formal_runtime_counters": {
            "updates": step,
            "optimizer_steps": step,
            "global_steps": step,
            "completions": expected_count,
            "generated_tokens": tokens,
        },
        "model_roles": {"fake_state_backend": True},
        "optimizer_steps": step,
        "global_steps": step,
        "completions": expected_count,
        "generated_tokens": tokens,
        "seen_pair_keys": list(contract.pair_keys[:expected_count]),
        "checkpoints": [value for value in contract.checkpoint_steps if value <= step],
        "validations": [],
        "base_weights_included": False,
        "optimizer_state_included": True,
        "scheduler_state_included": True,
        "rng_state_included": True,
        "online_counters": online_counters,
    }
    (root / "resume_manifest.json").write_text(
        json.dumps(resume, indent=2) + "\n", encoding="utf-8"
    )
    write_formal_checkpoint_artifact_manifest(root, contract, step)
