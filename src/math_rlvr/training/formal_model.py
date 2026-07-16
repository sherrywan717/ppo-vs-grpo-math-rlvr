"""Static 1.5B model-role and trainable-parameter contract; never loads a model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_IDENTITY_PATH = REPOSITORY_ROOT / "configs/formal_1p5b/model_identity.json"


def derive_static_parameter_contract(
    path: Path = DEFAULT_MODEL_IDENTITY_PATH,
) -> dict[str, Any]:
    identity = json.loads(path.read_text(encoding="utf-8"))
    hidden = identity["hidden_size"]
    heads = identity["num_attention_heads"]
    kv_heads = identity["num_key_value_heads"]
    layers = identity["num_hidden_layers"]
    if hidden % heads:
        raise ValueError("formal model head dimension is not integral")
    kv_width = (hidden // heads) * kv_heads

    def lora_parameters(rank: int, output_width: int) -> int:
        return rank * (hidden + output_width)

    policy_per_layer = (
        lora_parameters(16, hidden)
        + lora_parameters(16, kv_width)
        + lora_parameters(16, kv_width)
        + lora_parameters(16, hidden)
    )
    value_per_layer = lora_parameters(8, hidden) + lora_parameters(8, kv_width)
    policy_lora = policy_per_layer * layers
    value_lora = value_per_layer * layers
    scalar_head = hidden + 1
    value_trainable = value_lora + scalar_head
    return {
        "derivation": "Qwen2 attention projection dimensions from pinned config metadata",
        "model_revision": identity["revision"],
        "published_base_parameters": identity["published_parameter_count"],
        "hidden_size": hidden,
        "head_dimension": hidden // heads,
        "kv_projection_width": kv_width,
        "num_hidden_layers": layers,
        "policy_lora_trainable_parameters": policy_lora,
        "value_lora_trainable_parameters": value_lora,
        "value_scalar_head_trainable_parameters": scalar_head,
        "ppo_value_trainable_parameters": value_trainable,
        "ppo_optimizer_trainable_parameters": policy_lora + value_trainable,
        "grpo_optimizer_trainable_parameters": policy_lora,
        "policy_lora_trainable_percent_of_base": 100
        * policy_lora
        / identity["published_parameter_count"],
        "ppo_value_trainable_percent_of_base": 100
        * value_trainable
        / identity["published_parameter_count"],
        "ppo_optimizer_trainable_percent_of_two_bases": 100
        * (policy_lora + value_trainable)
        / (2 * identity["published_parameter_count"]),
        "policy_value_trainable_overlap": 0,
        "reference_trainable_parameters": 0,
        "reward_trainable_parameters": 0,
        "parameter_counts_require_model_load": False,
    }
