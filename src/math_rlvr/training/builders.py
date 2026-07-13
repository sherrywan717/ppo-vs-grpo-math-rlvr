"""Real TRL trainer builders with model-free dry-run contracts."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from math_rlvr.config import resolve_ppo_smoke_contract, validate_training_config
from math_rlvr.training.model_source import ValidatedModelSource


@dataclass(frozen=True)
class TrainerPlan:
    algorithm: str
    model_name: str
    manifest: Path
    output_dir: Path
    max_steps: int
    seed: int
    policy_lora: dict[str, Any]
    value_model: dict[str, Any] | None


def trainer_plan(config, output_dir):
    algorithm = config["experiment"]["algorithm"]
    validate_training_config(config, algorithm)
    return TrainerPlan(
        algorithm,
        config["model"]["name_or_path"],
        Path(config["data"]["manifest"]),
        Path(output_dir),
        config["training"]["max_steps"],
        config["experiment"]["seed"],
        dict(config["lora"]),
        dict(config["value_model"]) if algorithm == "ppo" else None,
    )


def policy_peft_config(config):
    from peft import LoraConfig, TaskType

    lora = config["lora"]
    return LoraConfig(
        r=lora["rank"],
        lora_alpha=lora["alpha"],
        lora_dropout=lora["dropout"],
        target_modules=lora["target_modules"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )


def value_peft_config(config):
    from peft import LoraConfig, TaskType

    value = config["value_model"]
    return LoraConfig(
        r=value["lora_rank"],
        lora_alpha=value["lora_alpha"],
        lora_dropout=0,
        target_modules=value["lora_target_modules"],
        bias="none",
        task_type=TaskType.SEQ_CLS,
        modules_to_save=["score"],
    )


def grpo_config(config, output_dir, cpu_only=False):
    from trl import GRPOConfig

    validate_training_config(config, "grpo")
    generation, training, model = config["generation"], config["training"], config["model"]
    model_init_kwargs = {"local_files_only": model["local_files_only"], "dtype": model["dtype"]}
    if model.get("revision"):
        model_init_kwargs["revision"] = model["revision"]
    return GRPOConfig(
        output_dir=str(output_dir),
        seed=config["experiment"]["seed"],
        bf16=not cpu_only,
        use_cpu=cpu_only,
        max_steps=training["max_steps"],
        per_device_train_batch_size=training["per_device_train_batch_size"],
        gradient_accumulation_steps=training["gradient_accumulation_steps"],
        learning_rate=1e-5,
        logging_steps=1,
        save_strategy=training["save_strategy"],
        save_steps=training["save_steps"],
        save_total_limit=training["save_total_limit"],
        save_only_model=training["save_only_model"],
        report_to=training["report_to"],
        push_to_hub=training["push_to_hub"],
        max_prompt_length=generation["max_prompt_length"],
        max_completion_length=generation["max_completion_length"],
        num_generations=generation["num_generations"],
        generation_batch_size=generation["generation_batch_size"],
        num_iterations=training["num_iterations"],
        temperature=generation["temperature"],
        top_p=generation["top_p"],
        use_vllm=False,
        gradient_checkpointing=config["model"]["gradient_checkpointing"],
        model_init_kwargs=model_init_kwargs,
    )


def ppo_config(config, output_dir, cpu_only=False):
    """Construct TRL 0.24.0 PPOConfig solely from the resolved smoke YAML."""
    from trl import PPOConfig

    validate_training_config(config, "ppo")
    resolve_ppo_smoke_contract(config)
    generation, training = config["generation"], config["training"]
    return PPOConfig(
        output_dir=str(output_dir),
        seed=config["experiment"]["seed"],
        bf16=not cpu_only,
        use_cpu=cpu_only,
        max_steps=training["max_steps"],
        per_device_train_batch_size=training["per_device_train_batch_size"],
        gradient_accumulation_steps=training["gradient_accumulation_steps"],
        learning_rate=1e-5,
        logging_steps=1,
        save_strategy=training["save_strategy"],
        save_steps=training["save_steps"],
        save_total_limit=training["save_total_limit"],
        save_only_model=training["save_only_model"],
        report_to=training["report_to"],
        push_to_hub=training["push_to_hub"],
        response_length=generation["max_new_tokens"],
        temperature=generation["temperature"],
        num_ppo_epochs=training["num_ppo_epochs"],
        num_mini_batches=training["num_mini_batches"],
        total_episodes=training["total_episodes"],
        local_rollout_forward_batch_size=training["local_rollout_forward_batch_size"],
        gradient_checkpointing=config["model"]["gradient_checkpointing"],
        num_sample_generations=0,
        stop_token="eos",
    )


def build_grpo_trainer(
    config,
    dataset,
    reward_func,
    output_dir,
    model=None,
    tokenizer=None,
    trainer_factory=None,
    cpu_only=None,
    model_source: ValidatedModelSource | None = None,
):
    from trl import GRPOTrainer

    factory = trainer_factory or GRPOTrainer
    if cpu_only is None:
        cpu_only = trainer_factory is not None
    args = grpo_config(config, output_dir, cpu_only=cpu_only)
    return factory(
        model=model
        or str(model_source.snapshot_path if model_source else config["model"]["name_or_path"]),
        reward_funcs=reward_func,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=None if model is not None else policy_peft_config(config),
    )


def build_ppo_trainer(
    config,
    dataset,
    policy,
    ref_model,
    reward_model,
    value_model,
    tokenizer,
    output_dir,
    trainer_factory=None,
    cpu_only=None,
):
    """Build only; caller owns the explicit guarded train invocation."""
    from trl import PPOTrainer

    factory = trainer_factory or PPOTrainer
    if cpu_only is None:
        cpu_only = trainer_factory is not None
    return factory(
        args=ppo_config(config, output_dir, cpu_only=cpu_only),
        processing_class=tokenizer,
        model=policy,
        ref_model=ref_model,
        reward_model=reward_model,
        train_dataset=dataset,
        value_model=value_model,
        peft_config=None,
    )


def _validated_local_name(config, model_source: ValidatedModelSource) -> str:
    if (
        config["model"]["name_or_path"] != model_source.repo_id
        or config["model"].get("revision") != model_source.revision
        or config["model"].get("local_files_only") is not True
        or not model_source.local_files_only
    ):
        raise ValueError("validated model source does not match resolved config")
    return str(model_source.snapshot_path)


def load_policy_and_tokenizer(config, model_source: ValidatedModelSource | None = None):
    import torch
    from peft import get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    name = (
        _validated_local_name(config, model_source)
        if model_source
        else config["model"]["name_or_path"]
    )
    kwargs = {"local_files_only": config["model"].get("local_files_only", True)}
    if model_source is None and config["model"].get("revision"):
        kwargs["revision"] = config["model"]["revision"]
    tokenizer = AutoTokenizer.from_pretrained(name, **kwargs)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        name, dtype=torch.bfloat16, attn_implementation="eager", **kwargs
    )
    model.config.use_cache = False
    return get_peft_model(model, policy_peft_config(config)), tokenizer


def load_value_model(config, model_source: ValidatedModelSource):
    """Load an independent value backbone from the exact local snapshot."""
    import torch
    from peft import get_peft_model
    from transformers import AutoModelForSequenceClassification

    name = _validated_local_name(config, model_source)
    if config["value_model"]["base_checkpoint"] != model_source.repo_id:
        raise ValueError("value model source differs from pinned policy source")
    model = AutoModelForSequenceClassification.from_pretrained(
        name, num_labels=1, local_files_only=True, dtype=torch.bfloat16, attn_implementation="eager"
    )
    model.config.pad_token_id = model.config.eos_token_id
    return get_peft_model(model, value_peft_config(config))


def audit_ppo_parameter_roles(policy, value_model, reward_model, ref_model=None, optimizer=None):
    """Prove a disjoint and exhaustive optimizer-role partition."""
    policy_trainable = {id(p): n for n, p in policy.named_parameters() if p.requires_grad}
    value_trainable = {id(p): n for n, p in value_model.named_parameters() if p.requires_grad}
    if not policy_trainable or any("lora_" not in n for n in policy_trainable.values()):
        raise RuntimeError("policy trainable parameters must be LoRA-only")
    if not value_trainable or any(
        "lora_" not in n and "score" not in n for n in value_trainable.values()
    ):
        raise RuntimeError("value trainables must be value LoRA or scalar score head")
    if set(policy_trainable) & set(value_trainable):
        raise RuntimeError("policy and value trainable parameter objects overlap")
    if any(p.requires_grad for p in reward_model.parameters()):
        raise RuntimeError("parameter-free reward adapter unexpectedly trainable")
    if ref_model is not None and any(p.requires_grad for p in ref_model.parameters()):
        raise RuntimeError("reference model unexpectedly trainable")
    expected_optimizer_ids = set(policy_trainable) | set(value_trainable)
    optimizer_ids = set()
    if optimizer is not None:
        optimizer_ids = {
            id(parameter) for group in optimizer.param_groups for parameter in group["params"]
        }
        if optimizer_ids != expected_optimizer_ids:
            missing = len(expected_optimizer_ids - optimizer_ids)
            unexpected = len(optimizer_ids - expected_optimizer_ids)
            raise RuntimeError(
                f"PPO optimizer role mismatch: missing={missing}, unexpected={unexpected}"
            )
    ref_ids = {id(parameter) for parameter in ref_model.parameters()} if ref_model else set()
    reward_ids = {id(parameter) for parameter in reward_model.parameters()}
    if optimizer_ids & (ref_ids | reward_ids):
        raise RuntimeError("reference/reward parameters entered PPO optimizer")
    return {
        "reference_mode": "peft_disable_adapter" if ref_model is None else "independent_frozen",
        "policy_trainable_names": sorted(policy_trainable.values()),
        "value_trainable_names": sorted(value_trainable.values()),
        "policy_trainable_parameters": sum(
            p.numel() for p in policy.parameters() if p.requires_grad
        ),
        "value_trainable_parameters": sum(
            p.numel() for p in value_model.parameters() if p.requires_grad
        ),
        "reward_trainable_parameters": 0,
        "reference_trainable_parameters": 0,
        "optimizer_parameter_tensors": len(optimizer_ids),
        "optimizer_parameter_elements": sum(
            parameter.numel() for group in optimizer.param_groups for parameter in group["params"]
        )
        if optimizer is not None
        else None,
        "optimizer_exact_role_match": optimizer is not None,
    }
