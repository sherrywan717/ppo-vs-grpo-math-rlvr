"""Real TRL trainer builders with a model-free dry-run plan."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from math_rlvr.config import validate_training_config
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

    v = config["value_model"]
    return LoraConfig(
        r=v["lora_rank"],
        lora_alpha=v["lora_alpha"],
        lora_dropout=0,
        target_modules=v["lora_target_modules"],
        bias="none",
        task_type=TaskType.SEQ_CLS,
        modules_to_save=["score"],
    )


def grpo_config(config, output_dir, cpu_only=False):
    """Construct the real TRL config without loading a model."""
    from trl import GRPOConfig

    validate_training_config(config, "grpo")
    generation = config["generation"]
    training = config["training"]
    model = config["model"]
    model_init_kwargs = {
        "local_files_only": model["local_files_only"],
        "dtype": model["dtype"],
    }
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
):
    from trl import PPOConfig, PPOTrainer

    factory = trainer_factory or PPOTrainer
    g = config["generation"]
    cpu_test = trainer_factory is not None
    args = PPOConfig(
        output_dir=str(output_dir),
        seed=config["experiment"]["seed"],
        bf16=not cpu_test,
        use_cpu=cpu_test,
        max_steps=config["training"]["max_steps"],
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        learning_rate=1e-5,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        response_length=g["max_new_tokens"],
        temperature=g["temperature"],
        num_ppo_epochs=1,
        total_episodes=8,
        local_rollout_forward_batch_size=4,
        gradient_checkpointing=True,
    )
    return factory(
        args=args,
        processing_class=tokenizer,
        model=policy,
        ref_model=ref_model,
        reward_model=reward_model,
        train_dataset=dataset,
        value_model=value_model,
    )


def load_policy_and_tokenizer(
    config, model_source: ValidatedModelSource | None = None
):
    import torch
    from peft import get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if model_source is not None:
        if (
            config["model"]["name_or_path"] != model_source.repo_id
            or config["model"].get("revision") != model_source.revision
            or not model_source.local_files_only
        ):
            raise ValueError("validated model source does not match resolved config")
        name = str(model_source.snapshot_path)
    else:
        name = config["model"]["name_or_path"]
    load_kwargs = {"local_files_only": config["model"].get("local_files_only", True)}
    if model_source is None and config["model"].get("revision"):
        load_kwargs["revision"] = config["model"]["revision"]
    tokenizer = AutoTokenizer.from_pretrained(name, **load_kwargs)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        name, dtype=torch.bfloat16, attn_implementation="eager", **load_kwargs
    )
    model.config.use_cache = False
    return get_peft_model(model, policy_peft_config(config)), tokenizer


def load_value_model(config):
    import torch
    from peft import get_peft_model
    from transformers import AutoModelForSequenceClassification

    name = config["value_model"]["base_checkpoint"]
    model = AutoModelForSequenceClassification.from_pretrained(
        name, num_labels=1, torch_dtype=torch.bfloat16, attn_implementation="eager"
    )
    model.config.pad_token_id = model.config.eos_token_id
    return get_peft_model(model, value_peft_config(config))
