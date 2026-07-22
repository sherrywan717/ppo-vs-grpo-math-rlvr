"""Delayed model-bound warm-start execution, imported only after dual confirmation."""

from __future__ import annotations

import hashlib
import json
import os
import random
import time
from pathlib import Path
from typing import Any

from math_rlvr.dataset import MathProblem
from math_rlvr.prompt import format_problem_v2
from math_rlvr.training.warmstart_runtime import (
    WarmstartBudgetGuard,
    WarmstartContractError,
    backup_warmstart_run,
    completion_only_collate,
    encode_completion_only,
)


def _atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def execute_real_warmstart(
    config: dict[str, Any],
    *,
    identity: dict[str, Any],
    model_source,
    run_dir: Path,
    environment: dict[str, str],
) -> dict[str, Any]:
    """Run one exact completion-only warm-start after the guarded CLI boundary."""
    import numpy as np
    import torch
    from peft import LoraConfig, get_peft_model
    from torch.utils.data import SequentialSampler
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainerCallback,
        TrainingArguments,
    )

    run_dir.mkdir(parents=False, exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(
        model_source.snapshot_path, local_files_only=True, trust_remote_code=False
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_source.snapshot_path, local_files_only=True, torch_dtype=torch.bfloat16
    )
    lora = config["lora"]
    model = get_peft_model(
        model,
        LoraConfig(
            r=lora["rank"],
            lora_alpha=lora["alpha"],
            lora_dropout=lora["dropout"],
            target_modules=lora["target_modules"],
            task_type="CAUSAL_LM",
        ),
    )
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    if trainable != config["training"]["trainable_parameter_count"]:
        raise WarmstartContractError(f"policy LoRA trainable count mismatch: {trainable}")
    _atomic_json(
        run_dir / "model_roles.json",
        {
            "policy_adapter_trainable_parameters": trainable,
            "base_model_trainable_parameters": 0,
            "model_revision": model_source.revision,
            "local_files_only": True,
        },
    )

    public_rows = [
        json.loads(line) for line in Path(config["data"]["manifest"]).read_text().splitlines()
    ]
    target_rows = {
        row["problem_id"]: row
        for row in (
            json.loads(line)
            for line in Path(config["data"]["trusted_targets"]).read_text().splitlines()
        )
    }
    trusted_path = Path(config["data"]["trusted_targets"]).with_name("warmstart_v2_trusted.jsonl")
    trusted_rows = {
        row["problem_id"]: row
        for row in (json.loads(line) for line in trusted_path.read_text().splitlines())
    }
    features = []
    for row in public_rows:
        problem = MathProblem(
            **{
                key: row[key]
                for key in (
                    "problem_id",
                    "source",
                    "prompt",
                    "category",
                    "difficulty",
                    "split",
                    "source_index",
                    "content_hash",
                    "metadata",
                )
            },
            gold_answer=trusted_rows[row["problem_id"]]["gold_answer"],
        )
        encoded = encode_completion_only(
            tokenizer,
            format_problem_v2(problem),
            target_rows[row["problem_id"]]["target_text"],
            max_prompt=config["prompt"]["max_prompt_length"],
            max_target=config["prompt"]["max_target_length"],
            max_sequence=config["prompt"]["max_sequence_length"],
        )
        features.append({**encoded, "problem_id": row["problem_id"]})
    if len(features) != 256:
        raise WarmstartContractError("warm-start feature count mismatch")

    guard = WarmstartBudgetGuard()
    metric_rows: list[dict[str, Any]] = []

    def collator(batch):
        guard.record_batch(
            sample_ids=[row["problem_id"] for row in batch],
            active_label_tokens=sum(row["active_label_tokens"] for row in batch),
        )
        payload = completion_only_collate(batch, pad_token_id=tokenizer.pad_token_id)
        return {key: torch.tensor(value, dtype=torch.long) for key, value in payload.items()}

    class OrderedWarmstartTrainer(Trainer):
        def _get_train_sampler(self, train_dataset=None):
            return SequentialSampler(train_dataset or self.train_dataset)

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            outputs = model(**inputs)
            loss = outputs.loss
            guard.record_microstep(float(loss.detach().float().cpu()))
            return (loss, outputs) if return_outputs else loss

    started = time.monotonic()

    class EvidenceCallback(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            guard.record_optimizer_step(int(state.global_step))
            return control

        def on_epoch_end(self, args, state, control, **kwargs):
            guard.record_epoch()
            return control

        def on_log(self, args, state, control, logs=None, **kwargs):
            logs = dict(logs or {})
            metric_rows.append(
                {
                    "global_step": int(state.global_step),
                    "loss": logs.get("loss"),
                    "learning_rate": logs.get("learning_rate"),
                    "grad_norm": logs.get("grad_norm"),
                    "grad_norm_available": logs.get("grad_norm") is not None,
                    "grad_norm_reason": None
                    if logs.get("grad_norm") is not None
                    else "trainer_did_not_expose_metric",
                    "wall_time_seconds": time.monotonic() - started,
                }
            )
            _atomic_json(run_dir / "metrics.json", metric_rows)
            return control

    arguments = TrainingArguments(
        output_dir=str(run_dir / "trainer"),
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        max_steps=16,
        num_train_epochs=1,
        learning_rate=config["training"]["learning_rate"],
        bf16=True,
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        remove_unused_columns=False,
        dataloader_num_workers=0,
        dataloader_drop_last=True,
        seed=42,
        data_seed=42,
    )
    try:
        trainer = OrderedWarmstartTrainer(
            model=model,
            args=arguments,
            train_dataset=features,
            data_collator=collator,
            callbacks=[EvidenceCallback()],
        )
        trainer.train()
        counters = guard.finalize()
        checkpoint = run_dir / "checkpoint-16"
        adapter = checkpoint / "adapter"
        adapter.mkdir(parents=True)
        model.save_pretrained(adapter, safe_serialization=True)
        torch.save(trainer.optimizer.state_dict(), checkpoint / "optimizer.pt")
        torch.save(trainer.lr_scheduler.state_dict(), checkpoint / "scheduler.pt")
        torch.save(
            {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch_cpu": torch.get_rng_state(),
                "torch_cuda": torch.cuda.get_rng_state_all(),
            },
            checkpoint / "rng_state.pt",
        )
        trainer.state.save_to_json(str(checkpoint / "trainer_state.json"))
        _atomic_json(checkpoint / "runtime_state.json", counters)
        _atomic_json(
            checkpoint / "checkpoint_identity.json",
            {
                "run_id": run_dir.name,
                "adapter_role": "policy",
                "seed": 42,
                "config_sha256": identity["config_sha256"],
                "model_repo": model_source.repo_id,
                "model_revision": model_source.revision,
                "data_cursor": 256,
                "dataset_order_sha256": identity["dataset_order_sha256"],
                "warmstart_manifest_sha256": identity["warmstart_manifest_sha256"],
                "train_manifest_sha256": identity["train_manifest_sha256"],
                "data_registry_sha256": identity["data_registry_sha256"],
                "prompt_sha256": identity["prompt_sha256"],
                "reward_sha256": identity["reward_sha256"],
                "parser_sha256": identity["parser_sha256"],
                "verifier_sha256": identity["verifier_sha256"],
            },
        )
        files = {}
        for path in sorted(checkpoint.rglob("*")):
            if path.is_file():
                files[path.relative_to(checkpoint).as_posix()] = {
                    "size": path.stat().st_size,
                    "sha256": _sha256(path),
                }
        artifact_sha = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
        _atomic_json(
            checkpoint / "artifact_manifest.json",
            {
                "files": files,
                "artifact_sha256": artifact_sha,
                "base_weights_included": False,
            },
        )
        _atomic_json(
            run_dir / "summary.json",
            {
                "status": "success",
                "counters": counters,
                "checkpoint_artifact_sha256": artifact_sha,
            },
        )
        backup = backup_warmstart_run(run_dir, failure=False)
        _atomic_json(run_dir / "backup_manifest.json", {"verified": True, **backup})
        return {"status": "success", "run_dir": str(run_dir), "backup": backup}
    except Exception as exc:
        _atomic_json(
            run_dir / "failure.json",
            {
                "status": "failure",
                "reason": str(exc),
                "counters": {
                    key: value for key, value in guard.__dict__.items() if key != "seen_sample_ids"
                },
            },
        )
        backup = backup_warmstart_run(run_dir, failure=True)
        _atomic_json(run_dir / "backup_manifest.json", {"verified": True, **backup})
        raise
