"""Delayed CUDA worker for the frozen matched GRPO-v2 dev evaluation."""

from __future__ import annotations

import gc
import hashlib
import json
import time
from pathlib import Path
from typing import Any

from math_rlvr.evaluation.grpo_v2_dev_runtime import (
    DevBudgetGuard,
    DevEvaluationContractError,
    aggregate_dev_rows,
    build_dev_plan,
    completion_record,
    require_finite_logits,
    validate_dev_rows,
    validate_inference_contract,
    write_csv,
)


def _trusted_problems(config: dict[str, Any], public_rows: list[dict[str, Any]]):
    from math_rlvr.dataset import MathProblem

    trusted = {
        row["problem_id"]: row
        for row in (
            json.loads(line)
            for line in Path(config["dev"]["trusted_manifest"]).read_text().splitlines()
            if line.strip()
        )
    }
    result = {}
    for row in public_rows:
        result[row["problem_id"]] = MathProblem(
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
            gold_answer=trusted[row["problem_id"]]["gold_answer"],
        )
    return result


def _flat_metric_rows(aggregate: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for name, metrics in aggregate["slices"].items():
        row: dict[str, Any] = {"slice": name, "problems": metrics["problems"]}
        for metric, evidence in metrics.items():
            if metric == "problems":
                continue
            row[f"{metric}_numerator"] = evidence["numerator"]
            row[f"{metric}_denominator"] = evidence["denominator"]
            row[metric] = evidence["value"]
        rows.append(row)
    return rows


def execute_dev_worker(
    *,
    config: dict[str, Any],
    identity: dict[str, Any],
    public_rows: list[dict[str, Any]],
    mode: str,
    checkpoint_identity: dict[str, Any] | None,
    model_source,
    run_dir: Path,
    environment: dict[str, str],
) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor

    from math_rlvr.artifacts.manager import ArtifactManager
    from math_rlvr.artifacts.monitor import ResourceMonitor
    from math_rlvr.prompt import PROMPT_V2_FORMAL_MATH, render_prompt_version
    from math_rlvr.rewards.formal import FORMAL_REWARD_POLICY
    from math_rlvr.training.model_source import FORMAL_REPO_ID
    from math_rlvr.training.resource_evidence import CudaAllocatorEvidence
    from math_rlvr.verifier import MathVerifier

    class FiniteLogits(LogitsProcessor):
        def __call__(self, input_ids, scores):
            require_finite_logits(bool(torch.isfinite(scores).all()))
            return scores

    command = (
        "HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src "
        "python -m math_rlvr.evaluation.grpo_v2_dev "
        f"--config configs/grpo_v2/dev_evaluation_seed42.json --mode {mode} "
        f"--run-dir {run_dir} --execute --confirm-grpo-v2-dev"
    )
    if mode == "warmstart":
        command += f" --checkpoint {config['warmstart_checkpoint']['path']}"
    manager = ArtifactManager(
        "grpo_v2_dev_evaluation",
        mode,
        FORMAL_REPO_ID,
        42,
        command,
        config,
        run_id=run_dir.name,
        create_checkpoints=False,
    )
    if manager.run_dir != run_dir:
        raise DevEvaluationContractError("dev ArtifactManager run directory mismatch")
    manager.write_json("resolved_config.json", config)
    manager.write_json(
        "evaluation_identity.json",
        {
            "mode": mode,
            "config_sha256": identity["config_sha256"],
            "data_registry_sha256": identity["data_registry_sha256"],
            "dev_manifest_sha256": identity["dev_manifest_sha256"],
            "model_repo": model_source.repo_id,
            "model_revision": model_source.revision,
            "local_files_only": model_source.local_files_only,
            "checkpoint": checkpoint_identity,
            "environment": environment,
        },
    )
    monitor = ResourceMonitor(run_dir / "resource_metrics.csv", interval=0.25)
    allocator = CudaAllocatorEvidence(torch.cuda)
    started = time.monotonic()
    plan = build_dev_plan(config, public_rows, mode=mode)
    guard = DevBudgetGuard(deadline=started + config["budget"]["max_wall_time_seconds"])
    model = tokenizer = None
    try:
        monitor.start()
        allocator.start()
        if not model_source.local_files_only:
            raise DevEvaluationContractError("dev evaluation requires local-only model source")
        tokenizer = AutoTokenizer.from_pretrained(
            model_source.snapshot_path, local_files_only=True, trust_remote_code=False
        )
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_source.snapshot_path,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
        )
        if mode == "warmstart":
            from peft import PeftModel

            model = PeftModel.from_pretrained(
                model,
                str(Path(config["warmstart_checkpoint"]["path"]) / "adapter"),
                local_files_only=True,
                is_trainable=False,
            )
        model.requires_grad_(False)
        model.eval()
        if any(parameter.requires_grad for parameter in model.parameters()) or model.training:
            raise DevEvaluationContractError("dev model must be eval with all parameters frozen")
        inference_contract = validate_inference_contract(
            model_training=model.training,
            parameter_requires_grad=[parameter.requires_grad for parameter in model.parameters()],
            inference_mode_used=True,
        )
        if not torch.cuda.is_available():
            raise DevEvaluationContractError("authorized dev evaluation requires CUDA")
        device = torch.device("cuda:0")
        model = model.to(device)
        meta_parameters = sum(
            parameter.numel() for parameter in model.parameters() if parameter.device.type == "meta"
        )
        if meta_parameters:
            raise DevEvaluationContractError("dev model retains meta parameters")
        manager.write_json(
            "model_roles.json",
            {
                "mode": mode,
                "adapter_loaded": mode == "warmstart",
                "adapter_role": "policy" if mode == "warmstart" else None,
                "parameters_require_grad": 0,
                "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
                "meta_parameters": meta_parameters,
                "model_eval": True,
                **inference_contract,
            },
        )
        problems = _trusted_problems(config, public_rows)
        verifier = MathVerifier()
        rows = []
        for expected in plan:
            problem = problems[expected["problem_id"]]
            prompt = render_prompt_version(tokenizer, problem, PROMPT_V2_FORMAL_MATH)
            prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
            encoded = tokenizer(
                prompt, return_tensors="pt", add_special_tokens=False, truncation=False
            )
            if int(encoded["input_ids"].shape[1]) > config["prompt"]["max_prompt_length"]:
                raise DevEvaluationContractError("dev prompt exceeds frozen max length")
            encoded = {key: value.to(device) for key, value in encoded.items()}
            torch.manual_seed(expected["generation_seed"])
            torch.cuda.manual_seed_all(expected["generation_seed"])
            prompt_width = int(encoded["input_ids"].shape[1])
            with torch.inference_mode():
                output = model.generate(
                    **encoded,
                    do_sample=True,
                    temperature=config["sampling"]["temperature"],
                    top_p=config["sampling"]["top_p"],
                    max_new_tokens=config["prompt"]["max_completion_length"],
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    logits_processor=[FiniteLogits()],
                )
            completion_ids = [
                int(value) for value in output[0, prompt_width:].detach().cpu().tolist()
            ]
            eos = tokenizer.eos_token_id in completion_ids
            if eos:
                completion_ids = completion_ids[: completion_ids.index(tokenizer.eos_token_id) + 1]
            else:
                while completion_ids and completion_ids[-1] == tokenizer.pad_token_id:
                    completion_ids.pop()
            text = tokenizer.decode(completion_ids, skip_special_tokens=True)
            evaluation = FORMAL_REWARD_POLICY.evaluate(
                text, lambda candidate, problem=problem: verifier(problem, candidate)
            )
            row = completion_record(
                plan_row=expected,
                prompt_hash=prompt_hash,
                completion_ids=completion_ids,
                completion_mask=[1] * len(completion_ids),
                text=text,
                eos=eos,
                truncated=(
                    not eos and len(completion_ids) == config["prompt"]["max_completion_length"]
                ),
                evaluation=evaluation,
                mode=mode,
                checkpoint_identity=checkpoint_identity,
            )
            peak_gib = torch.cuda.max_memory_reserved(0) / (1024**3)
            guard.record(row, peak_vram_gib=peak_gib)
            manager.append_jsonl("completions.jsonl", row)
            rows.append(row)
        rows = validate_dev_rows(plan, rows)
        counters = guard.finalize()
        aggregate = aggregate_dev_rows(rows)
        write_csv(run_dir / "per_problem.csv", rows)
        write_csv(run_dir / "domain_level_metrics.csv", _flat_metric_rows(aggregate))
        write_csv(
            run_dir / "status_distribution.csv",
            [
                {"status": status, "count": count}
                for status, count in aggregate["reward_status_counts"].items()
            ],
        )
        manager.write_json("aggregate_metrics.json", aggregate)
        manager.write_json("sample_ledger.json", counters)
        resource_summary = {"available": True, **monitor.summary()}
        allocator_summary = allocator.finalize()
        manager.write_json("resource_summary.json", resource_summary)
        manager.write_json("pytorch_allocator.json", allocator_summary)
        manager.write_text(
            "report.md",
            "# GRPO-v2 matched dev evaluation\n\n"
            f"- Mode: {mode}\n- Problems/completions: 128/128\n"
            "- Protocol: one candidate per problem; pass@4/pass@10 unavailable.\n"
            "- Training/backward/optimizer/checkpoint writes: 0.\n",
        )
        manager.finalize(
            "success",
            counters=counters,
            summary={"aggregate_metrics": aggregate, "resource_summary": resource_summary},
        )
        return {
            "status": "success",
            "run_id": run_dir.name,
            "run_dir": str(run_dir),
            "counters": counters,
            "aggregate": aggregate,
            "resource_summary": resource_summary,
            "pytorch_allocator": allocator_summary,
        }
    except BaseException as exc:
        manager.finalize(
            "failure",
            failed_stage="grpo_v2_dev_evaluation",
            exception=exc,
            stop_reason=str(exc),
            counters=guard.snapshot(),
        )
        raise
    finally:
        monitor.stop()
        model = None
        tokenizer = None
        gc.collect()
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()
