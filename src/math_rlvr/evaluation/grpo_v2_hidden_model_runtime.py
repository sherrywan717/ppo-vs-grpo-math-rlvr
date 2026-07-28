"""Delayed CUDA worker for one frozen GRPO-v2 hidden-test model role."""

from __future__ import annotations

import gc
import hashlib
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

from math_rlvr.evaluation.grpo_v2_dev_model_runtime import _flat_metric_rows
from math_rlvr.evaluation.grpo_v2_dev_runtime import (
    completion_record,
    require_finite_logits,
    validate_inference_contract,
    write_csv,
)
from math_rlvr.evaluation.grpo_v2_hidden_runtime import (
    CONFIG_PATH,
    HiddenBudgetGuard,
    HiddenEvaluationContractError,
    aggregate_hidden_candidate0,
    artifact_schema,
    build_hidden_plan,
    validate_hidden_rows,
)
from math_rlvr.grpo_v2_contract import (
    aggregate_unbiased_pass_k,
    validate_shared_candidate_batch,
)
from math_rlvr.training.warmstart_runtime import file_sha256


def _trusted_problems(config: dict[str, Any], public_rows: list[dict[str, Any]]):
    from math_rlvr.dataset import MathProblem

    trusted_path = Path(config["data"]["trusted_manifest"])
    if file_sha256(trusted_path) != config["data"]["trusted_manifest_sha256"]:
        raise HiddenEvaluationContractError("hidden trusted manifest SHA mismatch")
    trusted = {
        row["problem_id"]: row
        for row in (
            json.loads(line) for line in trusted_path.read_text().splitlines() if line.strip()
        )
    }
    if len(trusted) != 400:
        raise HiddenEvaluationContractError("hidden trusted manifest count mismatch")
    result = {}
    for row in public_rows:
        gold = trusted.get(row["problem_id"])
        if not isinstance(gold, dict) or gold.get("content_hash") != row["content_hash"]:
            raise HiddenEvaluationContractError("hidden trusted/public identity mismatch")
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
            gold_answer=gold["gold_answer"],
        )
    return result


def _pass_k_rows(rows: list[dict[str, Any]], shared_problem_ids: set[str]):
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row["problem_id"] in shared_problem_ids:
            grouped.setdefault(row["problem_id"], []).append(row)
    result = []
    for problem_id in sorted(grouped):
        evidence = validate_shared_candidate_batch(grouped[problem_id])
        first = grouped[problem_id][0]
        result.append(
            {
                "problem_id": problem_id,
                "dataset": first["dataset"],
                "math_level": first["math_level"],
                "n": evidence["n"],
                "c": evidence["c"],
                "candidate_correctness": json.dumps(evidence["candidate_correctness"]),
                "candidate_evidence_references": json.dumps(
                    evidence["candidate_evidence_references"]
                ),
                "candidate_indices": [
                    row["candidate_index"]
                    for row in sorted(grouped[problem_id], key=lambda item: item["candidate_index"])
                ],
                "candidate_seeds": [
                    row["sampling_seed"]
                    for row in sorted(grouped[problem_id], key=lambda item: item["candidate_index"])
                ],
                "duplicate_rate": 1
                - len(
                    {
                        (row["completion_text"], tuple(row["completion_ids"]))
                        for row in grouped[problem_id]
                    }
                )
                / 10,
                "generated_tokens": sum(
                    int(row["exact_token_count"]) for row in grouped[problem_id]
                ),
                "estimates": evidence["estimates"],
            }
        )
    if len(result) != 100:
        raise HiddenEvaluationContractError("hidden pass@k problem count mismatch")
    return result


def _pass_k_summary(problem_rows: list[dict[str, Any]]) -> tuple[dict, list[dict]]:
    slices: dict[str, list[dict[str, Any]]] = {
        "overall": problem_rows,
        "gsm8k": [row for row in problem_rows if row["dataset"] == "gsm8k"],
        "math": [row for row in problem_rows if row["dataset"] == "math"],
    }
    for level in range(1, 6):
        slices[f"math_level_{level}"] = [
            row for row in problem_rows if row["math_level"] == str(level)
        ]
    summary: dict[str, Any] = {}
    flat: list[dict[str, Any]] = []
    for name, subset in slices.items():
        aggregate = aggregate_unbiased_pass_k(subset)
        summary[name] = aggregate
        for k, metric in aggregate["metrics"].items():
            flat.append(
                {
                    "slice": name,
                    "k": int(k),
                    "problem_denominator": aggregate["problem_denominator"],
                    **metric,
                }
            )
    return summary, flat


def execute_hidden_worker(
    *,
    config: dict[str, Any],
    identity: dict[str, Any],
    public_rows: list[dict[str, Any]],
    shared_problem_ids: set[str],
    role: str,
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
        "python -m math_rlvr.evaluation.grpo_v2_hidden "
        f"--config {CONFIG_PATH} --role {role} --run-dir {run_dir} "
        "--execute --confirm-grpo-v2-hidden"
    )
    if checkpoint_identity is not None:
        command += f" --checkpoint {checkpoint_identity['checkpoint_path']}"
    manager = ArtifactManager(
        "grpo_v2_hidden_evaluation",
        role,
        FORMAL_REPO_ID,
        42,
        command,
        config,
        run_id=run_dir.name,
        create_checkpoints=False,
    )
    if manager.run_dir != run_dir:
        raise HiddenEvaluationContractError("hidden ArtifactManager run directory mismatch")
    manager.write_json("resolved_config.json", config)
    manager.write_json(
        "evaluation_identity.json",
        {
            **identity,
            "role": role,
            "checkpoint": checkpoint_identity,
            "model_repo": model_source.repo_id,
            "model_revision": model_source.revision,
            "local_files_only": model_source.local_files_only,
            "environment": environment,
        },
    )
    monitor = ResourceMonitor(run_dir / "resource_metrics.csv", interval=0.25)
    allocator = CudaAllocatorEvidence(torch.cuda)
    started = time.monotonic()
    guard = HiddenBudgetGuard(deadline=started + config["budget"]["max_wall_time_seconds"])
    plan = build_hidden_plan(public_rows, shared_problem_ids)
    plan_by_problem: dict[str, list[dict[str, Any]]] = {}
    for row in plan:
        plan_by_problem.setdefault(row["problem_id"], []).append(row)
    model = tokenizer = None
    try:
        monitor.start()
        allocator.start()
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
        if checkpoint_identity is not None:
            from peft import PeftModel

            model = PeftModel.from_pretrained(
                model,
                checkpoint_identity["adapter_path"],
                local_files_only=True,
                is_trainable=False,
            )
        model.requires_grad_(False)
        model.eval()
        inference = validate_inference_contract(
            model_training=model.training,
            parameter_requires_grad=[p.requires_grad for p in model.parameters()],
            inference_mode_used=True,
        )
        if not torch.cuda.is_available():
            raise HiddenEvaluationContractError("authorized hidden evaluation requires CUDA")
        device = torch.device("cuda:0")
        model = model.to(device)
        if any(parameter.device.type == "meta" for parameter in model.parameters()):
            raise HiddenEvaluationContractError("hidden model retains meta parameters")
        manager.write_json(
            "model_roles.json",
            {
                "role": role,
                "adapter_loaded": checkpoint_identity is not None,
                "adapter_role": "policy" if checkpoint_identity is not None else None,
                **inference,
            },
        )
        problems = _trusted_problems(config, public_rows)
        verifier = MathVerifier()
        rows: list[dict[str, Any]] = []
        sampling_base = {
            "temperature": config["sampling"]["temperature"],
            "top_p": config["sampling"]["top_p"],
            "top_k": config["sampling"]["top_k"],
            "max_completion_length": config["sampling"]["max_completion_length"],
        }
        for public in public_rows:
            expected_rows = plan_by_problem[public["problem_id"]]
            problem = problems[public["problem_id"]]
            prompt = render_prompt_version(tokenizer, problem, PROMPT_V2_FORMAL_MATH)
            prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
            encoded = tokenizer(
                prompt, return_tensors="pt", add_special_tokens=False, truncation=False
            )
            prompt_tokens = int(encoded["input_ids"].shape[1])
            if (
                prompt_tokens > config["prompt"]["max_prompt_length"]
                or prompt_tokens + 256 > config["prompt"]["max_sequence_length"]
            ):
                raise HiddenEvaluationContractError("hidden prompt capacity mismatch")
            encoded = {key: value.to(device) for key, value in encoded.items()}
            batch_seed = expected_rows[0]["batch_seed"]
            torch.manual_seed(batch_seed)
            torch.cuda.manual_seed_all(batch_seed)
            count = 10 if expected_rows[0]["shared_n10"] else 1
            sampling = {**sampling_base, "num_return_sequences": count}
            generate_call_id = f"{role}:{public['problem_id']}:{batch_seed}"
            prompt_width = int(encoded["input_ids"].shape[1])
            with torch.inference_mode():
                outputs = model.generate(
                    **encoded,
                    do_sample=True,
                    temperature=sampling["temperature"],
                    top_p=sampling["top_p"],
                    num_return_sequences=count,
                    max_new_tokens=256,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    logits_processor=[FiniteLogits()],
                )
            if int(outputs.shape[0]) != count:
                raise HiddenEvaluationContractError("hidden generate candidate count mismatch")
            for expected, output in zip(expected_rows, outputs, strict=True):
                token_ids = [int(value) for value in output[prompt_width:].cpu().tolist()]
                eos = tokenizer.eos_token_id in token_ids
                if eos:
                    token_ids = token_ids[: token_ids.index(tokenizer.eos_token_id) + 1]
                else:
                    while token_ids and token_ids[-1] == tokenizer.pad_token_id:
                        token_ids.pop()
                text = tokenizer.decode(token_ids, skip_special_tokens=True)
                evaluation = FORMAL_REWARD_POLICY.evaluate(
                    text, lambda candidate, problem=problem: verifier(problem, candidate)
                )
                row = completion_record(
                    plan_row=expected,
                    prompt_hash=prompt_hash,
                    completion_ids=token_ids,
                    completion_mask=[1] * len(token_ids),
                    text=text,
                    eos=eos,
                    truncated=not eos and len(token_ids) == 256,
                    evaluation=evaluation,
                    mode=role,
                    checkpoint_identity=checkpoint_identity,
                )
                row.update(
                    {
                        "model_identity": role,
                        "checkpoint_identity": checkpoint_identity or f"base:{FORMAL_REPO_ID}",
                        "batch_seed": batch_seed,
                        "sampling_config": sampling,
                        "generate_call_id": generate_call_id,
                        "canonical_correct": bool(row["canonical_correct"]),
                        "verifier_status": str(row["canonical_status"]).upper(),
                        "evidence_ref": (
                            f"completions.jsonl:{public['problem_id']}:{expected['candidate_index']}"
                        ),
                    }
                )
                guard.record(row, peak_vram_gib=torch.cuda.max_memory_reserved(0) / (1024**3))
                manager.append_jsonl("completions.jsonl", row)
                rows.append(row)
        rows = validate_hidden_rows(plan, rows)
        all_ids = {row["problem_id"] for row in public_rows}
        counters = guard.finalize(all_ids, shared_problem_ids, rows)
        candidate0 = [row for row in rows if row["candidate_index"] == 0]
        candidate_metrics = aggregate_hidden_candidate0(candidate0)
        pass_problem_rows = _pass_k_rows(rows, shared_problem_ids)
        pass_summary, pass_summary_rows = _pass_k_summary(pass_problem_rows)
        write_csv(run_dir / "per_problem.csv", candidate0)
        write_csv(run_dir / "candidate0_metrics.csv", _flat_metric_rows(candidate_metrics))
        manager.write_json("candidate0_metrics.json", candidate_metrics)
        write_csv(
            run_dir / "pass_k_per_problem.csv",
            [
                {
                    **{
                        key: row[key]
                        for key in (
                            "problem_id",
                            "dataset",
                            "math_level",
                            "n",
                            "c",
                            "candidate_indices",
                            "candidate_seeds",
                            "duplicate_rate",
                            "generated_tokens",
                        )
                    },
                    **{f"pass_at_{k}": row["estimates"][str(k)]["float_value"] for k in (1, 4, 10)},
                }
                for row in pass_problem_rows
            ],
        )
        manager.write_json("pass_k_summary.json", pass_summary)
        write_csv(run_dir / "pass_k_summary.csv", pass_summary_rows)
        status = Counter(row["canonical_status"] for row in rows)
        write_csv(
            run_dir / "status_distribution.csv",
            [{"status": key, "count": value} for key, value in sorted(status.items())],
        )
        write_csv(
            run_dir / "truncation_analysis.csv",
            [
                {
                    "scope": "all_candidates",
                    "truncated": sum(row["truncated"] for row in rows),
                    "denominator": len(rows),
                },
                {
                    "scope": "candidate0",
                    "truncated": sum(row["truncated"] for row in candidate0),
                    "denominator": len(candidate0),
                },
            ],
        )
        (run_dir / "figures").mkdir(exist_ok=True)
        resource_summary = {"available": True, **monitor.summary()}
        allocator_summary = allocator.finalize()
        manager.write_json("resource_summary.json", resource_summary)
        summary = {
            "status": "scientific_evaluation_success",
            "role": role,
            "counters": counters,
            "candidate0_metrics": candidate_metrics,
            "pass_k_summary": pass_summary,
            "resource_summary": resource_summary,
            "pytorch_allocator": allocator_summary,
            "checkpoint": checkpoint_identity,
            "artifact_schema": artifact_schema(),
        }
        manager.write_json("summary.json", summary)
        manager.write_text(
            "report.md",
            "# GRPO-v2 frozen hidden evaluation\n\n"
            f"- Role: `{role}`\n- Completion rows: 1,300\n"
            "- Candidate-0: 400 problems; shared n=10 subset: 100 problems.\n"
            "- Training/backward/optimizer/checkpoint writes: 0.\n",
        )
        manager.finalize("success", counters=counters, summary=summary)
        return {
            "status": "success",
            "run_id": run_dir.name,
            "run_dir": str(run_dir),
            "summary_path": str(run_dir / "summary.json"),
            "counts": {
                "completion_rows": 1300,
                "candidate0_rows": 400,
                "shared_problem_rows": 100,
            },
            "failure_reason": None,
        }
    except BaseException as exc:
        manager.finalize(
            "failure",
            failed_stage="grpo_v2_hidden_evaluation",
            exception=exc,
            stop_reason=str(exc),
            counters={"completions": guard.completions, "generated_tokens": guard.generated_tokens},
        )
        raise
    finally:
        monitor.stop()
        model = None
        tokenizer = None
        gc.collect()
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()
