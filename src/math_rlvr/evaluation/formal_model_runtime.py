"""Delayed pinned-model generation for formal baseline and policy-adapter evaluation."""

from __future__ import annotations

import hashlib
import json
import shlex
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from math_rlvr.evaluation.formal_cli import FormalEvaluationSelection
from math_rlvr.evaluation.formal_runtime import (
    execute_formal_evaluation,
    formal_evaluation_plan,
)
from math_rlvr.training.model_source import ValidatedModelSource

RUN_ROOT = Path("/root/autodl-tmp/runs/math_rlvr")


def _formal_completion_record(
    *,
    item: dict[str, Any],
    generation_seed: int,
    completion_ids: list[int],
    text: str,
    max_completion_length: int,
    eos_token_id: int | None,
    evaluation: Any,
) -> dict[str, Any]:
    from math_rlvr.parser import ParsedCompletion, parse_completion
    from math_rlvr.rewards.result import RewardStatus

    status = evaluation.canonical_result.status
    reward_evidence = evaluation.to_dict()
    return {
        **item,
        "generation_seed": generation_seed,
        "completion_ids": completion_ids,
        "completion_mask": [1] * len(completion_ids),
        "exact_token_count": len(completion_ids),
        "raw_completion": text,
        "truncated": (
            len(completion_ids) == max_completion_length
            and (not completion_ids or completion_ids[-1] != eos_token_id)
        ),
        "format_valid": isinstance(parse_completion(text), ParsedCompletion),
        "valid_answer": status
        in {RewardStatus.WRONG_ANSWER, RewardStatus.VERIFIED_PASS},
        "canonical_correct": status is RewardStatus.VERIFIED_PASS,
        "verifier_status": reward_evidence["canonical_status"],
        **reward_evidence,
    }


def _record_formal_completion(
    rows: list[dict[str, Any]],
    record: dict[str, Any],
    completion_recorder: Callable[[dict[str, Any]], None] | None,
) -> None:
    if completion_recorder is not None:
        completion_recorder(record)
    rows.append(record)


class RealFormalEvaluationBackend:
    def __init__(
        self,
        *,
        config: dict[str, Any],
        model_source: ValidatedModelSource,
        policy_adapter: Path | None,
        completion_recorder: Callable[[dict[str, Any]], None] | None = None,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not model_source.local_files_only:
            raise RuntimeError("formal evaluation requires a validated local-only snapshot")
        name = str(model_source.snapshot_path)
        self.config = config
        self.completion_recorder = completion_recorder
        self.tokenizer = AutoTokenizer.from_pretrained(name, local_files_only=True)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            name,
            local_files_only=True,
            dtype=torch.bfloat16,
            attn_implementation="eager",
        )
        if policy_adapter is not None:
            from peft import PeftModel

            model = PeftModel.from_pretrained(
                model,
                str(policy_adapter),
                local_files_only=True,
                is_trainable=False,
            )
        if not torch.cuda.is_available():
            raise RuntimeError("formal evaluation requires an authorized CUDA device")
        self.model = model.to(torch.device("cuda:0")).eval()
        self.torch = torch
        from math_rlvr.dataset import load_manifest
        from math_rlvr.training.formal_data import load_formal_data_registry

        registry = load_formal_data_registry()
        names = ("validation", "gsm8k_test", "math500_test")
        self.problems = {
            problem.problem_id: problem
            for name_value in names
            for problem in load_manifest(Path(registry["manifests"][name_value]["path"]))
        }

    def generate(self, plan):
        from math_rlvr.prompt import render_prompt_version
        from math_rlvr.rewards.formal import FORMAL_REWARD_POLICY
        from math_rlvr.verifier import MathVerifier

        verifier = MathVerifier()
        rows = []
        for item in plan:
            problem = self.problems[item["problem_id"]]
            prompt = render_prompt_version(
                self.tokenizer, problem, self.config["prompt"]["version"]
            )
            encoded = self.tokenizer(
                prompt,
                return_tensors="pt",
                add_special_tokens=False,
                truncation=False,
            )
            if int(encoded["input_ids"].shape[1]) > self.config["sampling"]["max_prompt_length"]:
                raise RuntimeError("formal evaluation prompt exceeds frozen max length")
            device = next(self.model.parameters()).device
            encoded = {name: value.to(device) for name, value in encoded.items()}
            seed_material = f"{item['seed']}::{item['pair_key']}".encode()
            generation_seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "big")
            self.torch.manual_seed(generation_seed)
            if self.torch.cuda.is_available():
                self.torch.cuda.manual_seed_all(generation_seed)
            with self.torch.inference_mode():
                output = self.model.generate(
                    **encoded,
                    do_sample=True,
                    temperature=self.config["sampling"]["temperature"],
                    top_p=self.config["sampling"]["top_p"],
                    max_new_tokens=self.config["sampling"]["max_completion_length"],
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            prompt_width = int(encoded["input_ids"].shape[1])
            completion_ids = [
                int(value) for value in output[0, prompt_width:].detach().cpu().tolist()
            ]
            text = self.tokenizer.decode(completion_ids, skip_special_tokens=True)
            evaluation = FORMAL_REWARD_POLICY.evaluate(
                text, lambda candidate, problem=problem: verifier(problem, candidate)
            )
            record = _formal_completion_record(
                item=item,
                generation_seed=generation_seed,
                completion_ids=completion_ids,
                text=text,
                max_completion_length=self.config["sampling"]["max_completion_length"],
                eos_token_id=self.tokenizer.eos_token_id,
                evaluation=evaluation,
            )
            _record_formal_completion(rows, record, self.completion_recorder)
        return rows




class _ResourceMonitoredEvaluationBackend:
    """Collect process/GPU evidence across delayed model load and generation."""

    def __init__(self, backend_factory, run_dir):
        self.backend_factory = backend_factory
        self.run_dir = run_dir
        self.rows = []
        self.resource_summary = {
            "available": False,
            "reason": "evaluation resource monitor did not start",
        }
        self.pytorch_allocator = {
            "available": False,
            "reason": "evaluation CUDA allocator did not start",
        }

    def generate(self, plan):
        import torch

        from math_rlvr.artifacts.monitor import ResourceMonitor
        from math_rlvr.training.resource_evidence import CudaAllocatorEvidence

        monitor = ResourceMonitor(self.run_dir / "resource_metrics.csv", interval=0.25)
        allocator = CudaAllocatorEvidence(torch.cuda)
        monitor.start()
        try:
            allocator.start()
            return self.backend_factory().generate(plan)
        finally:
            monitor.stop()
            self.rows = [dict(row) for row in monitor.rows]
            self.resource_summary = {"available": True, **monitor.summary()}
            self.pytorch_allocator = allocator.finalize()

    def resource_metrics(self):
        return list(self.rows)
def execute_real_formal_evaluation(
    *,
    config: dict[str, Any],
    selection: FormalEvaluationSelection,
    model_source: ValidatedModelSource,
    run_dir: Path | None,
) -> dict[str, Any]:
    from math_rlvr.artifacts.manager import ArtifactManager
    from math_rlvr.training.formal import FORMAL_MODEL
    from math_rlvr.training.formal_runtime import create_formal_backup

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    destination = run_dir or RUN_ROOT / (
        f"evaluation_formal_1p5b_{selection.mode}_seed{selection.seed}_{stamp}"
    )
    if not destination.is_absolute() or destination.parent != RUN_ROOT or destination.exists():
        raise RuntimeError("formal evaluation run directory must be a new direct child of RUN_ROOT")
    command = (
        "PYTHONPATH=src python -m math_rlvr.evaluation.formal "
        f"--phase {selection.phase} --seed {selection.seed} --mode {selection.mode} "
        "--execute --confirm-formal-evaluation"
    )
    if selection.checkpoint is not None:
        command += (
            f" --algorithm {selection.algorithm} "
            f"--checkpoint-step {selection.checkpoint_step} "
            f"--checkpoint {shlex.quote(str(selection.checkpoint))}"
        )
    manager = ArtifactManager(
        "formal_1p5b_evaluation",
        selection.algorithm or "base",
        FORMAL_MODEL,
        selection.seed,
        command,
        config,
        run_id=destination.name,
        create_checkpoints=False,
    )
    if manager.run_dir != destination:
        raise RuntimeError("ArtifactManager formal evaluation directory mismatch")
    backend = _ResourceMonitoredEvaluationBackend(
        lambda: RealFormalEvaluationBackend(
            config=config,
            model_source=model_source,
            policy_adapter=selection.policy_adapter,
            completion_recorder=lambda record: manager.append_jsonl(
                "completions.jsonl", record
            ),
        ),
        destination,
    )
    try:
        result = execute_formal_evaluation(
            backend,
            phase=selection.phase,
            seed=selection.seed,
            run_dir=destination,
            algorithm=selection.algorithm,
            checkpoint_step=selection.checkpoint_step,
            config=config,
            precreated_run_dir=True,
        )
        manifest = {
            "schema_version": 1,
            "selection": selection.to_dict(),
            "model_repo": model_source.repo_id,
            "model_revision": model_source.revision,
            "local_files_only": model_source.local_files_only,
            "prompt": config["prompt"],
            "reward": config["reward"],
            "parser": config["parser"],
            "verifier_bundle": config["verifier_bundle"],
            "resource_summary": backend.resource_summary,
            "pytorch_allocator": backend.pytorch_allocator,
            "test_driven_tuning": False,
        }
        (destination / "evaluation_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        completions = 64 if selection.phase == "validation" else 800
        manager.finalize(
            "success",
            counters={"completions": completions, "generated_tokens": None},
        )
        backup = create_formal_backup(
            destination,
            Path("/root/autodl-fs/math-rlvr-backups") / f"{destination.name}.tar.gz",
        )
        manager.write_json("backup_manifest.json", {"verified": True, **backup})
        manager.checksums()
    except Exception as exc:
        manager.finalize(
            "failure",
            failed_stage="formal_evaluation",
            exception=exc,
            stop_reason=str(exc),
        )
        backup = create_formal_backup(
            destination,
            Path("/root/autodl-fs/math-rlvr-backups") / f"{destination.name}.failure.tar.gz",
        )
        manager.write_json("backup_manifest.json", {"verified": True, **backup})
        manager.checksums()
        raise
    return {"status": "success", "run_dir": str(destination), "backup": backup, **result}


def run_checkpoint_validation(
    *,
    config: dict[str, Any],
    model_source: ValidatedModelSource,
    checkpoint: Path,
    algorithm: str,
    seed: int,
    checkpoint_step: int,
):
    from math_rlvr.evaluation.formal import load_evaluation_config

    evaluation_config = load_evaluation_config()
    backend = RealFormalEvaluationBackend(
        config=evaluation_config,
        model_source=model_source,
        policy_adapter=checkpoint / "policy_adapter",
    )
    plan = formal_evaluation_plan(
        evaluation_config,
        "validation",
        seed=seed,
        algorithm=algorithm,
        checkpoint_step=checkpoint_step,
    )
    return backend.generate(plan)
