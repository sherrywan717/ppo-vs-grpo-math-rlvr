import hashlib
import json
from pathlib import Path

import pytest
import torch
from transformers import AutoTokenizer

from math_rlvr.dataset import content_hash, load_manifest
from math_rlvr.parser import parse_completion
from math_rlvr.prompt import (
    PROMPT_V0_GRPO_SMOKE,
    PROMPT_V1_STRICT_CONCISE,
    SYSTEM_PROMPT,
    format_problem,
    format_problem_v1,
    render_candidate_prompt,
    render_prompt,
)
from math_rlvr.prompt_forensics import audit_rendered_prompt, classify_completion
from math_rlvr.rewards.result import RewardResult, RewardStatus
from math_rlvr.training.grpo import render_candidate_training_prompt as grpo_v1_renderer
from math_rlvr.training.ppo import render_candidate_training_prompt as ppo_v1_renderer

SNAPSHOT = Path(
    "/root/autodl-tmp/cache/huggingface/hub/"
    "models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/"
    "7ae557604adf67be50417f59c2c2f167def9a775"
)
MANIFEST_ROOT = Path("/root/autodl-tmp/datasets/math_rlvr/manifests")
RUN_REPORT = Path(
    "reports/runs/grpo_single_update_qwen25_05b_20260713T063829Z"
)


@pytest.fixture(scope="module")
def tokenizer():
    tok = AutoTokenizer.from_pretrained(SNAPSHOT, local_files_only=True)
    tok.padding_side = "left"
    return tok


@pytest.fixture(scope="module")
def smoke_problems():
    problems = load_manifest(MANIFEST_ROOT / "countdown_train.json")[:2]
    assert [p.problem_id for p in problems] == ["countdown:train:0", "countdown:train:1"]
    return problems


def test_v0_messages_and_runtime_hashes_reconstruct_exactly(tokenizer, smoke_problems):
    saved = json.loads((RUN_REPORT / "smoke_problems.json").read_text())
    for problem, row in zip(smoke_problems, saved, strict=True):
        assert format_problem(problem) == [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": problem.prompt},
        ]
        audit = audit_rendered_prompt(tokenizer, problem, PROMPT_V0_GRPO_SMOKE)
        assert audit["prompt_hash"] == row["prompt_hash"] == content_hash(problem.prompt)
        assert audit["role_sequence"] == ["system", "user"]
        assert audit["generation_boundary_correct"] is True
        assert audit["add_generation_prompt"] is True
        assert audit["template_application_count"] == 1
        assert audit["prompt_truncated"] is False


def test_v1_strict_protocol_is_last_and_complete(tokenizer, smoke_problems):
    for problem in smoke_problems:
        messages = format_problem_v1(problem)
        user = messages[-1]["content"]
        assert user.startswith(problem.prompt + "\n\n")
        assert user.endswith(
            "Use every input number exactly once. "
            "Use only +, -, *, /, and parentheses."
        )
        for text in (
            "<reasoning>...</reasoning>",
            "<answer>...</answer>",
            "no text outside",
            "two closed blocks",
            "an equals sign",
            "every input number exactly once",
        ):
            assert text in user
        assert problem.gold_answer not in user
        audit = audit_rendered_prompt(tokenizer, problem, PROMPT_V1_STRICT_CONCISE)
        assert audit["reasoning_tag_requirement_present"]
        assert audit["answer_tag_requirement_present"]
        assert audit["generation_boundary_correct"]
        assert not audit["prompt_truncated"]
        assert audit["format_instruction_distance_tokens"] < 16


def test_v1_candidate_is_shared_by_ppo_and_grpo(tokenizer, smoke_problems):
    assert ppo_v1_renderer is grpo_v1_renderer is render_candidate_prompt
    for problem in smoke_problems:
        assert ppo_v1_renderer(tokenizer, problem) == grpo_v1_renderer(tokenizer, problem)


def test_v0_and_v1_are_one_pass_chat_template_renders(tokenizer, smoke_problems):
    for problem in smoke_problems:
        v0 = audit_rendered_prompt(tokenizer, problem, PROMPT_V0_GRPO_SMOKE)
        v1 = audit_rendered_prompt(tokenizer, problem, PROMPT_V1_STRICT_CONCISE)
        assert v0["rendered_text"] == render_prompt(tokenizer, problem)
        assert v1["rendered_text"] == render_candidate_prompt(tokenizer, problem)
        assert v0["rendered_text"].count("<|im_start|>assistant") == 1
        assert v1["rendered_text"].count("<|im_start|>assistant") == 1


def test_countdown_gsm8k_math_all_render_without_truncation(tokenizer):
    countdown = load_manifest(MANIFEST_ROOT / "countdown_train.json")[0]
    training = load_manifest(MANIFEST_ROOT / "train_core_128.json")
    gsm = next(p for p in training if p.source == "gsm8k")
    math = next(p for p in training if p.source == "math")
    for problem in (countdown, gsm, math):
        audit = audit_rendered_prompt(tokenizer, problem, PROMPT_V1_STRICT_CONCISE)
        assert audit["role_sequence"] == ["system", "user"]
        assert audit["generation_boundary_correct"]
        assert audit["runtime_token_count"] <= 512
        assert not audit["prompt_truncated"]


def test_historical_completions_replay_as_same_format_errors(smoke_problems):
    problem_map = {p.problem_id: p for p in smoke_problems}
    completion_lines = (RUN_REPORT / "completions.jsonl").read_text().splitlines()
    rows = [json.loads(line) for line in completion_lines]
    audits = [classify_completion(row, problem_map[row["problem_id"]]) for row in rows]
    assert len(audits) == 8
    assert all(row["runtime_replay_consistent"] for row in audits)
    assert all(row["parser_status"] == RewardStatus.FORMAT_ERROR.value for row in audits)
    assert all(row["verifier_status"] == RewardStatus.FORMAT_ERROR.value for row in audits)
    assert sum(row["at_128_token_cap"] for row in audits) == 4


def test_strict_parser_behavior_is_unchanged():
    bad = [
        "<answer>1</answer>",
        "outside<reasoning>x</reasoning><answer>1</answer>",
        "<reasoning>x</reasoning><answer>1</answer>outside",
    ]
    for text in bad:
        result = parse_completion(text)
        assert isinstance(result, RewardResult) and result.status == RewardStatus.FORMAT_ERROR


def test_frozen_yaml_hashes_and_cpu_cuda_state():
    expected = {
        "configs/smoke/grpo.yaml": (  # noqa: E501
            "068ff8d742849ffa0d43ccf6f4e74898e08c5f031c0f837c18ac8e5b183d8979"
        ),
        "configs/smoke/ppo.yaml": (  # noqa: E501
            "1496c65309befbcf4c5143b5d19e963013a9c869ff4af4e82b838abc317a0379"
        ),
    }
    for name, digest in expected.items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest
    assert torch.cuda.is_initialized() is False


def test_v1_smoke_activation_identity_and_shared_renderer(tokenizer, smoke_problems):
    from math_rlvr.prompt import (
        PROMPT_RENDERER_VERSION,
        PROMPT_V0_SHA256,
        PROMPT_V1_SHA256,
        prompt_metadata,
    )
    from math_rlvr.training.common import preflight
    from math_rlvr.training.grpo import render_training_prompt as grpo_smoke_renderer
    from math_rlvr.training.ppo import render_training_prompt as ppo_smoke_renderer

    assert PROMPT_V0_SHA256 == "20b54a2ae00ebc762a1a90a3221f5c2409c7e64d2b35fcf2c6dfaaff48a9ef4f"
    assert PROMPT_V1_SHA256 == "6842002e4591630a4105a4ca8fdf4cab91676b3902708ec0cb7f7b458864ecd7"
    assert prompt_metadata(PROMPT_V1_STRICT_CONCISE) == {
        "prompt_version": PROMPT_V1_STRICT_CONCISE,
        "prompt_sha256": PROMPT_V1_SHA256,
        "renderer_version": PROMPT_RENDERER_VERSION,
        "candidate_status": "approved_for_smoke",
        "production_status": "not_approved",
    }
    grpo = preflight(Path("configs/smoke/grpo.yaml"), "grpo")
    ppo = preflight(Path("configs/smoke/ppo.yaml"), "ppo")
    for key in ("prompt_version", "prompt_sha256", "renderer_version"):
        assert grpo[key] == ppo[key]
    for problem in smoke_problems:
        assert grpo_smoke_renderer(tokenizer, problem, grpo) == ppo_smoke_renderer(
            tokenizer, problem, ppo
        )


def test_smoke_yaml_authorized_selectors_only_and_main_is_unactivated():
    before_prompt_activation = {
        "configs/smoke/grpo.yaml": "3e6ea0f568c7d946a3023eb14b67988751e37b1cb692b52018faa9dbb622a398",  # noqa: E501
        "configs/smoke/ppo.yaml": "1db287f772f11da9fb6e69a304857b0055dde2bb0b74baec3bfb07d0d7f0b820",  # noqa: E501
    }
    after_prompt_activation = {
        "configs/smoke/grpo.yaml": "5df5d72f71ada14a6ce903990b1b21bbd9d682ba8a05b1f77a91bc974c3872e0",  # noqa: E501
        "configs/smoke/ppo.yaml": "b888b12fb56fe356633b2d04f2c9713bb8d02c13be66fe349f60b5d40cbc1ee3",  # noqa: E501
    }
    after_staged_reward = {
        "configs/smoke/grpo.yaml": "068ff8d742849ffa0d43ccf6f4e74898e08c5f031c0f837c18ac8e5b183d8979",  # noqa: E501
        "configs/smoke/ppo.yaml": "1496c65309befbcf4c5143b5d19e963013a9c869ff4af4e82b838abc317a0379",  # noqa: E501
    }
    for name, digest in after_staged_reward.items():
        raw = Path(name).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == digest
        without_reward = b"\n".join(
            line for line in raw.split(b"\n") if not line.startswith(b"reward:")
        )
        assert hashlib.sha256(without_reward).hexdigest() == after_prompt_activation[name]
        without_both_selectors = b"\n".join(
            line
            for line in raw.split(b"\n")
            if not line.startswith((b"prompt:", b"reward:"))
        )
        assert (
            hashlib.sha256(without_both_selectors).hexdigest()
            == before_prompt_activation[name]
        )
    for name in ("configs/main/grpo.yaml", "configs/main/ppo.yaml"):
        text = Path(name).read_text(encoding="utf-8")
        assert "prompt:" not in text and "reward:" not in text


def test_successful_ab_history_is_immutable():
    expected = {
        "reports/runs/prompt_ab_qwen25_05b_20260713T105428Z/summary.json": "3d58de03a6b0724b290e4f22bc7efb76e888b5b77b97197e03ea4c15a75f1faa",  # noqa: E501
        "reports/runs/prompt_ab_qwen25_05b_20260713T105428Z/completions.jsonl": "2aff8ada1b6ab022579e39c4d3c2914e30229d8eb3782155ecbce8a7d9b079b1",  # noqa: E501
        "reports/runs/grpo_single_update_qwen25_05b_20260713T063829Z/summary.json": "39c40a0f87ebd069a7f0757e7bba3cac8ae58e93549f41bd681c7ba02e1b6e09",  # noqa: E501
    }
    for name, digest in expected.items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest
