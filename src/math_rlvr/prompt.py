"""Versioned, shared prompt renderers for PPO and GRPO."""

import hashlib
import json
from typing import Any

from math_rlvr.dataset import MathProblem

PROMPT_V0_GRPO_SMOKE = "prompt_v0_grpo_smoke"
PROMPT_V1_STRICT_CONCISE = "prompt_v1_strict_concise"
PROMPT_RENDERER_VERSION = "math_rlvr.prompt.chat_template.v1"

SYSTEM_PROMPT = (
    "Solve the math problem. Output exactly one <reasoning>...</reasoning> block "
    "followed by exactly one <answer>...</answer> block. Put only the final answer "
    "in <answer>."
)
SYSTEM_PROMPT_V1 = (
    "Solve the math problem concisely. Follow the output protocol at the end of the "
    "user message exactly."
)
_COUNTDOWN_V1_PROTOCOL = """OUTPUT PROTOCOL — follow exactly:
<reasoning>...</reasoning>
<answer>...</answer>
Return exactly those two closed blocks, in that order, with no text outside them.
Keep reasoning to at most two short sentences.
Inside <answer>, write one arithmetic expression only.
Do not include prose, an equals sign, or the target value.
Use every input number exactly once. Use only +, -, *, /, and parentheses."""
_GENERAL_V1_PROTOCOL = """OUTPUT PROTOCOL — follow exactly:
<reasoning>...</reasoning>
<answer>...</answer>
Return exactly those two closed blocks, in that order, with no text outside them.
Keep reasoning to at most two short sentences.
Inside <answer>, write only the final answer, with no prose and no equals sign."""


def _prompt_spec(version: str) -> dict[str, Any]:
    if version == PROMPT_V0_GRPO_SMOKE:
        system, protocols = SYSTEM_PROMPT, {}
    elif version == PROMPT_V1_STRICT_CONCISE:
        system, protocols = SYSTEM_PROMPT_V1, {
            "countdown": _COUNTDOWN_V1_PROTOCOL,
            "default": _GENERAL_V1_PROTOCOL,
        }
    else:
        raise ValueError(f"unknown prompt version: {version}")
    return {
        "version": version,
        "renderer_version": PROMPT_RENDERER_VERSION,
        "roles": ["system", "user"],
        "add_generation_prompt": True,
        "system": system,
        "user_protocol_by_source": protocols,
    }


def prompt_spec_sha256(version: str) -> str:
    canonical = json.dumps(
        _prompt_spec(version), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


PROMPT_V0_SHA256 = prompt_spec_sha256(PROMPT_V0_GRPO_SMOKE)
PROMPT_V1_SHA256 = prompt_spec_sha256(PROMPT_V1_STRICT_CONCISE)
PROMPT_APPROVAL_STATUS = {
    PROMPT_V0_GRPO_SMOKE: {
        "candidate_status": "historical_baseline",
        "production_status": "not_approved",
    },
    PROMPT_V1_STRICT_CONCISE: {
        "candidate_status": "approved_for_smoke",
        "production_status": "not_approved",
    },
}


def prompt_metadata(version: str) -> dict[str, str]:
    return {
        "prompt_version": version,
        "prompt_sha256": prompt_spec_sha256(version),
        "renderer_version": PROMPT_RENDERER_VERSION,
        **PROMPT_APPROVAL_STATUS[version],
    }


def prompt_version_from_config(config: dict[str, Any] | None) -> str:
    if config is None:
        return PROMPT_V0_GRPO_SMOKE
    selected = config.get("prompt", {}).get("version")
    is_smoke = str(config.get("experiment", {}).get("name", "")).startswith("smoke-")
    if is_smoke:
        if selected != PROMPT_V1_STRICT_CONCISE:
            raise ValueError("0.5B smoke requires approved strict-concise prompt")
        return selected
    if selected is not None:
        raise ValueError("main/formal configs must not activate a smoke prompt")
    return PROMPT_V0_GRPO_SMOKE


def format_problem_v0(problem: MathProblem) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": problem.prompt},
    ]


def format_problem_v1(problem: MathProblem) -> list[dict[str, str]]:
    protocol = _COUNTDOWN_V1_PROTOCOL if problem.source == "countdown" else _GENERAL_V1_PROTOCOL
    return [
        {"role": "system", "content": SYSTEM_PROMPT_V1},
        {"role": "user", "content": f"{problem.prompt}\n\n{protocol}"},
    ]


def format_problem_version(problem: MathProblem, version: str) -> list[dict[str, str]]:
    if version == PROMPT_V0_GRPO_SMOKE:
        return format_problem_v0(problem)
    if version == PROMPT_V1_STRICT_CONCISE:
        return format_problem_v1(problem)
    raise ValueError(f"unknown prompt version: {version}")


def render_prompt_version(tokenizer, problem: MathProblem, version: str) -> str:
    return tokenizer.apply_chat_template(
        format_problem_version(problem, version), tokenize=False, add_generation_prompt=True
    )


def format_training_problem(
    problem: MathProblem, config: dict[str, Any] | None = None
) -> list[dict[str, str]]:
    return format_problem_version(problem, prompt_version_from_config(config))


def render_training_prompt(
    tokenizer, problem: MathProblem, config: dict[str, Any] | None = None
) -> str:
    return render_prompt_version(tokenizer, problem, prompt_version_from_config(config))


def format_problem(problem: MathProblem) -> list[dict[str, str]]:
    """Historical v0 renderer retained byte-for-byte for replay."""
    return format_problem_v0(problem)


def render_prompt(tokenizer, problem: MathProblem) -> str:
    return render_prompt_version(tokenizer, problem, PROMPT_V0_GRPO_SMOKE)


def render_candidate_prompt(tokenizer, problem: MathProblem) -> str:
    return render_prompt_version(tokenizer, problem, PROMPT_V1_STRICT_CONCISE)
