"""Versioned, shared prompt renderers for PPO and GRPO."""

from math_rlvr.dataset import MathProblem

PROMPT_V0_GRPO_SMOKE = "prompt_v0_grpo_smoke"
PROMPT_V1_STRICT_CONCISE = "prompt_v1_strict_concise"

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
        format_problem_version(problem, version),
        tokenize=False,
        add_generation_prompt=True,
    )


def format_problem(problem: MathProblem) -> list[dict[str, str]]:
    """Production v0 message renderer retained byte-for-byte for historical replay."""
    return format_problem_v0(problem)


def render_prompt(tokenizer, problem: MathProblem) -> str:
    """Render the production v0 PPO/GRPO prompt with an open assistant turn."""
    return render_prompt_version(tokenizer, problem, PROMPT_V0_GRPO_SMOKE)


def render_candidate_prompt(tokenizer, problem: MathProblem) -> str:
    """Render the unactivated v1 forensic candidate; never selected by training config."""
    return render_prompt_version(tokenizer, problem, PROMPT_V1_STRICT_CONCISE)
