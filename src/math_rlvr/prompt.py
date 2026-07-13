from math_rlvr.dataset import MathProblem

SYSTEM_PROMPT = (
    "Solve the math problem. Output exactly one <reasoning>...</reasoning> block "
    "followed by exactly one <answer>...</answer> block. Put only the final answer "
    "in <answer>."
)


def format_problem(problem: MathProblem) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": problem.prompt},
    ]


def render_prompt(tokenizer, problem: MathProblem) -> str:
    """Render the shared PPO/GRPO prompt with an open assistant turn."""
    return tokenizer.apply_chat_template(
        format_problem(problem), tokenize=False, add_generation_prompt=True
    )
