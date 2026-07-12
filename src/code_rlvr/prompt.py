"""One prompt format shared by PPO, GRPO, and evaluation."""

from code_rlvr.dataset import CodeProblem

SYSTEM_PROMPT = (
    "You are an expert Python programmer. Return only a complete solution in one Python "
    "code block. Do not access the network or filesystem."
)


def format_problem(problem: CodeProblem) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": problem.prompt},
    ]

