"""Algorithm-neutral aggregation for verifier outputs."""

from dataclasses import dataclass

from math_rlvr.rewards.result import DEFAULT_REWARD_POLICY, RewardResult, RewardStatus


@dataclass(frozen=True)
class EvaluationSummary:
    attempted: int
    syntax_rate: float
    pass_at_1: float


def summarize(results: list[RewardResult]) -> EvaluationSummary:
    if not results:
        return EvaluationSummary(0, 0.0, 0.0)
    for result in results:
        DEFAULT_REWARD_POLICY.to_scalar(result)
    count = len(results)
    return EvaluationSummary(
        attempted=count,
        syntax_rate=sum(item.status is not RewardStatus.FORMAT_ERROR for item in results) / count,
        pass_at_1=sum(item.status is RewardStatus.VERIFIED_PASS for item in results) / count,
    )
