"""Compatibility export for the canonical scalar reward policy."""

from math_rlvr.rewards.result import DEFAULT_REWARD_POLICY, RewardResult


def score(result: RewardResult) -> float:
    return DEFAULT_REWARD_POLICY.to_scalar(result)
