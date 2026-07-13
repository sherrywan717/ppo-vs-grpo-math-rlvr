import pytest

from math_rlvr.rewards.code_reward import score
from math_rlvr.rewards.result import (
    RewardInfrastructureError,
    RewardResult,
    RewardStatus,
)


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (RewardStatus.VERIFIED_PASS, 1.0),
        (RewardStatus.WRONG_ANSWER, 0.2),
        (RewardStatus.FORMAT_ERROR, 0.0),
        (RewardStatus.RESOURCE_LIMIT, 0.1),
        (RewardStatus.RESOURCE_LIMIT, 0.1),
    ],
)
def test_valid_outcomes_have_explicit_scores(status: RewardStatus, expected: float) -> None:
    assert score(RewardResult(status)) == expected


@pytest.mark.parametrize("status", [RewardStatus.INFRA_ERROR, RewardStatus.INFRA_ERROR])
def test_infrastructure_outcomes_abort(status: RewardStatus) -> None:
    with pytest.raises(RewardInfrastructureError, match=status.value):
        score(RewardResult(status, "offline"))
