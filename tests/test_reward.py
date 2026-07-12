import pytest

from code_rlvr.rewards.code_reward import score
from code_rlvr.rewards.result import (
    RewardInfrastructureError,
    RewardResult,
    RewardStatus,
)


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (RewardStatus.VERIFIED_PASS, 1.0),
        (RewardStatus.VERIFIED_FAIL, 0.0),
        (RewardStatus.FORMAT_ERROR, -0.2),
        (RewardStatus.TIMEOUT, -0.5),
        (RewardStatus.RESOURCE_LIMIT, -0.5),
    ],
)
def test_valid_outcomes_have_explicit_scores(status: RewardStatus, expected: float) -> None:
    assert score(RewardResult(status)) == expected


@pytest.mark.parametrize(
    "status", [RewardStatus.SANDBOX_UNAVAILABLE, RewardStatus.INFRA_ERROR]
)
def test_infrastructure_outcomes_abort(status: RewardStatus) -> None:
    with pytest.raises(RewardInfrastructureError, match=status.value):
        score(RewardResult(status, "offline"))
