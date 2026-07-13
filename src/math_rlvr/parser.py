"""Strict reasoning/answer envelope parser."""

import re
from dataclasses import dataclass

from math_rlvr.rewards.result import RewardResult, RewardStatus

PATTERN = re.compile(
    r"\A<reasoning>(?P<reasoning>.*?)</reasoning>\s*<answer>(?P<answer>.*?)</answer>\s*\Z", re.S
)


@dataclass(frozen=True)
class ParsedCompletion:
    reasoning: str
    answer: str


def parse_completion(text: str, max_answer_length: int = 512):
    if any(
        text.count(tag) != 1 for tag in ("<reasoning>", "</reasoning>", "<answer>", "</answer>")
    ):
        return RewardResult(RewardStatus.FORMAT_ERROR, "each tag must appear exactly once")
    match = PATTERN.fullmatch(text)
    if not match:
        return RewardResult(RewardStatus.FORMAT_ERROR, "invalid envelope or trailing output")
    answer = match["answer"].strip()
    if not answer or len(answer) > max_answer_length:
        return RewardResult(RewardStatus.FORMAT_ERROR, "empty or long answer")
    return ParsedCompletion(match["reasoning"].strip(), answer)
