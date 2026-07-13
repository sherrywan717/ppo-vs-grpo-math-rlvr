"""Conservative normalization for trusted dataset gold answers only."""

from math_verify import parse


def normalize_gold_answer(gold: str) -> str:
    """Return a parseable gold, adding only missing LaTeX math delimiters."""
    stripped = gold.strip()
    if not stripped or len(stripped) > 512:
        raise ValueError("empty or oversized gold")
    if parse(stripped):
        return stripped
    delimited = f"${stripped}$"
    if parse(delimited):
        return delimited
    raise ValueError("gold is not stably parseable")
