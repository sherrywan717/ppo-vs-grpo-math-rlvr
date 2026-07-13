"""Conservative normalization for trusted dataset gold answers only."""

from math_verify import parse


def assert_delimiter_only(raw_gold: str, normalized_gold: str) -> None:
    if normalized_gold == raw_gold:
        return
    if not (normalized_gold.startswith("$") and normalized_gold.endswith("$")):
        raise ValueError("normalization may only add math delimiters")
    if normalized_gold[1:-1] != raw_gold:
        raise ValueError("normalization changed expression characters")


def normalize_gold_answer(gold: str) -> str:
    """Add exactly one pair of math delimiters only when required for trusted gold."""
    if not gold.strip() or len(gold) > 512:
        raise ValueError("empty or oversized gold")
    if parse(gold):
        return gold
    normalized = f"${gold}$"
    assert_delimiter_only(gold, normalized)
    if parse(normalized):
        return normalized
    raise ValueError("gold is not stably parseable")
