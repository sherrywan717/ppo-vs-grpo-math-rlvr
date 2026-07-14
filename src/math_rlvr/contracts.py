"""Stable parser/verifier contract identities from canonical serialization."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_contract_bytes(descriptor: dict[str, Any]) -> bytes:
    """Serialize a semantic descriptor without comments or presentation metadata."""
    return json.dumps(
        descriptor, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def contract_sha256(descriptor: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_contract_bytes(descriptor)).hexdigest()


PARSER_CONTRACT_VERSION = "strict_completion_parser_v1"
PARSER_CONTRACT_DESCRIPTOR = {
    "contract_version": PARSER_CONTRACT_VERSION,
    "envelope": ["reasoning", "answer"],
    "anchoring": "full_string",
    "tag_cardinality": "each_open_and_close_tag_exactly_once",
    "inter_block_whitespace": "allowed",
    "trailing_non_whitespace": "forbidden",
    "reasoning_normalization": "strip_outer_whitespace",
    "answer_normalization": "strip_outer_whitespace",
    "answer_empty": "format_error",
    "default_max_answer_characters": 512,
    "overlong_answer": "format_error",
}
PARSER_CONTRACT_SHA256 = contract_sha256(PARSER_CONTRACT_DESCRIPTOR)

COUNTDOWN_VERIFIER_CONTRACT_VERSION = "countdown_ast_fraction_v1"
COUNTDOWN_VERIFIER_CONTRACT_DESCRIPTOR = {
    "contract_version": COUNTDOWN_VERIFIER_CONTRACT_VERSION,
    "parser_contract_version": PARSER_CONTRACT_VERSION,
    "parser_contract_sha256": PARSER_CONTRACT_SHA256,
    "expression_parser": "python_ast_eval_mode_parse_only",
    "generated_code_execution": False,
    "default_max_ast_nodes": 64,
    "default_max_answer_characters": 512,
    "allowed_ast_nodes": [
        "Expression",
        "BinOp",
        "UnaryOp",
        "Constant",
        "Add",
        "Sub",
        "Mult",
        "Div",
        "UAdd",
        "USub",
        "Load",
    ],
    "constant_rule": "builtin_nonnegative_integer_only",
    "number_usage": "Counter(constants)_equals_Counter(input_numbers)",
    "arithmetic": "fractions.Fraction_exact",
    "division_by_zero": "invalid_expression",
    "target_comparison": "exact_fraction_equality",
    "statuses": {
        "syntax_or_forbidden_ast": "invalid_expression",
        "too_many_ast_nodes": "resource_limit",
        "number_multiset_mismatch": "invalid_number_usage",
        "target_equal": "verified_pass",
        "target_unequal": "wrong_answer",
    },
}
COUNTDOWN_VERIFIER_CONTRACT_SHA256 = contract_sha256(
    COUNTDOWN_VERIFIER_CONTRACT_DESCRIPTOR
)


def parser_verifier_metadata() -> dict[str, dict[str, str]]:
    return {
        "parser_contract": {
            "contract_version": PARSER_CONTRACT_VERSION,
            "contract_sha256": PARSER_CONTRACT_SHA256,
        },
        "verifier_contract": {
            "contract_version": COUNTDOWN_VERIFIER_CONTRACT_VERSION,
            "contract_sha256": COUNTDOWN_VERIFIER_CONTRACT_SHA256,
        },
    }
