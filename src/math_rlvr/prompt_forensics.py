"""CPU-only prompt and completion forensic helpers."""

from __future__ import annotations

import hashlib
from dataclasses import asdict

from math_rlvr.dataset import MathProblem, content_hash
from math_rlvr.parser import ParsedCompletion, parse_completion
from math_rlvr.prompt import (
    PROMPT_V0_GRPO_SMOKE,
    format_problem_version,
    render_prompt_version,
)
from math_rlvr.rewards.result import RewardResult
from math_rlvr.verifier import MathVerifier

TAGS = ("<reasoning>", "</reasoning>", "<answer>", "</answer>")


def classify_completion(
    row: dict, problem: MathProblem, max_completion_length: int = 128
) -> dict:
    text = row["raw_completion"]
    detected = {tag: text.count(tag) for tag in TAGS}
    parsed = parse_completion(text)
    verified = MathVerifier()(problem, text)
    categories = []
    missing_names = {
        "<reasoning>": "missing_reasoning_open",
        "</reasoning>": "missing_reasoning_close",
        "<answer>": "missing_answer_open",
        "</answer>": "missing_answer_close",
    }
    for tag, name in missing_names.items():
        if detected[tag] == 0:
            categories.append(name)
    answer_pair = detected["<answer>"] == detected["</answer>"] == 1
    no_reasoning = detected["<reasoning>"] == detected["</reasoning>"] == 0
    if answer_pair and no_reasoning:
        categories.append("answer_only")
    if isinstance(parsed, RewardResult):
        stripped = text
        for tag in TAGS:
            stripped = stripped.replace(tag, "")
        if stripped.strip() and (not answer_pair or not text.startswith("<answer>")):
            categories.append("prose_outside_envelope")
    if isinstance(parsed, ParsedCompletion) and verified.status.value == "invalid_expression":
        categories.append("malformed_expression")
    at_cap = row["exact_token_count"] == max_completion_length
    if at_cap:
        categories.append("truncated_at_max_tokens")
    if isinstance(parsed, ParsedCompletion):
        categories.append("complete_valid_envelope")
    if not categories:
        categories.append("other")
    if "complete_valid_envelope" in categories:
        primary = "complete_valid_envelope"
    elif "truncated_at_max_tokens" in categories:
        primary = "truncated_at_max_tokens"
    elif "answer_only" in categories:
        primary = "answer_only"
    else:
        primary = categories[0]
    parser_status = "parsed" if isinstance(parsed, ParsedCompletion) else parsed.status.value
    parser_detail = None if isinstance(parsed, ParsedCompletion) else parsed.detail
    return {
        "problem_id": row["problem_id"],
        "prompt_hash": row["prompt_hash"],
        "generation_index": row["generation_index"],
        "completion_index": row["completion_index"],
        "raw_completion": text,
        "exact_token_count": row["exact_token_count"],
        "at_128_token_cap": at_cap,
        "detected_tags": detected,
        "categories": categories,
        "primary_class": primary,
        "parser_status": parser_status,
        "parser_detail": parser_detail,
        "verifier_status": verified.status.value,
        "verifier_detail": verified.detail,
        "runtime_reward_status": row["reward_status"],
        "runtime_replay_consistent": verified.status.value == row["reward_status"],
    }


def audit_rendered_prompt(
    tokenizer,
    problem: MathProblem,
    version: str = PROMPT_V0_GRPO_SMOKE,
    max_prompt_length: int = 512,
) -> dict:
    messages = format_problem_version(problem, version)
    rendered = render_prompt_version(tokenizer, problem, version)
    direct = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    if rendered != direct:
        raise RuntimeError("renderer differs from one-pass apply_chat_template")
    tokenized = tokenizer(
        rendered,
        add_special_tokens=False,
        truncation=False,
    )["input_ids"]
    runtime_ids = tokenizer(
        rendered,
        add_special_tokens=False,
        truncation=True,
        max_length=max_prompt_length,
        padding=False,
    )["input_ids"]
    instruction_message = messages[0] if version == PROMPT_V0_GRPO_SMOKE else messages[-1]
    marker = instruction_message["content"]
    instruction_end = rendered.index(marker) + len(marker)
    prefix_ids = tokenizer(
        rendered[:instruction_end], add_special_tokens=False, truncation=False
    )["input_ids"]
    distance = len(tokenized) - len(prefix_ids)
    generation_boundary = "<|im_start|>assistant\n"
    source_hash = content_hash(problem.prompt)
    if source_hash != problem.content_hash:
        raise RuntimeError("source prompt hash mismatch")
    return {
        "version": version,
        "problem_id": problem.problem_id,
        "source": problem.source,
        "split": problem.split,
        "prompt_hash": source_hash,
        "rendered_sha256": hashlib.sha256(rendered.encode()).hexdigest(),
        "structured_messages": messages,
        "system_text": messages[0]["content"],
        "user_text": messages[-1]["content"],
        "rendered_text": rendered,
        "token_ids": tokenized,
        "token_count": len(tokenized),
        "runtime_token_ids": runtime_ids,
        "runtime_token_count": len(runtime_ids),
        "last_128_tokens_decode": tokenizer.decode(tokenized[-128:]),
        "format_instruction_distance_tokens": distance,
        "prompt_truncated": len(runtime_ids) != len(tokenized),
        "generation_boundary": generation_boundary,
        "generation_boundary_correct": rendered.endswith(generation_boundary),
        "add_generation_prompt": True,
        "role_sequence": [message["role"] for message in messages],
        "template_application_count": 1,
        "reasoning_tag_requirement_present": all(
            tag in rendered for tag in ("<reasoning>", "</reasoning>")
        ),
        "answer_tag_requirement_present": all(
            tag in rendered for tag in ("<answer>", "</answer>")
        ),
        "bos_token": tokenizer.bos_token,
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token": tokenizer.eos_token,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token": tokenizer.pad_token,
        "pad_token_id": tokenizer.pad_token_id,
        "padding_side": tokenizer.padding_side,
        "truncation_side": tokenizer.truncation_side,
        "problem": asdict(problem),
    }
