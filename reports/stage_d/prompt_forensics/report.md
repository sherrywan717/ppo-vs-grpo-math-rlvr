# GRPO prompt/chat-template forensic audit

- Result: PASS; CPU-only and offline; no model, generation, training, optimizer, GRPO, or PPO.
- Historical replay: 8/8 remain `format_error`; no parser misclassification evidence.
- Primary taxonomy: {'missing_reasoning_open': 1, 'answer_only': 3, 'truncated_at_max_tokens': 4}.
- Completion cap: 4/8 reached 128 tokens.
- Tag coverage: reasoning 0/8, answer pair 3/8, complete envelope 0/8.
- v0 prompt tokens: [85, 83]; v1: [157, 155].
- Format-instruction distance — v0: [43, 41]; v1: [5, 5].
- Prompt truncation: none. Qwen roles, one-pass template, and open assistant boundary are correct.
- PAD/BOS/EOS warning did not alter rendered text or the generation boundary.
- Root cause: v0 contains the literal tag requirement, but it is in the system message before the problem and omits explicit no-outside-text, closure, expression-only, no-equals/target, and concise-reasoning constraints. The 0.5B model showed weak adherence; the 128-token cap was secondary for four outputs.
- Recommendation: retain production v0 and test unactivated `prompt_v1_strict_concise` only in a separately authorized generation-only A/B diagnostic.
- Parser remains unchanged and strict.
- Saved `prompt_hash` is the normalized dataset user-text hash; both reconstructed hashes match. Rendered-chat SHA256 values are recorded separately.
- Tokenizer state: BOS `None`, EOS `<|im_end|>` (151645), PAD `<|endoftext|>` (151643), left padding, right truncation.
- No v2 one-shot candidate was added: v1 already supplies the literal envelope, and an example should wait for A/B evidence.
