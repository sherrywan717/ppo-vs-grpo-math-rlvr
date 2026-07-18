# Frozen baseline error analysis

The analysis uses only the two successful post-amendment runs and does not alter the prompt, reward, verifier, evaluation set, or sampling protocol.

## Seed 42

Of 800 completions, 692 were strict format failures, 12 were parse failures after clearing format, 67 were parseable canonical wrong answers, and 29 were canonical passes.
GSM8K sampled pass@1 was 0.035; MATH500 was 0.045. Truncated completions were 80/800 with canonical correctness 0.0000; non-truncated correctness was 0.0403.

## Seed 123

Of 800 completions, 706 were strict format failures, 5 were parse failures after clearing format, 72 were parseable canonical wrong answers, and 17 were canonical passes.
GSM8K sampled pass@1 was 0.015; MATH500 was 0.035. Truncated completions were 62/800 with canonical correctness 0.0000; non-truncated correctness was 0.0230.

## Interpretation

The dominant failure mode is strict output-format failure, not a verifier infrastructure error. The smaller parse-error group is kept separate from outputs that parse but produce a wrong mathematical answer. MATH500 level results are non-monotonic at this low baseline accuracy, and two seeds are insufficient to interpret small level-to-level differences as a stable capability curve.
Sampled pass@1 changed from 0.040 to 0.025; pass@4 changed from 0.100 to 0.060. This is disclosed seed variation, not a prompt-selection or checkpoint-selection signal. Truncation is material and is reported rather than silently treated as an ordinary full completion.
