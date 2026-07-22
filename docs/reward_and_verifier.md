# Reward and verifier

The scalar training reward is intentionally separated from canonical evaluation.

| Component | Weight | Meaning |
|---|---:|---|
| Answer block | 0.05 | A usable terminal answer block exists |
| Strict protocol | 0.05 | The required reasoning/answer envelope is respected |
| Domain-valid answer | 0.10 | The answer parses under the task's safe domain rules |
| Canonical correctness | 0.80 | The unchanged canonical verifier returns `VERIFIED_PASS` |

The parser accepts model output as text only. It never executes generated content. Arithmetic verification uses AST/Fraction-based evaluation and `math-verify`; `eval`, `exec`, dynamic imports, subprocesses, and generated-code execution are prohibited. Infrastructure failures abort rather than becoming reward zero.

`format_valid`, `parseable`, `valid_answer_component`, `canonical_pass`, and `accuracy_given_parseable` have explicit numerators/denominators in the metric-definition JSON. A zero denominator yields null/unavailable/reason, not a fabricated zero. `valid_answer_rate` is a reporting alias derived from canonical flat evidence and does not enter reward, advantages, losses, optimizer, scheduler, stopping, or checkpoint selection.

Reward shaping can provide gradient signal even when canonical pass remains low. Reports therefore show both scalar reward and canonical RewardStatus, plus GRPO within-group reward variance and zero-advantage rates, to distinguish optimization signal from mathematical success or output-mode contraction.
