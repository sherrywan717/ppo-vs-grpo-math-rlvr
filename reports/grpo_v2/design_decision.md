# Single-seed GRPO-v2 design decision

GRPO-v2 freezes seed 42, a 256-example one-epoch format/solution warm-start, then 128 GRPO updates over 512 unique prompts. GRPO produces 2,048 training completions under a 524,288-token cap, with checkpoints and 128-problem dev evaluations at 32/64/96/128. Dev selects one checkpoint using canonical pass, parseability, format, truncation, then earlier-step tie-breaking. Test is never used for selection.

The final hidden evaluation compares Base, old GRPO-v1 checkpoint-32, warmstart-only, and selected v2 on exactly 700 completions each. Candidate 0 covers 400 problems; a fixed 100-problem subset adds candidates 1–3, making a genuinely nested pass@4 pool. The primary success criterion is at least +3 percentage points candidate-0 pass@1 over old v1, with more paired improvements than regressions and a paired bootstrap interval. A warmstart-only gain is attributed to SFT; only v2 over warmstart supports incremental RLVR benefit.
