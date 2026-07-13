# Smoke prompt fairness contract

The Qwen 0.5B PPO and GRPO smoke paths share one selected prompt identity:
`prompt_v1_strict_concise`. This is a bounded integration choice, not activation for
main/formal experiments and not evidence that v1 is a final production prompt.

Fair comparison requires all of the following:

- the same `prompt_version`, `prompt_sha256`, and `renderer_version` in both smoke
  resolved configs;
- byte-identical rendering for the same `MathProblem` through the shared
  `math_rlvr.prompt` renderer;
- persistence of those three fields in resolved config, run manifest, and report;
- no algorithm-specific prompt edits, gold constructions, answers, or dataset-derived
  leakage;
- unchanged parser, verifier, RewardPolicy, data records, seeds, and sampling unless a
  later change is separately reviewed for both algorithms;
- comparison by actual completions and generated tokens, not nominal trainer steps.

The activation changes only the 0.5B smoke prompt selector. Main/formal 1.5B configs
remain on their prior state. `prompt_v0_grpo_smoke` remains immutable for replay of
historical runs.

The source A/B diagnostic found v1 improved complete-envelope rate from 0% to 25% and
created two nonzero within-problem reward-variance groups. Valid expression, number
usage, pass@1, and pass@4 remained zero. Accordingly v1 is
`approved_for_smoke` but `not_approved` for production.
