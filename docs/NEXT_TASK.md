# Next task: Stage N GRPO-v2 CPU-only design freeze

Portfolio v1 freezes the current scientific result. All four formal training/checkpoint-validation runs are complete; held-out final evaluation is complete for Base/PPO/GRPO at seed 42. GRPO seed-123 and PPO seed-123 final evaluations are deliberately `deferred_not_executed` and must not be inferred or represented as zero.

The only next task after Stage M publication is a separately authorized **CPU-only** GRPO-v2 design freeze:

- create a new versioned experiment identity; never edit portfolio-v1 configs or artifacts;
- use training/validation evidence only for design and selection; the published held-out test is forbidden for tuning;
- pre-register candidate changes, budgets, seeds, selection/stopping rules, and artifact contracts;
- preserve safe parser/verifier behavior, per-candidate evidence, token accounting, null/unavailable semantics, and model/cache/checkpoint publication boundaries;
- produce no CUDA initialization, model/tokenizer loading, generation, training, validation, or final evaluation without a later explicit authorization.

Planning entry point: `docs/grpo_v2_roadmap.md`.

Stage M itself authorizes only CPU-side portfolio audit, documentation, packaging, Git commit/tag, and GitHub publication. It does not authorize any GPU task.
