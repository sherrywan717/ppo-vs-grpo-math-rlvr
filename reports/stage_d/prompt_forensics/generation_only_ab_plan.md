# Generation-only v0/v1 A/B diagnostic — plan only

Requires independent `--generate-only --confirm-prompt-diagnostic`; `--confirm-single-update` is invalid. Two fixed problems, two prompt variants, four completions per problem/variant: 16 completions and at most 2,048 tokens. No train, backward, optimizer, or checkpoint. Expected/worst wall time: 40/120 seconds; expected peak/gate VRAM: 2.5/3.5 GiB; expected/worst cost at CNY 8.88/hour: CNY 0.099/CNY 0.296.
