# Next task: CPU-only reconcile the frozen GRPO-v2 prompt capacity

Stage R run `grpo_v2_seed42_20260726T030733Z` executed once and stopped before Trainer construction, generation or training. Frozen curriculum problem `math:DigitalLearningGmbH/MATH-lighteval:train:4567` is 914 prompt tokens, exceeding the frozen GRPO-v2 `max_prompt_length=832`. Counters are zero; no checkpoint or dev result exists. The run and failure backup are immutable and excluded from scientific analysis.

The sole next task requires separate CPU-only authorization: reconcile prompt/context capacity with the unchanged 512-problem curriculum and unchanged 256-token completion budget, repeat the complete tokenizer-length audit, update only the necessary capacity identities, and rerun targeted dry/fake gates. Do not remove or truncate the problem, change prompt/data/reward/parser/verifier, access hidden test, or retry Stage R without a new explicit GPU authorization.
