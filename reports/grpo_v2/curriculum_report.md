# Frozen curriculum

The 512 train prompts appear exactly once in 128 four-prompt updates. Updates 1–32 pair two shortest-solution GSM8K prompts with two MATH Level-1 prompts. Updates 33–96 mix GSM8K and Level 2 (32 updates at 3:1, then 32 at 2:2). Updates 97–128 pair one longest remaining GSM8K prompt with three Level-3 prompts. Split selection is hash-based; only the already-selected GSM8K official-train solution whitespace length is used as a deterministic curriculum proxy. No model output, validation result, or test field participates.
