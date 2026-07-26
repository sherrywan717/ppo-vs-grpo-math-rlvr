# Stage R GRPO-v2 seed-42 attempt

Run `grpo_v2_seed42_20260726T030733Z` is an immutable engineering failure before training and is excluded from scientific analysis. The one authorized command executed once; automatic retries were zero.

## Failure boundary

The frozen runtime stopped while rendering the frozen curriculum, before Trainer construction, generation, backward, optimizer steps, checkpoints, or dev evaluation. Config `max_prompt_length` is 832, while existing Stage O tokenizer evidence records `math:DigitalLearningGmbH/MATH-lighteval:train:4567` at 914 prompt tokens. That problem is frozen curriculum position 83 (update 21, slot 2). No data, prompt, completion cap, or configuration was changed.

The local model/tokenizer path and warm-start policy adapter load were reached before dataset rendering. A fresh GRPO optimizer/scheduler was not yet constructed; no SFT optimizer or scheduler state was loaded. Therefore this attempt provides no learning or optimizer evidence.

## Counters and metrics

- Updates/microsteps/optimizer/global steps: 0/0/0/0
- Unique prompts/completions/generated tokens: 0/0/0
- Checkpoints/dev evaluations/hidden-test accesses: 0/0/0
- Reward, loss, entropy, grad norm, KL, ratio, clip, format, parseable and canonical metrics: `null`, unavailable because no update completed.
- Worker resource-monitor wall time: 2.088978 seconds
- Peak nvidia-smi memory: 4 MiB; mean utilization: 0%
- GPU-hours/CNY: 0.000580272 / ¥0.005153

## Safety and recovery

Failure archive `grpo_v2_seed42_20260726T030733Z.failure.tar.gz` contains 17 lightweight entries, no checkpoint and no model weights. SHA256: `3fa2cbb730c5a72faa83cd35172873ce367537e19fc705d890c8d9bce4748fb8`. All 15 raw run checksums passed. After worker exit, GPU memory was 0 MiB with no compute process.

The unique blocker is a separately authorized CPU-only reconciliation of the frozen GRPO-v2 prompt/context capacity with the already frozen 512-problem curriculum. This attempt must not be resumed or retried. Hidden test remains sealed.
