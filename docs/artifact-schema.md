# GRPO Smoke Artifact Schema

This schema is for the guarded Qwen 0.5B single-update integration smoke. It is diagnostic evidence, not a formal PPO-versus-GRPO result.

## Completion evidence

`completions.jsonl` must contain exactly eight ordered records. Every record stores:

- `problem_id`, `prompt_hash`, `generation_index`, and global `completion_index`;
- padded `completion_ids` and binary `completion_mask`;
- `exact_token_count`, computed only as the mask sum;
- `decoded_completion`, the decoded TRL tensor output;
- `raw_completion`, the exact text supplied to the reward callback;
- `verifier_input`, the exact text supplied to the math verifier;
- `reward_status`, `canonical_status`, `scalar_reward`, and verifier detail;
- `reward_policy_version`, `reward_policy_sha256`, and
  `reward_component_weights`;
- the five explicit component fields: `answer_block_component`,
  `strict_protocol_component`, `valid_expression_component`,
  `exact_number_usage_component`, and `correctness_component`.

The three text fields must be byte-for-byte equal for this non-conversational smoke. IDs/masks, decoded text, reward input, and reward result are joined by their original index. Missing text, count mismatch, order mismatch, or more/fewer than eight records fails the run before success.

## Trainer and KL metrics

The raw `trainer.state.log_history` is retained. Normalized metrics preserve the original key. Reviewed TRL 0.24.0 KL aliases are `kl`, `train/kl`, and `objective/kl`.

The frozen GRPO config resolves to `beta=0.0`, so TRL does not compute reference-model KL. The required representation is:

- `kl_available=false`;
- `kl=null`;
- `kl_raw_key=null`;
- a non-empty `kl_unavailable_reason`.

Missing KL is never rewritten as zero.

## GPU memory

`pytorch_allocator.json` is separate from the nvidia-smi resource timeline. It records current and peak allocated/reserved memory in bytes and MiB plus the CUDA device. The guarded real path resets peak statistics after authorization and before model load. If CUDA collection is unavailable, every value is null with an explicit reason; CPU dry-runs do not initialize CUDA.

## Checkpoint and finalization

The summary embeds completion evidence count, trainer metrics/log history, allocator evidence, checkpoint inventory, and duplicate checkpoint count. All JSON is primitive-only, finite-number checked, reloadable, and covered by checksums. Success requires complete completion evidence, one canonical checkpoint, verified artifacts, and verified backup.


## Reward policy identity

For post-intervention 0.5B smoke runs, resolved config, run manifest, summary, and each
completion record must agree on the versioned reward contract. `RewardStatus` remains
the unchanged strict canonical result even when the staged training scalar is nonzero.
The historical v1 GRPO run predates `shaped_v2_staged` and is never relabeled.

The staged component weights are 0.05 answer block, 0.05 strict protocol, 0.05 safe
expression, 0.05 exact number use, and 0.80 correctness. Resource limits receive no
partial components and infrastructure errors abort. Sparse reward remains 1.0 only for
canonical `VERIFIED_PASS`.
