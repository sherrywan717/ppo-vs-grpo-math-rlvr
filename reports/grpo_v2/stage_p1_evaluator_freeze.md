# Stage P.1 matched dev-v2 evaluator freeze

The repository now has one shared guarded evaluator for Base and the immutable
warm-start policy adapter. Both modes consume the same 128-row `dev_v2` order,
prompt bytes, per-problem seeds, parser/verifier/reward contract, sampling, and
832/256 prompt/completion limits. Candidate index is always zero.

## Frozen identity

- Config: `configs/grpo_v2/dev_evaluation_seed42.json`
- Config raw SHA256: `8501bfb945f85dda895d9278bb5d1d74a5d9c2c0791f9daa7cb0152d25e02528`
- Runtime registry canonical SHA256: `fc1cbf10698528a084406adf7a88f9f64cd02141f63d5c91cb9b025d07997db2`
- Dev manifest SHA256: `bdf02e1202e564177fea59f80f0b0ac8a36649daf8636ed6dd5bf3e5f6356b80`
- Trusted verifier manifest SHA256: `2b1739b7ec64f53f3772f5ea975b8d0466d632cb13dce00e133926e88d30b328`
- Warm-start checkpoint artifact SHA256: `507749d393f38690915a76228b4c53a8b5c8927d40aada9f2768a90334d892f0`
- Policy adapter SHA256: `44066dd13d8cfa4f5c40f10cad705eea617c37ce2e2f85ff5407751fb5a972b9`

## CPU validation

- 15 evaluator/safety tests passed.
- Base and warm-start dry-runs passed with zero model loads, generation, CUDA,
  training, backward, optimizer steps, and checkpoint writes.
- The pinned local tokenizer rendered all 128 prompts without truncation; lengths
  ranged from 112 to 453 tokens under the frozen 832-token cap.
- GRPO-v2 manifest validation and the directly affected warm-start/O.3 registry
  regressions passed.
- Ruff and compileall passed for affected files.

The evaluator runs in a spawned worker. Its non-CUDA parent checks process exit,
new compute processes, and restoration to the pre-run GPU-memory baseline. Success
requires 128 unique completion records, all primary artifacts, and a verified backup.
Pass@4 and pass@10 are explicitly unavailable because dev uses one candidate per
problem. No warm-start training or dev generation occurred during this freeze.
