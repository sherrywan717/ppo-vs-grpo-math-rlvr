# Qwen 0.5B GRPO single-update smoke — FAILURE

- Diagnostic: single-update smoke diagnostic; not a formal experiment result.
- Run ID: grpo_single_update_qwen25_05b_20260713T053852Z
- Snapshot builder: passed; exact canonical local snapshot path used.
- BudgetGuard JSON finalization: passed; no clock callable serialized.
- Failure phase: checkpoint inventory.
- Failure: current allowlist rejected the 7,441-byte training-state file \`training_args.bin\`.
- Counters: prompts 2, completions 8, generated tokens 687, microsteps 4, optimizer/global step 1/1.
- Metrics: loss 0.0, reward mean/std 0.0/0.0, KL unavailable, grad norm 0.0.
- Warning: zero reward variance and zero-advantage fraction 1.0; this does not demonstrate learning.
- Verifier: format_error 8; pass@1/pass@4 and format/valid-expression/number-usage rates are all 0.
- Peak sampled VRAM: 2597 MiB; wall time 12.907s; GPU-hours 0.00358519; cost ¥0.031836.
- Adapter: 2,162,688 trainable parameters (0.435854% of parameters including adapter).
- Checkpoint: no full base model detected; inventory failed closed on \`training_args.bin\`.
- Artifact gap: generated completion text, exact per-completion token lengths, KL, and PyTorch allocator peaks were not persisted.
