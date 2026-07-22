# Stage O.3 CPU validation

Status: **passed CPU-only**. CUDA initialization, Qwen model loading, generation,
Trainer, backward, optimizer, dev evaluation, GRPO-v2, and hidden-test calls are all
zero.

## Amendment

- The O.2 50-problem nested pass@10 contract is
  `superseded_before_any_evaluation`; its byte-identical manifest remains under the
  legacy directory and is rejected by the active resolver.
- The active `pass_k_shared_n10_subset` is the unchanged 100-problem Stage N
  manifest: GSM8K 50 and MATH500 levels 3/8/10/14/15.
- One future generate call produces candidates 0--9 under one frozen sampling and
  per-problem seed identity. The seed is deterministically bound to evaluation seed 42, problem ID, content hash, and the frozen namespace. Missing, duplicate, mismatched, or incomplete evidence
  fails closed.
- Exact problem estimates use `1-C(10-c,k)/C(10,k)` for k=1/4/10, followed by
  arithmetic problem averaging, standard errors, and problem-level bootstrap CIs.
- `candidate0_accuracy_all_400` and `unbiased_pass_at_1_subset_100` are distinct.
  The ledger is 1,300 completions/model and 5,200/four models.

## Identity

Only evaluation/pass@k and propagated registry identities changed. The warm-start
config remains `c8e3e0a5...`, GRPO-v2 config `05955388...`, curriculum `7f7dcfa1...`,
and the train/warmstart/dev/test/shared-subset manifests remain byte-identical. Model,
LoRA, prompt, reward, parser, verifier, and 928/640/1,088 capacity are unchanged. The
runtime-registry update is an identity-only transitive data-registry binding; no
training field or warm-start runtime source changed.

## Verification

- Targeted pytest: 19 passed.
- Exact examples include c=0/1/2/10 and k=1/4/10; c=1 pass@4=2/5 and c=2
  pass@4=2/3.
- Invalid n/c/k, 9/11 candidates, duplicate/missing indices, problem drift,
  verifier disagreement, sampling mismatch, duplicate candidate 0, and missing
  identities are rejected.
- Per-problem and aggregate monotonicity passed; JSON/CSV estimator rebuild matched.
- Affected Ruff and compileall passed.
- Evaluation/config dry-run passed with 100 shared problems, 1,300 completions/model,
  and 5,200/four models.
- Manifest validation and `check_env` passed; CUDA/model/generation remained false.
- Secret/large-file audit found no credential pattern, forbidden model/checkpoint
  artifact, or file over 50 MiB.
- `git diff --check` passed.
