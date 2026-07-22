# Stage O.3 evaluation-contract amendment

Status: **frozen, not run**. The O.2 50-problem nested pass@10 design is `superseded_before_any_evaluation`. It remains immutable in commit `079206e` and its original manifest is retained under `configs/grpo_v2/manifests/legacy/`. No Base, v1, warm-start, or v2 hidden-test generation had run, and no test-v2 model result was observed. This is a user-preregistered statistical-method amendment, not result-driven tuning.

The active `pass_k_shared_n10_subset` is byte-identically the Stage N 100-problem pass@4 manifest (SHA `86864437418a1a112b6385991bdba83617b0cc66f85ec6c7032f0ee79763a553`): GSM8K 50 and MATH500 50 at levels 3/8/10/14/15. Each model makes one n=10 call per subset problem under one frozen decoding identity. The per-problem batch seed is deterministically derived from evaluation seed 42, the problem ID, content hash, and namespace `grpo_v2/pass_k_shared_n10`, then shared across all four models. Candidate 0 is reused in the separate 400-problem candidate-0 accuracy ledger.

Per problem, with c canonical passes among n=10, the unbiased estimator is `1 - C(n-c,k)/C(n,k)` for k=1,4,10. Exact integer combinations and rationals precede float conversion. Problem estimates are averaged; standard errors and deterministic problem-level bootstrap intervals are reported. McNemar applies only to the binary 400-problem candidate-0 comparison.

The active ledger is 300 non-subset candidate-0 rows plus 1,000 shared-subset rows = 1,300 completions/model and 5,200/four models. Compared with O.2 this adds 1,200 rows; compared with independent k pools it saves 2,000 rows.

Only evaluation/pass@k and their propagated registry identities changed. Warm-start/GRPO configs, capacity, train/warmstart/dev/test manifests, shared 100-problem identity/order, curriculum, model, LoRA, prompt, reward, parser, and verifier are unchanged. The runtime-registry SHA change is `identity_only_transitive_change`; no training field changed.
