# GRPO-v2 shared unbiased pass@k contract

Status: **frozen, not run**. The O.2 50-problem pass@10 design is `superseded_before_any_evaluation`; its immutable manifest is retained at `configs/grpo_v2/manifests/legacy/pass10_nested_subset_o2.json`. No hidden-test generation or model result existed when the user preregistered this statistical-method amendment.

The unchanged 100-problem Stage N pass@4 subset is now `pass_k_shared_n10_subset`: GSM8K 50 and MATH500 50 (levels 3/8/10/14/15). Each model makes one `generate` call per shared problem with `num_return_sequences=10`, candidates 0–9, a fixed per-problem batch seed, and one frozen decoding distribution. The seed is the low 63 bits of the first 64 bits of SHA256 over namespace `grpo_v2/pass_k_shared_n10`, evaluation seed 42, problem ID, and content hash; it is identical across all four models. Candidate 0 is reused for the 400-problem accuracy ledger; it is never regenerated.

For each problem with `n=10` and `c` canonical `VERIFIED_PASS` candidates, the exact estimator is `1 - C(n-c,k)/C(n,k)` for k=1,4,10. Integer combinations and an exact rational are saved before float conversion. Estimates are computed per problem and then arithmetically averaged. Standard errors and deterministic 10,000-resample problem-level bootstrap intervals are reported. Both per-problem and aggregate pass@1 <= pass@4 <= pass@10 are mandatory. MATH Level 1 has n=3 and remains `diagnostic_only_small_n`.

`candidate0_accuracy_all_400` is a separate binary metric from `unbiased_pass_at_1_subset_100`. The former supports improvement/regression, paired bootstrap, and McNemar; continuous problem-level unbiased pass@k differences use paired bootstrap, not McNemar.

Each model produces 300 non-subset candidate-0 rows plus 1,000 shared-subset rows, totaling 1,300; four models total 5,200. An independent 7,200-completion design would use 2,000 more. One n=10 call reduces repeated prompt tokenization/prefill and scheduling but does not eliminate decode work for any candidate.

Expected four-model usage from historical evaluation throughput is 5.047587 GPU-hours / CNY 44.82; the conservative ceiling is 5.923129 GPU-hours / CNY 52.60.
