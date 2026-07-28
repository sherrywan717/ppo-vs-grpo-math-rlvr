# Next task: publish the frozen GRPO-v2 result

Stage S.4 completed the remaining three hidden-test roles exactly once. Together with
the non-regenerated Base recovery, the frozen comparison now contains 5,200
completions: 1,300 for each of Base, old GRPO-v1, warm-start-only, and selected
GRPO-v2 checkpoint-96.

The primary 400-problem candidate-0 results are:

- Base: 6/400 (1.50%)
- old GRPO-v1: 17/400 (4.25%)
- warm-start-only: 10/400 (2.50%)
- selected GRPO-v2: 43/400 (10.75%)

On the shared 100-problem n=10 pool, selected GRPO-v2 reached unbiased
pass@1/pass@4/pass@10 of 14.40%/31.14%/42.00%. Full machine-readable results and
paired comparisons are in
[`reports/grpo_v2/hidden_test_final/`](../reports/grpo_v2/hidden_test_final/).

No additional training, dev evaluation, hidden-test generation, checkpoint selection,
or test-driven tuning is authorized. The next appropriate stage is CPU-only portfolio
publication/release review on `improve/grpo-v2`, with the single-seed and small
MATH-Level-1 limitations preserved.
