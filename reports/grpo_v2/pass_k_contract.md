# GRPO-v2 nested pass@k contract

Pass@1 (400 problems) and nested pass@4 (100 problems) remain primary. Secondary nested pass@10 uses a deterministic 50-problem strict subset: GSM8K 25 and MATH500 25 at L1–5 = 2/4/5/7/7. Candidate 0 is shared by all metrics; candidates 1–3 are reused from pass@4; only candidates 4–9 are new. Each model therefore has 400 + 300 + 300 = 1,000 completions; four models have 4,000.

On the same 50 problems, reports must show integer success@1, success@4, and success@10 and enforce success@1 ≤ success@4 ≤ success@10. Pass@10 is secondary/exploratory, never selects a checkpoint, changes training, triggers retraining, or replaces headline pass@1/pass@4. Level 1 is diagnostic-only small-n. Historical 800-completion throughput implies 1,200 extra completions cost about CNY 10.33, with observed-rate range CNY 7.79–12.14 and planning ceiling CNY 12.20.
