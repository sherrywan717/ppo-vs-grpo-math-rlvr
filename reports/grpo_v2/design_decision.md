# Single-seed GRPO-v2 design decision

GRPO-v2 freezes seed 42, a 256-example one-epoch format/solution warm-start, then 128 GRPO updates over 512 unique prompts. GRPO produces 2,048 training completions under a 524,288-token cap, with checkpoints and 128-problem dev evaluations at 32/64/96/128. Dev selects one checkpoint using canonical pass, parseability, format, truncation, then earlier-step tie-breaking. Test is never used for selection.

The final hidden evaluation compares Base, old GRPO-v1 checkpoint-32, warmstart-only, and selected v2 on exactly 1,300 completions each. Candidate 0 covers all 400 problems; the unchanged 100-problem subset uses one exchangeable n=10 batch and exact unbiased pass@1/pass@4/pass@10 estimators. Candidate-0 accuracy over 400 and unbiased pass@1 over 100 are distinct metrics. The primary success criterion remains at least +3 percentage points candidate-0 accuracy over old v1, with more paired improvements than regressions and a paired bootstrap interval. A warmstart-only gain is attributed to SFT; only v2 over warmstart supports incremental RLVR benefit.

## Stage Q executable experiment plan

The frozen design now has one guarded model-bound implementation. A new run starts from the warm-start checkpoint-16 policy adapter only, initializes a fresh GRPO optimizer/scheduler, and follows the 512-position curriculum without shuffle. Updates/checkpoints/dev are 128 and 32/64/96/128; training and dev ledgers are disjoint. Only same-run checkpoints 32/64/96 can resume. The final checkpoint selection rule and sealed hidden-test contract are unchanged. Stage Q adds execution wiring, not a scientific amendment.

## Stage R execution boundary

The first frozen GRPO-v2 command failed before optimization because one immutable train prompt measured 914 tokens against the registered 832-token cap. This is an engineering capacity mismatch, not a scientific result. No data, order, prompt, reward, parser/verifier, sampling, completion budget, or hidden-test identity was changed. Any capacity amendment must be explicit, CPU-audited and versioned before a new GPU authorization.

## Post-freeze Stage R.1 capacity amendment

The first Stage R attempt exposed a propagation defect before training: GRPO-v2 retained prompt cap 832 although the warm-start audit had already established prompts above that limit. Stage R.1 therefore replayed the exact tokenizer/renderer over the complete 512-train/128-dev universe before changing capacity. Maximum prompt length is 918; the predeclared `max(928, ceil(max/32)*32)` rule selects 928. Completion remains 256 and explicit sequence ceiling becomes 1,184. The model context is 32,768 and no row truncates. This amendment changes capacity identities only and does not change any scientific sample, order, text, optimization setting, or hidden-test contract. See [the full amendment](prompt_capacity_amendment.md).
