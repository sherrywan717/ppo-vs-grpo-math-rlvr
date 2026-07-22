# GRPO-v1 bottleneck analysis

This audit reconstructs only saved v1 CSV/JSON; it does not run a model. The dominant held-out failure is format error (604/800, 75.5%). Parse errors are 34/800, parseable-but-wrong answers 113/800, and canonical passes 49/800. All 64 truncated outputs are format failures, but truncation accounts for only 64/604 format failures; most format failures are non-truncated. Training format was 280/512 (54.69%) versus 196/800 (24.5%) on the observed final pool, consistent with a serious protocol-generalization/coverage gap rather than proof of a parser defect.

The v1 run saw only 128 unique RL problems. That narrow coverage plausibly limits generalization, but the audit cannot identify causality. Reward had usable within-group signal: 101/128 groups had nonzero variance; 27/128 were zero-advantage/all-equal and 15/128 all-zero. Therefore reward variance is not the primary infrastructure bottleneck. No correctness defect was found in the frozen parser, verifier, or reward, so v2 leaves their semantics unchanged.

The v2 intervention is explicitly **format/solution warm-start + GRPO-v2 RLVR**, not pure GRPO. Hidden-test attribution separates Base, old v1, warmstart-only, and v2.
