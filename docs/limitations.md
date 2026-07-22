# Limitations

1. **Scale:** the experiment uses a single 1.5B model family and does not establish scaling behavior.
2. **Budget:** each run has only 32 updates and 512 training completions.
3. **Validation size:** 64 problems make integer correct counts more informative than smooth percentages.
4. **Final-test coverage:** only seed 42 has complete Base/PPO/GRPO final evaluations. GRPO123 and PPO123 are `deferred_not_executed`.
5. **Candidate pools:** sampled pass@1 and independent pass@4 use different problem/candidate pools. Monotonicity and cross-pool per-problem comparisons are invalid.
6. **Algorithm metrics:** PPO and GRPO losses are definitionally different. Native entropy may come from different TRL fields/masks and is not compared numerically across algorithms.
7. **Inference scope:** two training seeds and one complete final-evaluation seed do not support a general algorithm-superiority or cross-task significance claim.
8. **Optional telemetry:** some KL/ratio/grad/entropy fields are unavailable; they remain null with reasons.
9. **Resource scope:** PPO42 validation recovery used separate processes, so timing includes extra load overhead.
10. **Test reuse:** published test results are terminal evidence and must not tune GRPO-v2.
11. **License:** no license has been selected; absence of a LICENSE file is intentional.
