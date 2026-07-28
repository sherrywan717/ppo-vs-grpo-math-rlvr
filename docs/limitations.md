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
## GRPO-v2 release limitations

- The final improvement experiment has one training seed and one 1.5B model.
- Candidate-0 accuracy uses 400 problems; unbiased pass@k uses a different shared
  100-problem universe and cannot be substituted for the headline metric.
- MATH Level 1 has only three hidden problems and is diagnostic only.
- Warm-start, expanded data coverage, curriculum and longer training changed together;
  this run does not causally isolate their individual effects.
- Optional ratio/KL telemetry is unavailable under the frozen runtime where noted.
- The hidden test was opened once and cannot support more tuning or retraining.
- Portfolio-v1 and GRPO-v2 use different held-out identities; only within-protocol
  paired comparisons are valid deltas.
