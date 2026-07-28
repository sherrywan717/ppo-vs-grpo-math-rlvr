# GRPO-v2 interview guide

## Thirty-second summary

I built an artifact-first RLVR pipeline around Qwen2.5-1.5B, first comparing PPO and GRPO under matched budgets and then preregistering a GRPO-v2 improvement protocol. GRPO-v2 combined completion-only format/solution warm-start with 128 group-relative RL updates over 512 unique prompts. Dev alone selected checkpoint-96. On a once-opened 400-problem hidden test, candidate-0 accuracy was 10.75%, versus 1.50% for Base, 4.25% for old GRPO-v1, and 2.50% for warm-start-only. The paired Base-to-v2 gain was +9.25 points, but this remains a single-seed result.

## Interview questions

### 1. Why RLVR?

Math provides executable-style feedback without executing generated code: a strict parser and canonical verifier can score final answers more objectively than a learned reward model. RLVR lets the policy optimize against that verifiable signal while retaining complete per-completion evidence.

### 2. What is the practical PPO versus GRPO difference?

PPO estimates advantages with a learned value model and optimizes a clipped policy objective. This adds a value backbone/adapter/head and value-loss stability concerns. GRPO samples multiple completions for the same prompt and normalizes rewards within the group, avoiding a learned value model but requiring within-group reward variance. Their loss and native entropy values are not directly comparable.

### 3. Why did old GRPO-v1 reach 4.25% while v2 reached 10.75% on the new hidden test?

V2 changed the training protocol before opening the new test: 256-example completion-only warm-start, 512 unique RL prompts rather than 128, 128 updates rather than 32, a deterministic easy-to-hard curriculum, and dev-only checkpoint selection. The experiment shows the combined protocol improved the frozen seed; it does not isolate the causal contribution of each change.

### 4. What did warm-start contribute?

It taught the exact `<reasoning>...</reasoning><answer>...</answer>` response protocol with assistant-only labels and trusted train gold. On matched dev, format improved from 17/128 to 23/128 and canonical pass from 6/128 to 8/128, but the paired interval crossed zero. It supplied a better-behaved initialization, not proof of math capability by itself.

### 5. Why is warm-start-only 2.50% still useful?

Its main role was optimization conditioning. It improved protocol adherence and made parseable rewards more reachable, giving subsequent GRPO groups more useful relative signal. Attribution remains explicit: the much larger v2 gain over warm-start is the evidence for incremental RLVR benefit.

### 6. Why checkpoint-96 rather than 128?

The rule was frozen before training: maximize canonical dev pass@1, then parseability, format, lower truncation, and earlier step. Step 96 achieved 33/128 canonical passes; step 128 fell to 28/128 despite better format and parseability. Hidden test never participated.

### 7. How was reward designed?

The fixed shaped reward separates strict answer envelope, valid/parseable answer, and canonical correctness, with correctness carrying the dominant weight. Canonical `RewardStatus` remains the scientific metric, so shaped-reward progress cannot be relabeled as mathematical success.

### 8. How were zero-advantage groups diagnosed?

Every four-completion group stores its full reward list and variance. Across 512 GRPO-v2 groups, 367 had nonzero variance and 145 were zero-advantage/all-equal; six were all-zero. These are learning-signal diagnostics, not post-hoc stopping rules.

### 9. Why use the unbiased pass@k estimator?

Checking only the first k samples makes the estimate depend on arbitrary candidate ordering. For n sampled candidates with c successes, the estimator `1 - C(n-c,k)/C(n,k)` is the probability that a uniformly selected k-subset contains a success.

### 10. How can one n=10 generation estimate pass@1, pass@4, and pass@10?

All ten completions are exchangeable draws from the same frozen decoding distribution. The same correct count c is inserted into the combinatorial estimator for k=1, 4, and 10. No resampling or different prompt pool is needed, and monotonicity is checked per problem.

### 11. How was leakage prevented?

Train, dev, and hidden test were selected before training with stable hash keys and separate execution/trusted-gold manifests. Content and source-ID overlap with every v1 manifest is zero. Warm-start is an explicitly declared train subset; the n=10 set is an explicitly declared hidden-test subset.

### 12. Why can hidden test not be used for more tuning?

It was the terminal scientific measurement. Any prompt, reward, checkpoint, or hyperparameter change after seeing it would convert it into development data and invalidate the registered comparison. A future study needs a new untouched test identity.

### 13. What does the single-seed limitation mean?

The paired result controls problem-level randomness for seed 42, but it does not measure training-seed variance. I can claim improvement under this frozen run and protocol, not broad stability or universal algorithm superiority.

### 14. How do memory, time, and cost compare?

GRPO-v2 training-only telemetry used 0.295499 GPU-hours, about ¥2.624, with 11,247 MiB peak nvidia-smi memory. The warm-start was short but peaked at 23,443 MiB. The four-model hidden evaluation used 1.980286 GPU-hours, ¥17.5849, and at most 5,321 MiB. Training-only telemetry is reported separately from full instance occupancy.

### 15. How are engineering and scientific failures separated?

A zero-update capacity or lifecycle failure is excluded from science. Conversely, if training, checkpoints, primary evidence, and checksums finalize before a launcher/IPC failure, the scientific result remains valid and the launcher status is disclosed separately. Original failure artifacts are immutable.

### 16. How are checkpoints and artifacts verified?

Each checkpoint is adapter-only and records config/model/data identities, counters, optimizer/scheduler/RNG state, curriculum cursor, prefix evidence, file sizes, and SHA256 inventory. Full base weights are rejected. Public Git contains only lightweight summaries and manifests; full archives remain remote-only.

### 17. Why did the launcher IPC failure not invalidate GRPO-v2 training?

The worker had already completed all 128 updates, four dev evaluations, four trusted checkpoints, primary evidence finalization, and a verified backup. It then blocked trying to return an oversized already-finalized result through a multiprocessing queue. The worker and parent were released without rerun; science and launcher state are therefore distinct.

### 18. What would you study next?

A new preregistered study would add multiple training seeds and perhaps a larger model or broader tasks, while keeping the existing hidden result sealed. It could ablate warm-start, curriculum, and expanded data coverage on a new dev/test split. It should not tune on the current hidden test.

## Deep-dive links

- [Final comparison](../reports/grpo_v2/hidden_test_final/final_comparison.md)
- [Training report](../reports/grpo_v2/grpo_v2_training/grpo_v2_seed42_20260726T044303Z/report.md)
- [Data freeze](../reports/grpo_v2/data_freeze_report.md)
- [Pass@k contract](../reports/grpo_v2/pass_k_contract.md)
- [Cost ledger](../reports/grpo_v2/portfolio/cost_ledger.json)
- [Engineering history](engineering_postmortem.md)
