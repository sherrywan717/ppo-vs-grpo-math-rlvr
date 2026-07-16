# Formal experiment interview questions

## Design and fairness

1. Why compare PPO and GRPO at matched completions and generated-token caps instead of
   merely matching Trainer steps?
2. How does the 2-GSM8K/2-MATH update schedule remove a domain-order confound?
3. Why does PPO use 512 repeated episode rows while GRPO uses 128 rows with four
   generations, and how are both mapped to the same 512 comparison keys?
4. Which properties are intentionally identical—policy LoRA, prompt, sampling, data,
   reward, seeds, updates—and which PPO value-model differences are unavoidable?
5. Why is the fixed step-32 checkpoint used for final test rather than the checkpoint
   with the best test or validation score?

## Reward and verification

6. Why was Countdown exact-number-usage removed from the formal reward?
7. What distinguishes shaped training reward from canonical pass@1/pass@4?
8. How do GSM8K numeric exactness and MATH symbolic equivalence differ?
9. Why must `INFRA_ERROR` abort while a wrong answer receives a legitimate low reward?
10. What evidence shows PPO and GRPO use the same reward policy and domain router?

## TRL and optimization

11. Derive PPO's 32 optimizer steps from 512 episodes, rollout batch 16, one PPO epoch,
    one minibatch, microbatch 4, and gradient accumulation 4.
12. Derive GRPO's 512 completions and 128 microsteps from generation batch 16,
    `num_generations=4`, per-device batch 4, GA4, and `max_steps=32`.
13. Why is `num_generations` forbidden in PPOConfig?
14. What did the 0.5B pilot teach about Accelerate's sync boundary and reliable
    microbatch evidence?
15. How are policy/value optimizer membership and checkpoint role separation audited?

## Statistics and claims

16. Why report raw seed values, sample SD, and paired problem bootstrap intervals?
17. What can and cannot be inferred from two seeds?
18. Why is pass@4 evaluated on a frozen subset, and how is it paired pre/post?
19. How would zero reward variance or zero canonical accuracy be reported without
    silently changing the experiment?
20. What result would justify saying one algorithm performed better, and what result
    would only justify saying its execution path was more stable?

## Reproducibility and operations

21. How do exact model revision, manifest hashes, semantic prompt/reward/verifier hashes,
    and resolved config hashes prevent identity drift?
22. Why are model weights excluded from Git and static backups while adapters are kept
    in verified persistent archives?
23. How can every plot be reproduced without access to the live Trainer object?
24. What is the disclosed validation provenance issue, and why was historical evidence
    preserved rather than rewritten?
25. Which future stages need separate authorization, and why does a successful sanity
    or baseline not implicitly authorize training?
