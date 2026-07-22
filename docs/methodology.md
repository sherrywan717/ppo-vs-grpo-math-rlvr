# Methodology

The study asks whether GRPO offers a better sample-efficiency/resource trade-off than PPO for few-shot mathematical RLVR at 1.5B parameters. It uses an artifact-first design: every claim must trace to persisted per-update, per-candidate, identity, resource, and checksum evidence.

Both algorithms use the same Qwen2.5-1.5B revision, frozen data order, prompt, policy LoRA, sampling, reward, parser/verifier, completion cap, 512-completion budget, generated-token cap, and checkpoint cadence. PPO's separate value model/value loss and GRPO's four-candidate relative advantages are algorithm-intrinsic differences, not hidden mismatches.

Training uses 64 GSM8K plus 64 MATH problems in a deterministic 32-update order. Validation uses a separate frozen 64-problem set at steps 8/16/24/32. Test uses 200 GSM8K and 200 MATH500 problems for sampled pass@1 and an independent fixed 50+50 subset for four-candidate pass@4. MATH500 never enters training.

Canonical correctness comes from the strict parser/verifier rather than the shaped reward scalar. Training diagnostics include reward, canonical status, reward-group variance, zero-advantage groups, losses, KL/ratio/clip where available, entropy with its native definition, lengths, EOS/truncation, tokens, time, VRAM, GPU-hours, and cost. Optional unavailable metrics remain null with a reason.

The headline paired analysis compares seed-42 Base, PPO checkpoint-32, and GRPO checkpoint-32 on identical candidate keys within each candidate pool. It reports paired transitions, 10,000-resample bootstrap intervals, and exact McNemar tests. The two pools are never crossed or treated as nested.
