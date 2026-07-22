# GRPO-v2 warm-start seed 42

Status: **scientific_training_success**. Run `warmstart_grpo_v2_seed42_20260722T051218Z` executed exactly once from commit `6895fa0a00c82ed0fcef12ba8514b1fc9c14b53e`. No GRPO-v2 or hidden-test evaluation ran.

## Contract and outcome

- Samples/epoch: 256/256, 1/1; every frozen sample appeared once in Stage N order.
- Batches/microsteps: 64/64; optimizer/global/scheduler steps: 16/16/16.
- Active supervised tokens: 46,058. Prompt/padding labels were ignored, assistant target and EOS active; truncation, duplicate, missing and prompt-label leakage counts were zero.
- Loss was finite at every step: first 2.2638, last 1.9100, mean 2.3387, range 1.7295–2.9399.
- Grad norm was available at all 16 steps: mean 4.5790, range 2.8796–5.8242.
- Reward, entropy, KL, canonical pass, advantage and value loss are unavailable because they are not native SFT metrics; none is represented as zero.

## Model and checkpoint

Qwen2.5-1.5B revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` ran BF16 on cuda:0 from the canonical offline snapshot. The tokenizer was `Qwen2TokenizerFast` with EOS 151645 and the frozen chat template; all 256 runtime encodes completed. Policy LoRA is r16/alpha32/dropout0 over q/k/v/o. The adapter has 4,358,144 trainables across 224 tensors; the verified optimizer has 224 parameter references and exactly 4,358,144 state elements. Base parameters are frozen.

`checkpoint-16` artifact SHA is `507749d393f38690915a76228b4c53a8b5c8927d40aada9f2768a90334d892f0` and adapter SHA is `44066dd13d8cfa4f5c40f10cad705eea617c37ce2e2f85ff5407751fb5a972b9`. It contains the adapter, optimizer, scheduler, Python/NumPy/PyTorch CPU/CUDA RNG, trainer/runtime/cursor identities and inventory. It contains no base-model weights. Future GRPO receives only the policy adapter and starts a fresh optimizer.

## Resources

The command through verified backup took 15.324s (0.004257 GPU-hours; CNY 0.0378 at 8.88/hour). Trainer runtime was 9.3468s. Peak nvidia-smi VRAM was 23,443 MiB (22.89 GiB), peak sampled utilization 74%. PyTorch allocator metrics are unavailable because the frozen runtime did not persist them. After process exit GPU memory was 0 MiB with no compute process.

## Dev evaluation disposition

Base dev-v2 and warm-start dev-v2 are both `not_executed_evaluator_unavailable`: at the authorized HEAD there is no frozen model-bound dev-v2 evaluator or CLI. No parameters were invented and no source/config was changed. This does not alter the successful training status; it is the sole blocker before a matched dev comparison and GRPO-v2 authorization review.
