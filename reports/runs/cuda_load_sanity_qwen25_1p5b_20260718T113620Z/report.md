# Qwen 2.5 1.5B CUDA/model-load sanity

- Status: **SUCCESS**
- Run ID: `cuda_load_sanity_qwen25_1p5b_20260718T113620Z`
- Repository: `Qwen/Qwen2.5-1.5B-Instruct`
- Revision: `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`
- Snapshot: `/root/autodl-tmp/cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306`
- Local-only/offline: true
- GPU: NVIDIA H800 PCIe, `cuda:0`
- Generation, Trainer, LoRA, reward/verifier rollout, backward, optimizer,
  checkpoint, baseline, and training counters: all 0

## Load and forward results

- Architecture/model type: `Qwen2ForCausalLM` / `qwen2`
- Parameter count: `1543714304`
- Dtype/device: `torch.bfloat16` / `cuda:0`
- Meta parameter count: `0`
- Tokenizer: `Qwen2TokenizerFast`; chat template available
- Frozen formal examples: `gsm8k:train:1534` and `math:DigitalLearningGmbH/MATH-lighteval:train:4929`
- Prompt token lengths: `137`, `120`
- Logits shapes: `[1,137,151936]`,
  `[1,120,151936]`
- Logits finite checks: `[true,true]`
- Tokenizer load: `0.2915848046541214` seconds
- Model load: `1.5479077696800232` seconds
- Forward total: `0.48325344175100327` seconds

## Resource and cost evidence

- PyTorch peak allocated: `3117.28 MiB`
- PyTorch peak reserved: `3308.00 MiB`
- `nvidia-smi` peak used memory: `3915 MiB`
- `nvidia-smi` peak utilization: `9%`
- Resource samples: `24`
- Worker wall time: `6.630834210664034` seconds
- GPU-hours: `0.0018418983918511206`
- Cost at CNY 8.88/GPU-hour: `0.016356057719637954`

## Cleanup and warning

The worker deleted model/tokenizer/input/output references, ran `gc.collect()`, called
`torch.cuda.empty_cache()`, synchronized, and exited. Its final in-process allocator
reading remained 32 MiB allocated and 32 MiB reserved; this is a non-blocking warning,
not a release failure. The independent post-exit `nvidia-smi` check reported 0 MiB,
0% utilization, and no compute process, which is the authoritative release result.

The PNG figure is rebuilt only from `resource_metrics.csv` (with summary identity
from `summary.json`). This sanity does not authorize baseline or formal training.
