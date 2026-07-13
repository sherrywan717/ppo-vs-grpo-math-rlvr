# GPU smoke plan (not executed)

Price: ¥8.88/GPU-hour. Estimates are planning bounds, not measured results.

## A. CUDA/model-load sanity check

- Data/prompts/completions: 2 samples, 2 prompts, 0 per prompt, 0 total
- Token caps: prompt 128, completion 0, total generation 0
- Batch/accumulation: 1 / 1
- LoRA: none; base BF16 read-only load
- Checkpoint: none
- Peak VRAM estimate: 1.5 GiB
- Time: 2 min estimated; 5 min worst
- Cost: ¥0.30 estimated; ¥0.74 worst
- Success: fixed revision loads once; two tokenized prompts forward-pass finite; CUDA device/peak memory JSON recorded
- Automatic stop: 300 s wall cap, VRAM >2.5 GiB, exception, NaN/Inf
- OOM/NaN/timeout: stop immediately; preserve diagnostics only; no retry in same paid run

## B. GRPO 0.5B single-update smoke

- Data/prompts/completions: 2 samples, 2 prompts, 4 per prompt, 8 total
- Token caps: prompt 512, completion 128, total generation 1024
- Micro-batch/gradient accumulation: 2 / 4
- LoRA: policy BF16 LoRA r=16 alpha=32 dropout=0; q/k/v/o
- Checkpoint: save final adapter only after successful update; save_total_limit=1
- Generation batch / inferred steps per generation / iterations: 8 / 4 / 1
- Batching revision: the old 2/1 setting could not make one optimizer update consume all 8 completions. The revised 2/4 configuration lets TRL infer four micro-batches from `generation_batch_size=8`; this is a smoke integration repair, not an experiment result.
- Peak VRAM estimate: 6.5 GiB
- Time: 10 min estimated; 15 min worst
- Cost: ¥1.48 estimated; ¥2.22 worst
- Success: exactly one optimizer update; 8 bounded completions; finite loss/reward; artifacts/checksum complete
- Automatic stop: 900 s, 1024 generated tokens, 8 completions, VRAM >10 GiB, OOM, NaN/Inf
- Guarded command: `PYTHONPATH=src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m math_rlvr.training.grpo --config configs/smoke/grpo.yaml --execute --confirm-single-update`
- Authorization: both flags, frozen config, clean Git, fixed local snapshot, and all budget gates; never retries or enters PPO.
- Accounting: isolated `trl==0.24.0` shim validates exact completion IDs/masks; guards 8 completions, 1,024 tokens, 4 microsteps, one optimizer/global step, and 900 seconds.
- State: success only after complete artifacts, adapter-only checkpoint inventory, tar backup, and SHA256 verification.
- OOM/NaN/timeout: OOM: stop and report measured peak, then propose shorter completion/smaller batch; NaN: retain metrics and stop; timeout: terminate gracefully, no automatic rerun

## C. PPO 0.5B single-update smoke

- Data/prompts/completions: 4 samples, 4 prompts, 1 per prompt, 4 total
- Token caps: prompt 512, completion 128, total generation 512
- Batch/accumulation: 4 / 1
- LoRA: policy r=16 alpha=32 q/k/v/o; value r=8 alpha=16 q/v plus score head; BF16
- Checkpoint: save final policy/value adapters only after successful update; save_total_limit=1
- Peak VRAM estimate: 10.0 GiB
- Time: 10 min estimated; 20 min worst
- Cost: ¥1.48 estimated; ¥2.96 worst
- Success: exactly one PPO optimizer update; 4 bounded responses; finite policy/value/KL metrics; artifacts/checksum complete
- Automatic stop: 1200 s, 512 generated tokens, 4 completions, VRAM >14 GiB, OOM, NaN/Inf
- OOM/NaN/timeout: OOM: stop and report component/peak, then propose batch 2 with grad-accum 2; NaN: retain metrics and stop; timeout: terminate gracefully, no automatic rerun

## Required artifacts

Each run saves config.json, summary.json, metrics.csv, gpu_metrics.csv, stdout.log, stderr.log, checksums.sha256, reward/loss/KL/VRAM charts as applicable, and only the stated adapter checkpoint.
