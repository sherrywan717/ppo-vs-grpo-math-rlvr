# CUDA/model-load sanity check

- Status: **SUCCESS**
- Run ID: `cuda_load_sanity_qwen25_05b_20260713T042511Z`
- Revision: `7ae557604adf67be50417f59c2c2f167def9a775`
- Local-only: true
- Generation completions: 0
- Training updates: 0
- Checkpoints written: 0

## Results

- reason: `all local-only BF16 load and forward contracts passed`
- parameter_count: `494032768`
- dtype: `['torch.bfloat16']`
- device: `['cuda:0']`
- meta_parameter_count: `0`
- lora_trainable_parameters: `0`
- lora_trainable_ratio: `0.0`
- matched_lora_target_modules: `{'q_proj': 24, 'k_proj': 24, 'v_proj': 24, 'o_proj': 24}`
- wall_time_seconds: `1.8961350359022617`
- gpu_hours: `0.0005267041766395172`
- actual_cost_cny: `0.004677133088558913`
- torch_peak_allocated_mib: `1001.73046875`
- torch_peak_reserved_mib: `1046.0`
- nvidia_smi_peak_used_mib: `459.0`
- warning_count: `1`

## Warnings

- FutureWarning: `torch_dtype` is deprecated; use `dtype` instead.
