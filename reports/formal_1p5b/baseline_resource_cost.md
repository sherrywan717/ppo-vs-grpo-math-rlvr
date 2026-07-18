# Frozen baseline resource and cost report

| Run | Seed | Wall seconds | Tokens | Tokens/s | PyTorch peak allocated MiB | PyTorch peak reserved MiB | nvidia-smi peak MiB | Mean GPU util % | GPU-hours | Cost CNY |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline_formal_1p5b_seed42_20260718T125833Z | 42 | 2105.2 | 96150 | 45.672 | 3090.0 | 3304.0 | 3913 | 38.05 | 0.584780 | 5.1928 |
| baseline_formal_1p5b_seed123_20260718T133624Z | 123 | 2049.6 | 91651 | 44.717 | 3090.0 | 3306.0 | 3915 | 36.78 | 0.569323 | 5.0556 |

Combined: 4154.8 wall seconds, 187801 generated tokens, 1.154103 GPU-hours, ¥10.2484. Cost uses ¥8.88 per GPU-hour.

Worker-exit allocator residue is a warning only. Independent post-process checks found 0 MiB and no compute process after both runs.
