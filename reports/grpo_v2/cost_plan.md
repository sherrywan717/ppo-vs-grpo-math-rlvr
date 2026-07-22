# GRPO-v2 cost plan

These are preregistered planning estimates, not measurements. The v1 GRPO seed-42 training plus validation used about 0.331 GPU-hours, peak 10.95 GiB, and CNY 2.93. Warm-start is expected at 0.25–0.50 GPU-hours (15–30 min), 12–18 GiB, CNY 2.22–4.44; its ceiling is 0.75 GPU-hours, 24 GiB, CNY 6.66. GRPO-v2 plus four dev evaluations is expected at 0.9–1.4 GPU-hours (54–84 min), 11–16 GiB, CNY 7.99–12.43; its ceiling is 2 GPU-hours, 24 GiB, CNY 17.76. Hidden testing is separately authorized and budgeted later.


## Stage O.2 update

The full tokenizer audit itself took roughly 9 seconds of CPU wall time. Warm-start planning remains 15–30 minutes, 12–18 GiB and CNY2.22–4.44, with ceilings 45 minutes, 24 GiB, 0.75 GPU-hours and CNY6.66. Expected checkpoint footprint is roughly 50–70 MiB (adapter plus Adam/scheduler/RNG state), ceiling 100 MiB. A 128-completion warmstart dev evaluation is projected at about 7.5 minutes / 0.125 GPU-hours / CNY1.11, ceiling 10 minutes / CNY1.48. GRPO-v2 remains at the Stage N expected 0.9–1.4 GPU-hours and CNY7.99–12.43, ceiling 2 GPU-hours/CNY17.76. Pass@10 adds 1,200 completions across four models; historical 800-completion throughput projects 1.163 GPU-hours and CNY10.33, with planning ceiling CNY12.20. It adds no training cost.
