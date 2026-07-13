# Guarded PPO single-update runner CPU gate

Status: **PASS**. This was implementation and fake-contract verification only. CUDA
was not initialized, no real model/tokenizer was loaded, no completion was generated,
and no real PPO/GRPO train or optimizer call occurred.

TRL 0.24.0 resolves the smoke YAML to four fixed prompts, one response per prompt,
four completions, at most 512 generated tokens, one PPO epoch, one minibatch, and one
optimizer/update/global step. The configured `num_generations: 4` is not consumed by
TRL PPO and is explicitly reported as ignored; it cannot multiply the run to 16
completions. The compatibility shim applies YAML top-p 0.95 over TRL's internal 1.0.

Policy and value are distinct local-only loads of the same validated Qwen 0.5B
snapshot. The optimizer must match exactly policy LoRA plus value LoRA/scalar-head
trainables. Reference evaluation uses the frozen policy base with the adapter disabled;
the verifier reward has zero parameters.

The only authoritative checkpoint is role-separated `checkpoint-1`: policy adapter,
value adapter, scalar head, trainer JSON, and resume JSON. Full base-model weights,
optimizer state, symlinks, unexpected files/directories, duplicate adapters, and
full-model-sized files are rejected.

All CPU gates passed: compileall, Ruff, 270 tests, environment check, manifest
validation, GRPO/PPO dry-runs, and explicit fake guarded PPO execute. The real PPO
command is documented but remains unauthorized and was not executed.
