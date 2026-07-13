# Qwen 0.5B staged-reward GRPO single-update smoke

Run ID: `grpo_single_update_qwen25_05b_20260713T122258Z`  
Scope: single-update smoke diagnostic; not a formal algorithm comparison.

## Conclusions

- Infrastructure smoke: **PASS**.
- Reward-integration smoke: **PASS**.
- Learning-signal smoke: **PASS**.
- PPO was not started.

The online run produced 8 completion records and 276 exact generated tokens, consumed 4 microsteps, and performed exactly 1 optimizer step / global step. Both problem groups had nonzero shaped-reward variance, no group was zero-advantage, and the finite nonzero gradient norm was 4.6481008530.

## Frozen identity

- Prompt: `prompt_v1_strict_concise` / `6842002e4591630a4105a4ca8fdf4cab91676b3902708ec0cb7f7b458864ecd7`
- Renderer: `math_rlvr.prompt.chat_template.v1`
- Reward: `shaped_v2_staged` / `90af0614676279eb8a47636acfdbeaded6d92237d3b16f027d79557057ca0e14`
- Components: `{"answer_block": 0.05, "correctness": 0.8, "exact_number_usage": 0.05, "strict_protocol": 0.05, "valid_expression": 0.05}`

## Budget and training metrics

- Completions/tokens: 8 / 276 (caps 8 / 1024)
- Microsteps / optimizer steps / global step: 4 / 1 / 1
- Loss: 0.6453
- Grad norm: 4.648100852966309
- Entropy: 0.5070152431726456
- Learning rate: 1e-05
- KL: unavailable (`beta=0.0`, `kl=null`); GRPO beta=0.0; TRL 0.24.0 does not compute reference-model KL.
- Canonical status distribution: `{'format_error': 8}`. Strict status remains unchanged even where shaped reward is nonzero.

## Online group rewards

- `countdown:train:0`: rewards `[0.1, 0.1, 0.15, 0.0]`, mean 0.087500, population variance 0.00296875, zero-advantage=false, diagnostic normalized advantages `[0.229416, 0.229416, 1.147079, -1.60591]`.
- `countdown:train:1`: rewards `[0.1, 0.05, 0.1, 0.05]`, mean 0.075000, population variance 0.00062500, zero-advantage=false, diagnostic normalized advantages `[1.0, -1.0, 1.0, -1.0]`.

Normalized advantages above are diagnostic population-z scores derived from saved online rewards; they are not claimed as a dump of TRL's private internal tensor.

## LoRA and checkpoint

- LoRA: r=16, alpha=32, dropout=0, targets q/k/v/o projections.
- Trainable parameters: 2,162,688 / 496,195,456 including adapter (0.4358540518%).
- Exactly one authoritative `checkpoint-1`; 12 files, 24570947 bytes; duplicate checkpoint count 0.
- One adapter only; no full base-model weight file.

- `README.md` — 5417 bytes — `edd6a919864f6dcca89367b5c2b7a2d4059ad1e78a5bd834d9705af1c54e0ac3` — tokenizer_or_metadata
- `adapter_config.json` — 1152 bytes — `3882d617ea651fc2bb7bd47f5d20f40e8a6fd6cd9ae8d9ec0be59100a5f5644c` — adapter_config
- `adapter_model.safetensors` — 8676008 bytes — `c9ed148676fd5c6d704125e2bab7a6be5b625f5a252f28f25d3471dc635d323d` — lora_adapter
- `added_tokens.json` — 605 bytes — `58b54bbe36fc752f79a24a271ef66a0a0830054b4dfad94bde757d851968060b` — tokenizer_or_metadata
- `chat_template.jinja` — 2507 bytes — `cd8e9439f0570856fd70470bf8889ebd8b5d1107207f67a5efb46e342330527f` — tokenizer_or_metadata
- `merges.txt` — 1671853 bytes — `8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5` — tokenizer_or_metadata
- `special_tokens_map.json` — 613 bytes — `76862e765266b85aa9459767e33cbaf13970f327a0e88d1c65846c2ddd3a1ecd` — tokenizer_or_metadata
- `tokenizer.json` — 11422161 bytes — `eb1ea0ffbb9ce6886361fefe110952fa83e3bcac0231c7f24b68cfa6e06cf0c9` — tokenizer_or_metadata
- `tokenizer_config.json` — 4686 bytes — `0a04a9d7d4a62b28482bdfe726c122756de85714fb64166ace92ae75b8f57614` — tokenizer_or_metadata
- `trainer_state.json` — 1671 bytes — `a62e574cb1839ca24d4afe99e02e32cf0980c9c859278be79cc8c2ad0e3c66f3` — trainer_state
- `training_args.bin` — 7441 bytes — `ef9f23f34a358b9668c7f893cb392f192875a339c2ff067d4ff35689c33320c0` — trainer_metadata
- `vocab.json` — 2776833 bytes — `ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910` — tokenizer_or_metadata

## Resources and release

- BudgetGuard wall time: 10.604900 s; resource monitor window: 11.028895 s.
- GPU-hours: 0.002945806; cost at ¥8.88/GPU-hour: ¥0.026159.
- PyTorch peak allocated/reserved: 1501.549 / 1936.000 MiB.
- nvidia-smi sampled peak memory/utilization: 2613 MiB / 37%.
- Worker pre-exit allocator current allocated/reserved: 64 / 108 MiB; recorded as `worker_allocator_nonzero_before_process_exit`.
- After process exit: nvidia-smi 0 MiB and no compute process; GPU release verified.

## Warnings

- All 8 canonical results remain `FORMAT_ERROR`; shaped reward creates learning signal without changing strict evaluation semantics.
- One completion reached the 128-token cap.
- Transformers aligned PAD/BOS/EOS config (`pad_token_id=151643`); no budget or evidence failure resulted.
- `model_init_kwargs` was ignored because the model was already instantiated.
- This run follows the staged-reward post-smoke intervention and is only diagnostically comparable to the old-reward run.

## Raw completions

### Completion 0 — countdown:train:0 / generation 0

- Tokens: 17
- Canonical status: `format_error`
- Scalar reward: 0.10
- Components: answer=0.05, strict=0.00, expression=0.05, number-use=0.00, correctness=0.00
- Verifier: each tag must appear exactly once

```text
<answer>
13 + 13 / 6
</answer>
```

### Completion 1 — countdown:train:0 / generation 1

- Tokens: 16
- Canonical status: `format_error`
- Scalar reward: 0.10
- Components: answer=0.05, strict=0.00, expression=0.05, number-use=0.00, correctness=0.00
- Verifier: each tag must appear exactly once

```text
<answer>13 * 19 / 5</answer>
```

### Completion 2 — countdown:train:0 / generation 2

- Tokens: 20
- Canonical status: `format_error`
- Scalar reward: 0.15
- Components: answer=0.05, strict=0.00, expression=0.05, number-use=0.05, correctness=0.00
- Verifier: each tag must appear exactly once

```text
<answer>
13 * (6 - 9) + 5
</answer>
```

### Completion 3 — countdown:train:0 / generation 3

- Tokens: 128
- Canonical status: `format_error`
- Scalar reward: 0.00
- Components: answer=0.00, strict=0.00, expression=0.00, number-use=0.00, correctness=0.00
- Verifier: each tag must appear exactly once

```text
<reasoning>
- Use 5, 9, 13, 6 in any order
- Apply operations: +, -, *, /
- Make 15/26
- Possible combinations: 5 + 13 - 6, 5 * 6 + 13, (5 * 6) - 13
- Evaluate combinations:
  5 + 13 - 6 = 12
  5 * 6 + 13 = 37
  (5 * 6) - 13 = 23
- Determine the
```

### Completion 4 — countdown:train:1 / generation 0

- Tokens: 16
- Canonical status: `format_error`
- Scalar reward: 0.10
- Components: answer=0.05, strict=0.00, expression=0.05, number-use=0.00, correctness=0.00
- Verifier: each tag must appear exactly once

```text
<answer>4 - 11 * 13</answer>
```

### Completion 5 — countdown:train:1 / generation 1

- Tokens: 22
- Canonical status: `format_error`
- Scalar reward: 0.05
- Components: answer=0.05, strict=0.00, expression=0.00, number-use=0.00, correctness=0.00
- Verifier: each tag must appear exactly once

```text
<answer>
-4 = -(16 + 4) - 11
</answer>
```

### Completion 6 — countdown:train:1 / generation 2

- Tokens: 16
- Canonical status: `format_error`
- Scalar reward: 0.10
- Components: answer=0.05, strict=0.00, expression=0.05, number-use=0.00, correctness=0.00
- Verifier: each tag must appear exactly once

```text
<answer>11 - 13 + 4</answer>
```

### Completion 7 — countdown:train:1 / generation 3

- Tokens: 41
- Canonical status: `format_error`
- Scalar reward: 0.05
- Components: answer=0.05, strict=0.00, expression=0.00, number-use=0.00, correctness=0.00
- Verifier: each tag must appear exactly once

```text
<reasoning>
-4 = -(4 - 11) * 13
<answer>
-4 = -(4 - 11) * 13
</answer>
```

