# PPO Value Model and Optimizer Contract

The guarded Qwen 0.5B PPO smoke loads policy and value backbones as distinct objects
from the same validated local snapshot revision
`7ae557604adf67be50417f59c2c2f167def9a775`. Both loaders pass
`local_files_only=True`; no repository-ID or network fallback is permitted.

The policy uses BF16 causal LM LoRA `r=16`, alpha 32, dropout 0 on q/k/v/o. Reference
log-probabilities use the same frozen policy base under PEFT's disabled-adapter context;
there is no independent trainable reference object. The value role uses
`AutoModelForSequenceClassification(num_labels=1)`, independent LoRA `r=8`, alpha
16, dropout 0 on q/v, and a trainable scalar score head. TRL value logits must have
shape `[batch, sequence, 1]`.

The verifier reward adapter is parameter-free. Before training, the runner proves:

- policy trainables are LoRA-only;
- value trainables are value LoRA or scalar-head parameters;
- policy/value trainable parameter objects are disjoint;
- reward/reference trainable counts are zero;
- optimizer parameters equal exactly the union of policy and value trainables.

The authoritative checkpoint layout is:

```text
checkpoint-1/
  policy_adapter/{adapter_config.json,adapter_model.safetensors}
  value_adapter/{adapter_config.json,adapter_model.safetensors}
  value_head/{config.json,value_head.safetensors}
  trainer_state.json
  resume_manifest.json
```

No base-model weights or optimizer state are saved. CPU mocks validate local-only
loader kwargs, LoRA targets, scalar-head shape, optimizer partition, checkpoint
inventory, fake reload, and rejection of unexpected/full-model-like files. No real
model was loaded and no PPO update was executed during this implementation gate.
