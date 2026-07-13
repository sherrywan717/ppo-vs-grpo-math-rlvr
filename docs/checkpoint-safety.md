# GRPO Checkpoint Safety

The authoritative checkpoint is the Trainer-created top-level `checkpoint-1` produced by the frozen `save_strategy=steps`, `save_steps=1`, and `max_steps=1` contract. The runner does not call `save_model` after training and does not create a second adapter copy.

A successful inventory requires exactly one checkpoint directory, one `adapter_model.safetensors`, one `adapter_config.json`, and no duplicate adapter SHA256.

## training_args.bin

Only the exact basename `training_args.bin` is accepted, and only when all conditions hold:

- it is directly inside the canonical checkpoint root;
- it is a regular file and not a symlink;
- its resolved parent is the canonical checkpoint root inside the current run;
- its size is at most 1 MiB;
- it is hashed as bytes and classified only as `trainer_metadata`.

The inventory records name, byte size, SHA256, and classification. It never calls `torch.load`, `pickle.load`, or another deserializer.

Unknown `.bin` files, `pytorch_model.bin`, oversized or symlinked `training_args.bin`, path escapes, unknown safetensors, duplicate adapters, and full-size weights fail closed.
