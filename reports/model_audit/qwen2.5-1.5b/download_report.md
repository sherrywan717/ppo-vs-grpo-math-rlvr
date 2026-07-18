# Pinned Qwen 2.5 1.5B snapshot download

Status: passed on 2026-07-18 UTC. This stage downloaded and verified the exact
formal-model snapshot only. It did not initialize CUDA, load the model/tokenizer,
generate text, instantiate a Trainer, run backward/optimizer, create a checkpoint,
or enter baseline/training.

## Frozen identity

- Repository: `Qwen/Qwen2.5-1.5B-Instruct`
- Revision: `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`
- Branch/baseline: `pivot/math-rlvr` at
  `e3cf482b395e49ee3fe7cb2241e54b77d7edc831`
- Canonical active-suite SHA256:
  `f6de8c555a70837d08c1e34e13a738a21ce247b3a09531df6244a2f1d3ef53bd`
- Active-suite raw-file SHA256:
  `a78df532c2d31a11a63790993d9ce2b1425844c46d5013fec6820a3609dffc49`

The revision was parsed from the templates referenced by all four active resolved
configs, not copied from chat. Each resolved-config SHA and referenced-template SHA
matched its active-suite evidence. All four templates selected the same repository,
revision, and `local_files_only=true`; the model identity registry matched exactly.
The revision is 40 lowercase hexadecimal characters and matches the required prefix
`989aa798` and suffix `71aa306`.

## Download

- Cache root: `/root/autodl-tmp/cache/huggingface`
- Snapshot:
  `/root/autodl-tmp/cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306`
- Downloader: `huggingface_hub.snapshot_download` 0.36.2, one worker, resumable
- Attempts: 1
- Outer retries: 0
- Download elapsed time: 797.248814 seconds
- Snapshot logical size: 3,098,973,447 bytes
- Repository cache physical size: 3,098,973,447 bytes
- Remaining `/root/autodl-tmp` space after download: 220,973,981,696 bytes

The downloader emitted one non-blocking warning that the explicit
`resume_download` argument is deprecated because resume is now automatic. No
dependency was installed or upgraded.

## Offline verification

- `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`
- Exact revision resolved with `local_files_only=True` to the canonical snapshot
- `config.json`, `tokenizer_config.json`, `tokenizer.json`,
  `vocab.json`, and `merges.txt` are present
- `model_type=qwen2`
- Architecture contains `Qwen2ForCausalLM`
- Tokenizer class is `Qwen2Tokenizer`; a chat template is present
- `model.safetensors` is present and is 3,087,467,144 bytes
- This revision is a single-file safetensors snapshot and declares no shard index;
  therefore there are no indexed shards to be missing
- No `.incomplete` file remains
- No Hugging Face download Python process remains
- A second offline resolution returned the identical snapshot without network access

The inventory intentionally records filenames and sizes without hashing the multi-GB
base-model weight. The cache was not copied into Git, a run directory, a checkpoint,
or the AutoDL backup filesystem.

## Active resolved-config SHA256 values

- PPO seed 42: `717502aa665e9d5ef967e04a5ab27aa53329ccb061bda228db3c715f4dab967b`
- GRPO seed 42: `6776f8894e9ac725a39748b06b57b62782cea2dab61faf51fd3cc3ceb5ae58bf`
- GRPO seed 123: `4ce0918f7284220c36555b9f23db181354168ebe252d7244ac3ac9587be236fa`
- PPO seed 123: `a68524e85e427e335abf6447aa2cc391686fd3aa4da6d42efb0e522beec1a0b3`
