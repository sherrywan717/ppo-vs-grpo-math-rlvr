# Qwen2.5 0.5B tokenizer audit

- Revision: 7ae557604adf67be50417f59c2c2f167def9a775
- Snapshot: /root/autodl-tmp/cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775
- Execution: CPU-only tokenizer load; no model load, CUDA initialization, training, or generation
- Result: PASS

## Checks

- PASS - qwen_chat_template
- PASS - system_user_assistant_boundaries
- PASS - reasoning_answer_contract
- PASS - eos_token
- PASS - pad_token
- PASS - left_padding
- PASS - variable_length_batch_padding
- PASS - prompt_token_counts
- PASS - completion_extraction_boundary
- PASS - prompt_not_counted_as_completion
- PASS - max_length_truncation
- PASS - two_samples_per_dataset
- PASS - completion_parser
- PASS - shared_ppo_grpo_renderer
- PASS - required_snapshot_files

## Token behavior

- EOS: <|im_end|> (151645)
- PAD: <|endoftext|> (151643)
- Padding side: left
- Model max length: 131072
- Audit truncation: 773 to 512 tokens (limit 512)
- Anonymous per-sample lengths: token_lengths.csv

## Snapshot files

- .gitattributes - 1519 bytes
- LICENSE - 11343 bytes
- README.md - 4917 bytes
- config.json - 659 bytes
- generation_config.json - 242 bytes
- merges.txt - 1671839 bytes
- model.safetensors - 988097824 bytes
- tokenizer.json - 7031645 bytes
- tokenizer_config.json - 7305 bytes
- vocab.json - 2776833 bytes
