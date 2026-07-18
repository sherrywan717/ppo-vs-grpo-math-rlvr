# Post-freeze prompt-length capacity amendment

This is a capacity-only amendment after the frozen tokenizer audit. It does not alter scientific content.

## Why the amendment is required

- The pinned Qwen 1.5B tokenizer found three frozen problems above the old 512-token cap.
- The maximum is 800 tokens at `math:HuggingFaceH4/MATH-500:test:219`.
- Formal train also contains a 713-token prompt, so both PPO and GRPO training templates require the same amendment.
- The smallest reasonable next 64-token boundary is 832.
- 832 prompt + 256 completion = 1,088 tokens, below the model context limit of 32,768.

## Identity changes

| Identity | Old | New |
|---|---|---|
| max_prompt_length | `512` | `832` |
| evaluation canonical SHA256 | `b87ea305d4253f41f337fde3a0850ceb7e6925c87f6a9b8e0e6ac452e730ab50` | `d8ba5ab80ab0553d2ec7246fb4876956dcbc5dd0bcf8642fd33c4ec19da6fe44` |
| evaluation raw SHA256 | `3b1a682b9ecebb51d8cd3de65aa57a201049e8efa5ab4536081598450605821f` | `85100dd0f613f295a7219a45a42a03e3ad4a45e24893c7f296e1d8da9a1f4a35` |
| active-suite canonical SHA256 | `f6de8c555a70837d08c1e34e13a738a21ce247b3a09531df6244a2f1d3ef53bd` | `1d7c29f76d9bfbf11e1838cd6b0bc8f3da6d0133e0605420e9ed838de729d600` |
| active-suite raw SHA256 | `a78df532c2d31a11a63790993d9ce2b1425844c46d5013fec6820a3609dffc49` | `11869c63f4365aee5d4bf8e13fe263c9d0397164a18a88b419da07218f6a2017` |
| ppo_seed_42 config SHA256 | `717502aa665e9d5ef967e04a5ab27aa53329ccb061bda228db3c715f4dab967b` | `1093e87a8363a0a2a6ab640a6f723c04cb6cfb22edef2e38a8c3a0062693ec43` |
| grpo_seed_42 config SHA256 | `6776f8894e9ac725a39748b06b57b62782cea2dab61faf51fd3cc3ceb5ae58bf` | `3371d23166d01834c67830eb8dfd51a02d4af483687b0b29e941194174099199` |
| grpo_seed_123 config SHA256 | `4ce0918f7284220c36555b9f23db181354168ebe252d7244ac3ac9587be236fa` | `cc95138f50f37fafa76766d3a08b0995ffd5e0bf87cd7b9050acedb5e0bbc75e` |
| ppo_seed_123 config SHA256 | `a68524e85e427e335abf6447aa2cc391686fd3aa4da6d42efb0e522beec1a0b3` | `3d6cc1f30f7b72bfadb5191613298ac3f64a1ba3c699cc8d1e30ce147218c15e` |

## Fairness and evidence boundary

- Only the capacity upper bound increases; no prompt text, renderer, sampling, reward, parser, verifier, dataset content/order, max completion length, LoRA, optimizer, or budget changes.
- Token IDs for every previously fitting prompt are unchanged because tokenization remains non-truncating.
- PPO and GRPO retain one shared scientific protocol and the same 832-token cap.
- The old 642 completion rows are not reused or mixed across config identities.
- Both engineering-failure runs remain immutable and excluded from baseline statistics.

See [the full tokenizer audit](prompt_length_audit.md) and [the exact config diff](config_diff_prompt_length.patch).
