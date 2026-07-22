# GRPO-v2 warm-start post-freeze capacity amendment

Stage O.1 remains an immutable failed capacity audit: 48/256 targets exceeded 256 active tokens and one prompt exceeded 832. No truncation occurred.

The authorized capacity-only amendment changes `max_prompt_length` from 832 to **928** and `max_target_length` from 256 to **640**. `max_sequence_length` remains **1,088** and is enforced against each actual concatenated sequence; 928+640 is not a new context budget. Observed maxima are prompt 914 (14-token margin), active target including EOS 609 (31-token margin), and actual combined 1,019 (69-token margin). The round caps preserve explicit margin without changing any text, sample, order, epoch, batch, GA, step, LoRA, model, parser, verifier, reward, GRPO budget, or hidden-test identity.

- Old config SHA256: `f3a68c204f356932e037c08ae87aa539aaac4b932aa9e982d2de6cb928763c35`
- New config SHA256: `c8e3e0a52d55f46b201c1b4a95e0f9b2f910ae558e477b55a35d03fd4ec8549a`
- Old contract SHA256: `8de472421e2fd7f1a900fdda981cb76605de195e77e0b57c8c5a95f9504742c2`
- New contract SHA256: `6cb07b3ffc1e86113e09327f356171f9b8ee99e16f85b0697f61408187cc1f1d`
- Non-capacity identity SHA256: `102e754bb930bb8c8102cadca44c19d55d7f4793bf685c9a40bda672e761836b` (unchanged)

The full amended re-audit passed 256/256 with zero prompt/target/combined over-cap rows and zero truncation. Final amended config SHA256: `c8e3e0a52d55f46b201c1b4a95e0f9b2f910ae558e477b55a35d03fd4ec8549a`; final amended contract SHA256: `6cb07b3ffc1e86113e09327f356171f9b8ee99e16f85b0697f61408187cc1f1d`. All listed manifest, curriculum, GRPO-v2, evaluation, model, prompt, reward, parser, and verifier identities remain unchanged.
