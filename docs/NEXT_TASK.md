# Next task: CPU-only freeze the guarded 128-update GRPO-v2 runtime

Stage P warm-start and both Stage P.1 matched dev evaluations are complete and immutable. Base achieved 6/128 candidate-0 pass@1; warm-start achieved 8/128 with a +1.5625 pp paired delta whose bootstrap interval includes zero. This supports protocol-following improvement and only an uncertain dev gain.

The sole next task is CPU-only: implement and freeze the model-bound GRPO-v2 CLI/runtime from `configs/grpo_v2/grpo_v2_seed42.json`, initialized only from warm-start checkpoint-16 policy adapter SHA `44066dd13d8cfa4f5c40f10cad705eea617c37ce2e2f85ff5407751fb5a972b9`. It must preserve the pre-registered 128-update/2,048-completion/524,288-token contract, incremental evidence, checkpoints/dev at 32/64/96/128, fresh GRPO optimizer, exact resume and adapter-only safety.

Do not execute GRPO-v2, rerun warm-start/dev, access hidden test, or initialize CUDA during that freeze. A later real GRPO-v2 run requires new explicit GPU authorization.
