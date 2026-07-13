# Prompt A/B Generation Diagnostic Artifact Schema

This schema applies only to the guarded, generation-only v0/v1 diagnostic. It is a
small-sample integration diagnostic, not a training run or a statistically significant
PPO-versus-GRPO result.

## Authorization and immutable contract

The only real entry point is `python -m math_rlvr.evaluation.prompt_ab` with the fixed
`configs/diagnostics/prompt_ab.yaml`. Real generation requires both
`--generate-only` and `--confirm-prompt-diagnostic`; the GRPO
`--confirm-single-update` flag is explicitly rejected. Before delayed CUDA/model imports,
the runner requires a clean `pivot/math-rlvr` worktree, both offline environment flags,
and the exact validated local Qwen 0.5B snapshot.

The contract is two ordered conditions, two fixed Countdown train prompts per condition,
four completions per prompt, 128 tokens per completion, 16 completions total, at most
2,048 generated tokens, 120 seconds, and a 3.5 GiB nvidia-smi stop gate. It loads the
BF16 base model only, sets `eval()`, freezes all parameters, and executes generation
inside `torch.inference_mode()`. Trainer, LoRA, optimizer, backward, checkpoint, model
writes, automatic retry, and prompt activation are forbidden.

## Completion evidence

Each `completions.jsonl` record contains condition and prompt version, problem/prompt
hash, generation index, matched Python/CPU/CUDA seed, condition order, rendered prompt
hash, input token count, exact completion IDs/count, decoded raw text, EOS/truncation
state, parser status/detail, RewardStatus/scalar/verifier detail, tag fields,
complete-envelope/answer-only/prose flags, expression validity, number-usage validity,
and final correctness.

Generated sequences are split at the padded input tensor width. The attention-mask sum
records the unpadded input token count, so left padding and unequal prompt lengths never
become completion IDs. The first EOS is retained and post-EOS padding is excluded.
Completion text is parsed and verified as data and is never executed.

## Metrics and candidate gate

Per-condition metrics include complete-envelope and individual tag rates, answer-only and
outside-prose rates, truncation, format/expression/number validity, pass@1/pass@4,
RewardStatus counts, reward mean/std/variance, per-problem group variance, nonzero
advantage-potential groups, and token length mean/median/max.

v1 is never activated automatically. It is merely eligible for a later GRPO review when
its complete-envelope rate exceeds v0, at least one envelope is complete, truncation does
not increase, and at least one problem group has nonzero reward variance. All-WRONG_ANSWER
v1 output is reported as a no-advantage warning.

## Lifecycle

Success requires 16 complete records, all budget and zero-training assertions, reloadable
primitive JSON, figures labelled `generation-only prompt diagnostic; no training`,
checksums, a secret/model/checkpoint scan, and a verified tar.gz plus SHA256. Artifact or
backup failure leaves status as failure. Full artifacts live only below
`/root/autodl-tmp/runs/math_rlvr`; Git receives only the bounded report directory.


## Paired and group evidence

`paired_comparison.json` and `.csv` contain exactly eight pairs joined only on
`problem_id`, `generation_index`, and `matched_seed`. Each pair records both rendered
prompt hashes, completion indices, exact token counts, raw text, RewardStatus/reward,
format/truncation/expression/number/final flags, plus status transition and reward,
token, and format deltas. Missing, duplicate, or mismatched pairs fail closed.

`per_problem_rewards.json` preserves four rewards and statuses in generation order for
each condition/problem. It reports population mean/min/max/std/variance, unique reward
count, `zero_variance`, and `zero_advantage_group`. A group is zero-advantage exactly
when all four scalar rewards are identical; global variance is never substituted.

## Capability and consistency gates

`configs/diagnostics/prompt_ab_capabilities.json` uses schema
`prompt-ab-evidence-contract-v1`. All seven reviewed capability booleans must be true;
tests independently turn each one off and require a preflight rejection before run ID,
CUDA, model load, or generation. An independent validator re-reads summary, manifest,
JSONL, JSON/CSV metrics and pairs, seed map, prompt hashes, resource evidence, backup
manifest, and checksums. Counts, token totals, RewardStatus totals, ordered group reward
lists, zero-advantage counts, seeds/pairs, prompt hashes, planned/actual budget, and all
zero-side-effect counters must agree before success.

## Parent/worker lifecycle

The parent process does not initialize CUDA and starts only a fixed Python worker using
`multiprocessing` spawn—never `shell=True` or a user command. Inside the worker,
`CudaAllocatorEvidence` records normalized integer device index, current/peak
allocated/reserved bytes and MiB, separately from nvidia-smi. The worker finally drops
generated/model/tokenizer references, runs garbage collection, synchronizes, empties the
cache, and requires current allocated/reserved to be zero.

After worker exit, the parent proves the PID is gone, absent from nvidia-smi compute
processes, and device-0 memory is restored to its pre-run baseline. Only then does it
perform consistency validation, finalization, backup and Git-safe publication.

Runtime failures after run creation are also finalized and backed up with a `-failure`
archive name and SHA256. A primitive-only minimal failure record prevents recursive
finalization errors. Static preflight rejection occurs before run creation and therefore
does not create an empty archive.

## Isolated-worker GPU cleanup evidence

`pytorch_allocator.json` records worker-side, pre-process-exit current and peak allocator values. Nonzero cleanup current values are retained verbatim with `worker_allocator_nonzero_before_process_exit`; CUDA context/cache can be released only as the worker exits. This warning is not an independent final failure gate.

`post_worker_gpu_verification.json` is authoritative: the parent must remain CUDA-uninitialized, prove worker PID exit and compute-process absence, reject newly introduced compute PIDs, and verify GPU memory returned to the pre-run baseline. A parent-gate failure is fatal even when worker allocator current values are zero.
