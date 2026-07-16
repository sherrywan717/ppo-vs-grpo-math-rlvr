# Accelerate 1.14.0 PPO microbatch semantics

`Matched 0.5B pilot - not the final benchmark`

## CPU reproduction

A real deterministic CPU `Accelerator`, tiny linear parameter, GA=4 and real
`AcceleratedOptimizer` produced four accumulate entries and four backward calls. Its
wrapper `.step()` was called four times; the underlying optimizer was called once,
only on microbatch 3 with `sync_gradients=true`. Parameters did not change on the first
three wrapper calls and changed on the fourth. Thus the first call observed by a
bottom optimizer hook legitimately has `sync_gradients=true`; it cannot count prior
microbatches.

The exact TRL-like shape exposed an additional end-of-loader interaction. TRL consumes
one prepared rollout batch before its inner PPO loop. With Accelerate's default
`sync_with_dataloader=true`, `end_of_dataloader=true` forced all four inner contexts to
sync and caused four tiny underlying updates. Setting the existing gradient
accumulation plugin to `sync_with_dataloader=false` restored the frozen semantics:

| microbatch | size | sync | underlying update | parameter changed |
|---:|---:|:---:|:---:|:---:|
| 0 | 4 | false | no | no |
| 1 | 4 | false | no | no |
| 2 | 4 | false | no | no |
| 3 | 4 | true | yes | yes |

## Old and new guard boundaries

The failed guard replaced `self.optimizer.step`, which is the
`AcceleratedOptimizer.step` wrapper, not only the bottom optimizer. Its first call was
already synchronized because the consumed DataLoader forced early sync; the call was
therefore not a safe microbatch clock.

The compatibility shim now counts microbatches only from successful
`accelerator.backward` events. A gradient-enabled TRL training forward supplies the
actual `query_responses.shape[0]` batch size. It requires four backward events, sizes
`[4,4,4,4]`, 16 processed samples and sync trace `[false,false,false,true]`. The bottom
optimizer hook is separate and enforces exactly one real update; it no longer infers
microbatch count. Loop keys still enforce exactly one epoch, one minibatch, one outer
update and one global step.

The existing `ppo_loader_contract.json` receives the validated backward-event evidence
after a successful train call; no new artifact file or schema family was added.

## Verification

- Real tiny CPU standard GA4: backward 4, wrapper calls 4, bottom updates 1.
- Real tiny CPU consumed-loader default diagnostic: backward 4, bottom updates 4.
- Real tiny CPU fixed TRL-like path: backward 4 × batch 4, bottom updates 1.
- Negative tests reject 1/3/5 backward events, non-16 sample totals, early sync,
  optimizer step 2, epoch 2 and minibatch 2.
- Stage D PPO, fake matched PPO finalization and all GRPO regressions passed.
- Full pytest: 358 tests. Ruff, compileall, check_env, manifest validation, six pilot
  dry-runs and two Stage D dry-runs passed.
- The tiny CPU reproduction intentionally performed tiny backward/optimizer work.
  Qwen/tokenizer loads, Qwen generation, the real TRL PPO train loop and CUDA calls
  remained zero.
- Frozen configs/manifest and all four historical PPO seed-42 failure trees are
  unchanged.
