# Four anchors including the bonus, plus fifteen speculative tokens

Branch: `four-anchor-15-train-benchmarks`.
Checkpoint format: `3`; `anchor_layout: target_four`;
inference version: `target_four_materialized_v3`.

## Exact layout

Both drafting passes have **19 query positions: four anchors and 15 masks**.
The pending correction/bonus is one of the four first-pass anchors.
Exactly 15 candidates go to native vLLM verification. Maximum reported acceptance
including one correction/bonus is **16**. Conditioning anchors are excluded.

The bonus occupies the fourth anchor slot; it is not an additional query.
Suppose the first anchors are t1–t4, where t4 is the bonus.

| Stage | Four anchor positions | Feature added to the token embedding | Prediction slots |
|---|---|---|---|
| Round one | t1, t2, t3, t4 | Actual target h1, h2, h3, **h4**, including the bonus | t5–t19 (15 masks) |
| Round two, retain through t12 | t9, t10, t11, t12 | Projected first-pass drafter d9, d10, d11, d12 | t13–t27 (15 masks) |
| Final merged candidates | Anchors excluded | Keep first-pass t5–t12 and redrafted t13–t19 | Exactly 15 candidates |

Every valid first-round anchor has its actual same-position target state.
The second round preserves the original training rule: anchors from the
processed prefix keep target states, and anchors from the retained first-pass
proposals use projected first-pass drafter states. For example, retaining only
t5 gives second anchors t2, t3, t4, t5 with h2, h3, h4, d5. If at least four
proposals are retained, all four second-round anchor features are drafter states.
Early missing anchors are padded and masked. The retained draft prefix is
selected by confidence and has not yet been accepted by the target verifier.

A newly sampled correction/bonus does **not** yet have its own target state:
verification computed states for its input tokens, and then sampled this new
token from those states. This implementation therefore makes **one extra target
forward** to process the bonus before drafting. It caches the bonus's actual
state along with the confirmed prefix; rejected suffix states are excluded.
Native vLLM subsequently processes `[bonus, 15 candidates]` for verification,
replaying the bonus in its cache slot so scheduler accounting stays correct.
If its KV page has not been allocated, the adapter falls back to an ordinary
target step and retries drafting on the next cycle.

The extra target computation does **not** add a query slot: both draft rounds
are still **4 + 15 = 19**. Its cost is included in throughput and its actual
count appears as `extra_target_forwards` in the benchmark results. Acceptance
is measured per native verification cycle, excluding these extra forwards.

This differs from the preserved `known_bonus_cached_v1` experiment, which used
four processed past anchors **plus a separate bonus slot**, for 20 queries.
Old 20-query and format-2 embedding-only checkpoints cannot be resumed or
served as the new format. Start a fresh training run for this recipe.
The published result tables remain measurements of the old method.

## Training

- Shared regular DFlash drafter, five bidirectional layers, no prefix module or MTP.
- Target Qwen3-4B; decoder output indices 1, 9, 17, 25, 33.
- Four anchors and **15 supervised speculative positions** in round one; no bonus-token loss.
- Round two uses the same four-position feature-source rules as inference.
- The original cached training objective is retained with proposals reduced
  from 16 to 15. Each view supervises all 15 predicted positions where labels
  are valid. A shifted second view can extend beyond the final inference window,
  as in the original training recipe.
- When no first-pass proposals are retained, the second training view is
  identical and reuses the first computation; both loss weights still apply.
- The original target prefix, including the processed bonus, is visible to
  the drafter. Target states at speculative positions are excluded from its
  attention memory and anchor features; they supply teacher supervision only.
- Loss: `0.1 CE + 0.9 TV` on each view, using the existing DSpark-style
  distribution loss, distance weights and detached first-pass anchor states.
- Existing confidence boundary selection and exploration remain. During training,
  the retained prefix is capped at its first dataset-token mismatch so teacher
  forcing stays valid. Inference uses confidence alone, followed by verification.
- FSDP, BF16 computation, FP32 optimizer parameters; existing AdamW optimizer,
  learning rate 0.0006, cosine schedule, five epochs.
- **9216 packed tokens per training rank**, up to **1536 sampled blocks per rank**.
  These settings are not the four-token conditioning-anchor count.
- Default `TRAIN_DATA_RATIO=1.0`: use all trainable rows; no held-out dataset
  validation pass. The actual packing plan determines update counts.
- Target NPUs 0,1; trainers 2,3,4,5,6; benchmark evaluator NPU 7.
  The evaluator is separate so training can continue during evaluation.

## Start on Ascend

Activate `attention_prefix_ali`, then run:

```bash
FORK=/home/n84449292/a84384644/attention_head_forked
ROOT="$FORK/four-anchor-15-train-benchmarks"
OLD_RUN="$FORK/four-anchor-dflash-pipeline/output/four_anchor_dflash_20261031_025300_2111006"
BENCHMARK_PLAN="$OLD_RUN/evaluation/cached_suite_20261103_115831_3028663/plan.json" \
  bash "$ROOT/four_anchor/scripts/train_bonus19_ascend.sh"
```

`BENCHMARK_PLAN` reuses the **exact prompt token files** from the measured suite
that was copied into the published results branch. Those original files must
still exist, and their recorded hashes must match. Omitting the variable
discovers the same eight benchmark types from local caches and freezes a fresh
prompt plan before training. It never downloads datasets.

Default evaluation: all eight cached benchmarks, up to 128 prompts each
(30 AIME25 and 80 MT-Bench in the previous cache), 2048 maximum new tokens,
concurrency one, greedy BF16, thinking off, target and drafter graphs off,
same prompt formatting, warmup and vLLM acceptance accounting as the prior suite.
`EVAL_LIMIT=0` selects all rows in each chosen cached source; omit
`BENCHMARK_PLAN` when changing the sample count. A frozen manifest may be a subset.
MT-Bench remains first-turn only. This measures **acceptance and throughput**,
not task accuracy. It does not claim parity or speedup without a matching baseline.

`CHECK_ONLY=1` runs training configuration/import/packing checks without starting
training or benchmark engines. An actual FSDP forward/backward/checkpoint-restore
preflight on the trainer NPUs runs automatically before normal training starts.

The launcher prints the new output directory. It waits for all queued epoch
evaluations after training ends. Benchmark failures are recorded, later epochs
are still attempted, and the launcher exits nonzero if any evaluated epoch fails.

## Checkpoints and results

For the printed `OUTPUT_DIR`:

| Path | Contents |
|---|---|
| `checkpoints/checkpoint-XXXXXX/` | Model, config, optimizer/RNG state and file hashes |
| `checkpoints/latest_checkpoint.txt` | Latest saved checkpoint path |
| `checkpoints/logs/four_anchor_dflash/` | TensorBoard training metrics |
| `evaluation/prepared/plan.json` | Frozen prompts and evaluation settings for all epochs |
| `evaluation/queue/epoch-NNN.json` | Job published only after a complete epoch checkpoint |
| `evaluation/epoch-NNN/<attempt>/summary.md` | Per-epoch benchmark table |
| `evaluation/epoch-NNN/<attempt>/<benchmark>/two_pass/` | Per-prompt JSON, summary, engine settings and log |
| `evaluation/epoch_results.md` | Cumulative epoch-by-benchmark table |
| `evaluation/epoch_results.csv` | Same results with checkpoint/result paths, LF line endings |
| `evaluation/epoch_results.json` | Detailed state and rows, including failures |
| `evaluation/worker.log` | Evaluator queue progress |

Every epoch keeps its checkpoint and benchmark results. No selection based on
training-dataset validation is performed. Failed attempts retain their logs;
the queue worker accepts `--retry-failed` for a later retry.

To resume **a format-3 checkpoint from this new training recipe**, keep the same training
settings and NPU count, and set `RESUME=/.../checkpoints/checkpoint-XXXXXX`.
The launcher infers the original output directory, preserves the frozen prompts,
and skips already completed evaluations. Partial-epoch stops save checkpoints
but do not publish an epoch evaluation until that epoch has actually finished.

Acceptance is comparable across epochs using the frozen samples. Evaluation
throughput measured while training is running can be affected by shared host
resources; repeat the selected final checkpoint with training stopped for a
controlled speed comparison. Reducing the number of training NPUs from six to
five and using all training rows changes the step count and training duration.

## Validation limits

The targeted CPU regression suite passed **252 tests**. Shell launchers also
passed syntax checks. FSDP CPU checks use single-rank `NO_SHARD`; they do not
establish multi-NPU `FULL_SHARD` correctness or performance.

CPU tests cover all 15 restart positions, train/serve hidden-state parity,
future-feature isolation, cached/reference loss and gradients, portable model
loading, greedy output parity with bonus materialization and verification,
actual extra-forward accounting excluding warmup, paged-cache metadata,
frozen offline prompts, queue failures, and interrupted training/resume.

No new Qwen3-4B/Ascend benchmark result is claimed by this package. The local
environment has no Ascend hardware; its multiprocess Gloo test was blocked by
socket permissions. The included on-server FSDP preflight checks the new objective
on the actual trainer NPUs before allocating the target server.
