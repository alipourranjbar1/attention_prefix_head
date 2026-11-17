# DLoop: document-isolated token-budget packing

The original Ascend launcher processed one saved record per rank and treated
MAX_LENGTH as a truncation cap. It did not make a 9,216-token batch. This entry
point corrects batching; it does not introduce the four-anchor method.

## Launch

On server 104, activate the existing environment and use a newly extracted
copy of this ZIP. Stop the old DLoop job in its own terminal if it still owns
the requested NPUs. The launcher itself never kills another job.

```bash
(
  set -eo pipefail
  source /home/n84449292/miniforge3/etc/profile.d/conda.sh
  conda activate four_anchor_ali
  unset DRAFT_INIT OUTPUT_DIR RESUME MAX_STEPS CHECK_ONLY
  NPUS=0,1,2,3,4,5,6,7 INIT_MODE=scratch \
  PACKED_TOKEN_BUDGET=9216 MAX_LENGTH=3072 NUM_ANCHORS=768 \
  EPOCHS=1 LR=0.0006 \
    bash /absolute/path/to/new/package/run_packed_new_server.sh
)
```

All eight NPUs are training ranks, each with a frozen HF Qwen3-4B target.
No separate teacher server, external project import, dependency update or
four-anchor weights are used. CANN/ATB initialization is retained.

## Exact batching contract

| Item | Packed mode |
|---|---|
| Per-rank input budget | At most 9,216 tokens including right-padding |
| Individual record cap | 3,072 tokens |
| Layout | Independent documents on batch dimension B; no document concatenation |
| Length bucketing | Width 128, shuffled records within buckets and shuffled packs |
| Batch shape | B x L; B x L <= 9216 |
| Maximum records per pack | 128, also bounded by the anchor budget |
| Starts sampled | At most 768 across the whole rank batch, not per document |
| Stage length | Original one anchor embedding + 15 masks = 16 rows |
| Target features | Original Qwen3 hidden states; unused teacher vocabulary logits skipped |
| Loop | Stage zero plus stages 1..k, k uniform in 1,2,3 |
| Loss | Original stage-zero/final-stage weighted CE, jointly normalized per rank |
| Backpropagation | Original differentiable draft-state carriers retained |
| Memory settings | Checkpointed full-vocabulary CE chunks of 128 rows; layer recomputation |
| Save frequency | Every 500 updates and final/max-step completion |

For example, three length-3072 records fit together, or twelve length-768
records fit together. Actual batches use real length buckets; short tails
need not fill the budget. Nothing pads a short record to 9,216 just to claim
that budget was used. Each document starts from its own positions and PoSE
offset; it cannot read another document's prompt or response. Target attention
uses the original HF padding/causal mask. The native draft mask restricts each
block to its own document's target prefix and its own carrier/block states.

The anchor allocator shares quotas fairly across documents, uniformly samples
eligible starts without replacement within each document, and uses each
document's true end for the native 32-token boundary margin. A document is
capped at ceil(768/B) starts, so a response-rich record cannot create an enormous
number of padded blocks for its neighbors. Actual starts can be below 768 when
records have too few eligible positions. Padded anchor slots are bounded by
768+B-1 and are excluded from loss. The progress bar reports actual starts.

Every original dataset index is scheduled once per epoch. The upstream
index-seeded replacement of records with fewer than 64 trainable tokens is
preserved and its count is reported. No 90/10 split or dataset subsampling is
added. To give all distributed ranks the same update count, up to seven packs
are repeated at the distributed tail for eight ranks; this is reported, not
silently dropped. Epochs reshuffle records/packs. Fixed length buckets keep
the update count stable across epochs, allowing exact batch-order resume.

Loop depth uses an independent uniform RNG keyed by global step and seed.
All FSDP ranks therefore execute the same number of stages despite their
different batch shapes. This avoids rank-dependent collective counts.

## What changes in the experimental recipe

The draft architecture, single-anchor stage layout, loop propagation, CE
labels/decay and optimizer are still the pinned upstream implementation.
Batching, anchor allocation across records and the loop RNG stream are explicit
recipe changes. Packed per-rank loss is a joint weighted mean over sampled
blocks, not an average of independently normalized document losses. Floating
reduction order and resulting training trajectories may differ. Scratch
initialization is also a user-requested variation from the published pretrained
initialization. This package is not a claim of an exact paper reproduction.

## Planning and outputs

An initial CPU scan builds a fingerprinted packing index, with a visible
progress bar. Later launches reuse it. `CHECK_ONLY=1` prints the real plan
without loading full models or starting torchrun. It does not create a fresh
OUTPUT_DIR, so a subsequent real launch is not blocked by the plan-only run.

The report includes `steps_per_epoch`, `total_steps`, mean records/tokens per
pack, actual token utilization and distributed tail repetitions. It is based
on the dataset, not a hardcoded desired step count. Do not use the old
177,614-update denominator to estimate this run.

```text
output/dloop_packed_TIMESTAMP_PID/
  packing_plan.json
  launch_provenance.json
  train_args.json
  train.log
  packed_metrics.jsonl
  source_snapshot/
  epoch_0_step_500/
  epoch_0_step_1000/
  epoch_1_step_TOTAL/
  latest_checkpoint.txt
```

Progress shows documents, real input tokens and actual sampled starts for rank
zero. `iter_time` is measured after device synchronization, not before
asynchronous work finishes. Throughput still needs a hardware measurement;
fewer optimizer updates do not imply an identical speedup factor.

## Resume and old checkpoints

Do not resume the old unpacked optimizer/scheduler into this changed batch
schedule. The launch command starts a fresh scratch run in a new output folder.
It leaves old checkpoints and code untouched. Ctrl-C does not automatically
save the current step; only completed checkpoints are resumable.

For a checkpoint produced by this packed version, preserve its original code,
data, rank count and settings, then use:

```bash
RESUME=1 OUTPUT_DIR=/absolute/path/to/original/packed/run \
  bash /absolute/path/to/same/package/run_packed_new_server.sh
```

`MAX_STEPS=20` can optionally limit a hardware check while keeping the full
packed LR schedule. Removing that limit with RESUME=1 continues from the
complete saved state. It is not used to fake a short epoch.

## Validation boundary

CPU tests compare padded-batch DLoop losses/parameter gradients against
separate unpadded documents at k=1,2,3; target states before/after skipping the
unused head; cross-document isolation; sampling/budget bounds; dataset
coverage; and actual trainer save/resume. They do not establish Ascend/HCCL
performance or full-model memory use. See VALIDATION.md for details.
