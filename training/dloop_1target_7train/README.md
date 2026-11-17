# Parallel pretrained four-anchor DFlash and DLoop training

## Latest selection: DLoop, one target card and seven training cards

Use these new launchers for the current request. They run **DLoop only**:

| Setting | Value |
|---|---|
| Target serving | NPU 0, one frozen HF target server |
| DLoop training | NPUs 1,2,3,4,5,6,7, seven FSDP ranks |
| Training starts | 256 per 3,072-token budget unit; up to 768 per rank/update |
| Packed token budget | 3 × 3,072 = 9,216 per rank |
| Extra training | One epoch from the existing regular DFlash initialization |
| Draft attention | Full attention, zero dropout |
| GSM8K | Existing 128 prompts at 25%, 50%, 75%, 100% of the planned document pass |
| DLoop inference gate | -0.5 |

Stop the previous launcher with Ctrl+C in its terminal before starting this
job; its cleanup stops the target processes it owns. This starts a **new
seven-rank run from the original prepared base weights**, with new optimizer
state. It cannot resume the six-rank optimizer shards. Existing checkpoints and
results are retained. Extract the updated ZIP into a separate code directory:

```bash
cd /home/n84449292/a84384644
unzip -oq parallel_dflash_dloop.zip -d dloop_1target_7train_code
cd dloop_1target_7train_code/parallel_dflash_dloop
export DLOOP_RUN_DIR=/home/n84449292/a84384644/parallel_training_runs/dloop_1target_7train_a256_extra1
bash prepare_dloop_8card.sh
bash run_dloop_8card.sh
```

Run in the activated `four_anchor_ali` environment. Preparation reuses
`/home/n84449292/a84384644/parallel_training_runs/dedup_base_pair` as its source.
It does not repeat dataset selection or GSM8K downloads, and leaves that source
experiment unchanged. The new seven-rank document plan and exact evaluation step
numbers are printed during preparation. Do not reuse the old four- or six-rank numbers.
Each short original document stays isolated; 768 is the sampled-start budget
across the packed rank batch, not a fixed quota on every original document.
Across seven ranks, the maximum update budget is 64,512 padded tokens and 5,376
sampled starts. Actual counts depend on packing and eligible supervised positions.

New results are under:

```text
/home/n84449292/a84384644/parallel_training_runs/dloop_1target_7train_a256_extra1/
```

Training pauses for each GSM8K check, evaluates on cards 1–7, and resumes with
all seven optimizer/RNG states and the full-epoch learning-rate schedule.
The target server remains idle during evaluation. The current saved prompts
have no reference answers, so acceptance length and throughput are available;
GSM8K accuracy is unavailable. This does not invent an accuracy score.

`dloop/results.csv` holds the checkpoint comparison. Per-phase `train.log` and
`eval.log` files are under `dloop/phase-*/`; the server log is `target_0.log`.
Rerunning the same launcher resumes the latest complete seven-rank checkpoint,
retries an unfinished evaluation, and skips completed evaluations. All eight
cards must be free at startup. The launcher only stops processes it started.

The target server uses the same HF hidden-only forward as the current DLoop
trainer, including its random per-document RoPE offsets up to the existing
28,672-position budget. The existing vLLM launcher cannot convey those offsets.
Token IDs, padding masks, layer taps and BF16 feature values are preserved.
Features are transferred as temporary safetensors and deleted after consumption.
When at least 4 GiB is free in `/dev/shm`, exchange files use that memory-backed
directory; otherwise they use the run directory. No full-dataset feature cache
is created. `TEACHER_EXCHANGE_DIR` can select a different shared local directory.
The existing exact block-isolated DLoop attention optimization stays enabled.
Target serving/transfer can still limit throughput; no measured NPU speedup is
claimed for this new placement.

Optional settings: `PREPARED_PAIR` selects the existing prepared source;
`DLOOP_RUN_DIR` selects a new output directory (set it in both commands);
`TEACHER_PORT` selects the single target port before preparation (default 8093);
`TRAIN_MASTER_PORT` defaults to 29847. `bash run_dloop_8card.sh --check-only`
checks imports and frozen inputs without allocating NPUs. The regular launcher
also performs live token/mask/offset transfer checks on the target server
before starting the seven trainers.

## Earlier paired mode

Two independent jobs for Ali's 8-card Ascend server. Both warm-start from the
same regular DFlash tensors. Each job saves a complete training checkpoint,
temporarily releases its four NPUs for GSM8K evaluation, and resumes training
with the saved optimizer, scheduler progress, and RNG state. The other job can
continue throughout. There is no simultaneous evaluator competing for the same
NPU memory, and no extra evaluator card is required.

## Start

For the already-prepared `dedup_base_pair` experiment, apply the matched recipe
before starting either training job:

```bash
cd /home/n84449292/a84384644/parallel_dflash_dloop
python align_prepared_run.py
```

This reuses prepared data, initial weights, cached indices and GSM8K prompts.
Both jobs use the same DLoop length-bucket document batches, 512 sampled-start
budget slots per 3,072-token unit (1,536 per 9,216-token rank batch), and **one
additional fine-tuning epoch from the base checkpoint**. For the prepared
1,112,451-row dataset, both plans have 26,847 updates and evaluate at steps
6,694 / 13,392 / 20,140 / 26,847. The updater records a backup and refuses to
modify running or already-started experiments. It does not rebuild the dataset.

For a fresh preparation, follow the steps below; these matching settings are
now the defaults.

Copy and extract this ZIP under `/home/n84449292/a84384644`, then run:

```bash
cd /home/n84449292/a84384644/parallel_dflash_dloop
source /home/n84449292/miniforge3/etc/profile.d/conda.sh
conda activate four_anchor_ali
bash prepare.sh
```

Preparation is CPU-only. It verifies and converts the base checkpoint, writes
one common eligible subset of the deduplicated training data, freezes GSM8K
questions and tokenization, and calculates the actual evaluation steps.
Set `GSM8K_SOURCE` to an existing local benchmark JSONL to prepare fully offline.
It can take several minutes for a large dataset. The original dataset and base
checkpoint remain unchanged. No dependencies are installed or upgraded.

Optional import/config checks after preparation:

```bash
bash run_mine.sh --check-only
bash run_dloop.sh --check-only
```

In **terminal 1**:

```bash
cd /home/n84449292/a84384644/parallel_dflash_dloop
bash run_mine.sh
```

In **terminal 2**, while the first is running:

```bash
cd /home/n84449292/a84384644/parallel_dflash_dloop
bash run_dloop.sh
```

The launchers refuse occupied cards and never kill existing workloads.
Leave each terminal open; training and evaluation progress is printed there.

## Defaults

| Setting | Your four-anchor method | Native DLoop |
|---|---|---|
| Physical NPUs | 0,1,2,3 | 4,5,6,7 |
| Training ranks | 4 | 4 |
| Frozen teacher | Local HF Qwen3-4B on each rank | Local HF Qwen3-4B on each rank |
| Initial weights | Same regular DFlash checkpoint | Same regular DFlash checkpoint |
| Draft attention | Full attention; sliding window disabled | Full attention; sliding window disabled |
| Epochs | 1 | 1 |
| Per-rank token budget | Same document packs, fitting within 9,216 padded tokens; consumed as isolated flat documents | 9,216 padded tokens, independent document rows |
| Maximum original record length | 3,072 | 3,072 |
| Sampled-start budget per rank/update | Up to 1,536 across the whole packed batch | Up to 1,536 across the whole packed batch |
| Budget per 3,072-token unit | 512 | 512 |
| Document batches and update count | Shared length-bucket plan | Same shared plan |
| Model | Five draft layers, 4 conditioning anchors, 15 proposals | Five draft layers, block size 16 |
| Learning rate / warmup | 0.0006 / 4% of full run | 0.0006 / 4% of full run |
| Optimizer | Native AdamW, FP32 trainable weights, weight decay 0 | Native AdamW over FP32 master weights, weight decay 0 |
| Objective | Original 0.1 CE + 0.9 TV, two-pass suffix training | Original rollout CE, main + sampled loop stage 1–3 |
| Reused draft-state gradients | Original detached anchors | Original differentiable carriers |
| Eval inference | Original bonus-add-on 20/19 two-pass layout | Native DLoop, loop gate **-0.5**, stage cap 16 |

The four conditioning anchors in your model are different from the sampled
training starts. The budget is **3 × 512 = 1,536** for the 9,216-token rank batch.
The 512 value specifies the budget per 3,072-token unit. Sampling still operates
across the packed batch rather than enforcing an identical quota on every short
original document; eligibility and sampled starts remain method-specific.
For new runs, `ANCHORS_PER_SEQUENCE` can override this budget before preparation.

Base checkpoint:

```text
/home/n84449292/a84384644/dfalsh_check/checkpoints/0
```

Dataset source:

```text
/home/n84449292/a84384644/datasets/open_perfectblend.qwen3-4b-rollout.qwen3.seq3072.dedup
```

Target default:

```text
/home/n84449292/a84384644/models/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c
```

Both initializations are **weight warm-starts**, with new method-specific
optimizer state. Your extra anchor projection starts as identity, and its RMSNorm
starts at ones. After each GSM8K evaluation, continuation is a **full training-state
resume**, not another warm-start. The scheduler always retains the entire epoch
budget; it is not restarted or compressed into quarter-epoch runs.

The source checkpoint may use sliding attention with a 2,048-token window.
Per the selected training recipe, conversion preserves every pretrained draft
tensor and changes both new draft configurations to
`layer_types=["full_attention"]*5`, `use_sliding_window=false`, and
`sliding_window=null`. Dropout remains zero. This intentionally changes the
attention connectivity for fine-tuning; it does not claim identical forward
outputs to the original sliding-window checkpoint. The original checkpoint
and frozen target configuration are unchanged. Document isolation and the
context-before-anchor mask still apply. `initialization/conversion.json`
records both the source and training attention settings.

## Evaluation timing and measurements

“Every 0.25” is interpreted as **every 25% of the planned dataset pass**:
25%, 50%, 75%, 100%. The CPU planner counts documents across all four ranks and
chooses the first complete update crossing each threshold. It does not assume
that one quarter of optimizer steps equals one quarter of documents. The exact
processed fraction and document count are written into every result.

For a literal **0.25%** interval (400 evaluations per epoch), set this **before
preparation**, with a new output directory:

```bash
export EVAL_FRACTION=0.0025
export RUN_DIR=/home/n84449292/a84384644/parallel_training_runs/quarter_percent_pair
bash prepare.sh
```

Use the same `RUN_DIR` in both training terminals. Frequent full evaluations
and model reloads can dominate runtime. Default preparation refuses to overwrite
an existing run directory containing initialization, data, or training outputs.
An empty directory or a directory containing only `gsm8k.json` can be reused
after a preparation failure. Existing prompts are reused only when they match
the requested source and limit; they are not overwritten.

Both methods evaluate the same 128 GSM8K **test** questions. With the Hub dataset,
selection uses seed 0. With `GSM8K_SOURCE`, the first 128 records of the supplied
file are retained in their existing order, and saved token IDs are preserved exactly.
Evaluation uses
greedy decoding, thinking disabled, and at most 2,048 output tokens per question.
The four cards evaluate distinct questions, one active request per card. Each
checkpoint reports:

- Pooled acceptance length including the bonus/known token, and accepted draft
  tokens per verification round.
- GSM8K numeric exact match from the last `\\boxed{...}` or `####` answer, plus
  missing-answer and truncation counts.
- Aggregate generation throughput, including prefill, excluding model loading
  and parity checks. Graphs are off; this is not a vLLM/CUDA-graph throughput run.
- Greedy target parity checks on the first two questions, up to 128 output tokens.

Acceptance uses the complete final verification round, even when an EOS or
output limit truncates the emitted text. DLoop may verify more than 15 draft
tokens through looping; your method verifies 15. Both metrics use the same
accepted-proposals-plus-one convention. Accuracy is a simple strict numeric
check, not the full lm-evaluation-harness protocol.

GSM8K must be available in the current Hugging Face cache, or supply a local
HF saved dataset / JSONL. Supported records include `input_ids`,
`prompt_token_ids`, `question`, `prompt`, `text`, `turns`, or system/user
`messages`. Saved token IDs are reused without applying the chat template again.
Already-rendered Qwen chat text is encoded without double-wrapping it.
Other text prompts use the local target tokenizer. Local JSONL parsing does not
construct a Hugging Face dataset builder or consult the Hub:

```bash
export GSM8K_SOURCE=/absolute/path/to/gsm8k_test.jsonl
bash prepare.sh
```

If this server can access Hugging Face and has no cached copy, permit the public
test-set download during preparation:

```bash
HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 HF_DATASETS_OFFLINE=0 bash prepare.sh
```

Reference `answer`, `gold_answer` or `reference_answer` fields are optional.
Without parseable references, acceptance and throughput are still measured;
accuracy is `null` in JSON and blank in CSV, with an explicit status and scored
prompt count. Answers are never invented or inferred from the model output.
For partially labeled inputs, accuracy is computed over the scored subset only.

No evaluation set is silently substituted. Change `EVAL_LIMIT` or
`EVAL_MAX_NEW_TOKENS` before preparation if desired. These choices are frozen in
the prepared run.

## DLoop acceleration

The native dense SDPA computes scores for `[all context | all carriers | all
draft blocks]`, including connections later hidden by its mask. The added
attention adapter keeps native Q/K/V projections and RoPE but computes each
block only against `[context | its own carriers | its own draft tokens]`.
The same context-before-anchor mask is applied. It does not truncate context,
detach carrier gradients, reduce anchors, or change loop depth or losses.

For example, with 3,072 context tokens, 256 blocks, 16 rows/block and 44 carriers
per block, the dense key axis has 18,432 positions. The isolated key axis has
3,132 positions per block. This is about 5.9× fewer attention score positions
at that stage; **it is not a claim of 5.9× faster end-to-end training**. The frozen
teacher, projection, feed-forward, and vocabulary losses still cost time.

Block attention is evaluated in groups of 8, with recomputation to bound the
temporary K/V expansion. Full-vocabulary CE chunks increase from 128 rows to
512; activation checkpointing and the hidden-only teacher path remain enabled.
Four ranks share each job; there are no teacher-only cards.

For a baseline or a smaller memory footprint, use a **new** prepared run:

```bash
export RUN_DIR=/home/n84449292/a84384644/parallel_training_runs/dloop_dense_comparison
FAST_DLOOP=0 HEAD_LOSS_CHUNK_SIZE=128 bash prepare.sh
```

`ATTN_BLOCK_CHUNK=4` or `HEAD_LOSS_CHUNK_SIZE=256` can reduce temporary memory.
Set these before preparation. The runtime records the selected values. No
measured NPU speedup or NPU memory-fit guarantee is included in this package.

## Outputs and recovery

Default run root:

```text
/home/n84449292/a84384644/parallel_training_runs/dedup_base_pair/
```

| Relative path | Content |
|---|---|
| `settings.json` | Frozen recipe, actual evaluation step plans, data counts |
| `training_data/selection_report.json` | Original and retained row counts |
| `initialization/conversion.json` | Per-tensor hashes and shared base source |
| `mine/checkpoints/checkpoint-*/` | Your saved weights, optimizer, per-rank RNG |
| `dloop/checkpoints/epoch_*_step_*/` | DLoop weights and all four optimizer/RNG shards |
| `mine/results.csv`, `dloop/results.csv` | Acceptance and GSM8K results across checkpoints |
| `{method}/phase-*/train.log` | Training progress and timing |
| `{method}/phase-*/gsm8k/summary.json` | Metrics for one evaluation |
| `{method}/phase-*/gsm8k/all_records.json` | Generated tokens, answers, acceptance traces |

Running the same launcher again skips completed evaluations, retries an
unfinished evaluation from its checkpoint, and resumes unfinished training.
An evaluation or parity failure stops that job before further updates; inspect
its `eval.log`, correct the external problem, and rerun the same launcher.
Stopping one terminal does not stop the other job. No checkpoints are deleted.

The common subset excludes rows with fewer than 64 loss tokens or insufficient
valid training starts, preventing DLoop's native short-row replacement from
reintroducing duplicates. Both methods now use the same length-bucket sampler,
seed, document order, rank assignment and update count. Both can repeat the same
at-most-three tail packs to equalize ranks. Fractions refer to this shared planned
pass, including those tail rules. The own-method sampler is adapted separately
at runtime, without editing its native objective or model code.

Equal settings do not guarantee equal wall time: your method uses two drafting
rounds; native DLoop uses the main round plus 1–3 sampled loop stages, averaging
three rounds. Loss computation and teacher execution also differ. Measure actual
step timings on the server rather than assuming an exact runtime ratio.

## Provenance and validation

`SOURCE_COMMITS.json` pins the original four-anchor branch, Ascend DLoop port,
official NAVER DLoop, and Speculators dependency. Source and licenses are bundled.
The original DLoop method and optimizer source files are checked by the port's
hash verifier. The block-attention adapter is installed separately at runtime;
setting `FAST_DLOOP=0` selects the native dense path.

`VALIDATION.md` records tests actually executed in the preparation environment.
NPU kernels, four-rank HCCL/FSDP, the user's real checkpoint/data, and real GSM8K
scores must still be validated on the server. The package does not connect to
that server or start jobs remotely.
