# NAVER DLoop training on Ascend

An independent, unofficial Ascend port of the **official NAVER DLoop DFlash
trainer**, pinned to `775dedd71aa95e1d98e6780efe5c444a2a27575c`.
Source: <https://github.com/naver-ai/DLoop>.

**For the corrected 9,216-token packing run, use `run_packed_new_server.sh`.**
It defaults to scratch initialization, a per-rank padded token budget of 9,216,
maximum individual record length 3,072, and up to 768 sampled block starts
**across each entire packed batch**. It indexes the real dataset and prints
the exact packed update count before loading any models. See [PACKING.md](PACKING.md).
The original `run_new_server.sh` remains the explicitly **unpacked** recipe.

The default is `INIT_MODE=pretrained`, matching the published initialization.
`INIT_MODE=scratch` is an explicit experimental option: it randomly initializes
the same drafter and uses the unchanged DLoop objective. This option is not
the paper's checkpoint-based training recipe. It needs no downloaded DFlash
weights; the pretrained target is still required.

The actual `upstream/specforge/core/dflash.py` and
`upstream/specforge/modeling/draft/dflash.py` are unmodified. The optimizer,
scheduler, HF target wrapper and original data preprocessing are also
unmodified. `UPSTREAM_SHA256.json`, `ASCEND_PORT.patch` and
`ascend_support.verify_sources` make this auditable.

No custom four-anchor model, suffix-redrafting objective, confidence-boundary
selector, TV/KL distillation, configuration importer, cached head loss, or
bonus20 adapter is used. No `speculators` dependency is needed. The optional
packed mode adds independent-document token-budget batching and a batch-wide
anchor allocator; it keeps the original DLoop forward/stages and CE objective.

## Training method

1. Initialize the regular DFlash drafter from the released
   `z-lab/Qwen3-4B-DFlash-b16` checkpoint.
2. Sample up to 256 block starts per sequence using NAVER's sampler. For
   DFlash, every stage is **one token embedding + 15 masks = 16 rows**.
3. Stage zero conditions on frozen target features before its anchor.
4. Draw final supervised loop depth `k` uniformly from `{1,2,3}`.
5. Compute stages 1 through k. Each extends to a later block; the input anchor
   ID comes from the stored rollout during training. Previous draft prediction
   states enter K/V with gradients. At the first loop stage the carrier has
   offsets 1..14, while its anchor is at offset 15. Later carriers grow.
6. Supervise stage zero and final stage k using CE against rollout token IDs,
   exponential position weights `exp(-j/7)`, and joint weight normalization.
   Intermediate stages receive gradients through the carrier.

The original target context stays fixed throughout a training loop. There is
no target call between loop stages. Confidence gating belongs to inference;
it does not select the training loop stage.

## Original unpacked recipe and disclosed platform differences

| Setting | This launcher |
|---|---|
| Initialization | Released regular DFlash weights by default; explicit scratch option below |
| Target | Your local Qwen3-4B, frozen HF target per training rank |
| Epochs | 1 |
| Batch size | 1 sequence per rank |
| Sequence length cap | 4096 |
| Sampled block starts | 256 per sequence |
| Learning rate | 0.0006 |
| Warmup | 4% of total updates |
| Optimizer | Upstream FP32-master AdamW, weight decay 0 |
| Gradient clipping | 1.0 |
| Seed | 42 |
| Loop depth | k uniformly from 1,2,3 |
| Position offset maximum | 28672, applied to target and drafter |
| Loss | Upstream DFlash CE, gamma 7 |
| Distributed execution | Upstream FSDP SHARD_GRAD_OP; HCCL instead of NCCL |
| Attention | SDPA on target and draft; the published draft recipe uses FlexAttention |
| Model/log checkpoints | Every 5000 updates and at final/max-step completion |

This is the official **method** and default recipe with explicit platform and
input-format adaptations, not a claim of exact paper-result reproduction.
The published recipe uses on-the-fly raw conversation tokenization. By default
this package reads your existing save_to_disk `input_ids`/`loss_mask` directly.
It inherits upstream's deterministic replacement of rows with fewer than 64
loss tokens after truncation. It does not apply the former 90/10 split or
9,216-token packing policy. A seq3072 corpus remains at most 3072 tokens per
record even with a 4096 cap. To use the original raw input path, set DATASET to
a compatible conversation JSONL; that runs upstream's on-the-fly loader.

These defaults intentionally do not import your previous five-epoch config.
Epochs, corpus preprocessing, initialization, rank count and compute should
be reported when comparing experiments. Matching this recipe does not match
your previous run's compute budget.

## Launch on new server 104

Unzip this package to `/home/n84449292/a84384644/dloop-official-ascend`.
Then, when your selected cards are free:

```bash
(
  set -eo pipefail
  source /home/n84449292/miniforge3/etc/profile.d/conda.sh
  conda activate four_anchor_ali
  NPUS=0,1,2,3,4,5,6,7 \
    bash /home/n84449292/a84384644/dloop-official-ascend/run_new_server.sh
)
```

The launcher looks for the released DFlash checkpoint in named local model
directories and HF caches. If it is stored elsewhere, specify:

```bash
DRAFT_INIT=/absolute/path/to/Qwen3-4B-DFlash-b16 \
  bash /home/n84449292/a84384644/dloop-official-ascend/run_new_server.sh
```

In pretrained mode, it will never substitute a four-anchor checkpoint or
randomly initialize a missing drafter. If the released checkpoint is absent, download it on a
connected machine and transfer it, or run this in the existing environment
on the server if that server has Hugging Face access:

```bash
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    'z-lab/Qwen3-4B-DFlash-b16',
    local_dir='/home/n84449292/a84384644/models/Qwen3-4B-DFlash-b16',
    allow_patterns=['*.json', '*.safetensors'],
)
PY
```

Model weights are not bundled. The launcher itself is offline. It does not
install packages or change an existing repository/environment. It refuses
occupied/unclear selected NPUs and never stops existing processes.

All eight selected cards are **training ranks with their own frozen HF
target**; there is no separate vLLM server on cards 0 and 1. The progress bar
and losses remain visible in the terminal and are also recorded in train.log.

## Explicit training from scratch (original unpacked entry point)

To train DLoop with a randomly initialized drafter, set `INIT_MODE=scratch`
and unset `DRAFT_INIT`. The bundled architecture-only JSON has five regular
bidirectional DFlash layers, block size 16, target taps `[1,9,17,25,33]`, and
mask token 151669. Its dimensions match the released Qwen3-4B DFlash config:
<https://huggingface.co/z-lab/Qwen3-4B-DFlash-b16/blob/main/config.json>.
No pretrained drafter tensors are included, downloaded, or loaded in this mode.
The original `DFlashDraftModel` constructor initializes its trainable matrices
from the training seed. Normalization parameters retain the constructor defaults.

The target Qwen3-4B, target token embeddings, and vocabulary output head remain
pretrained and frozen, as in the native DFlash training wrapper. "Scratch"
refers to the trainable drafter, not to the target language model. DLoop's
loop depth, carrier gradients, target-context mask, CE labels/weights and
optimizer are unchanged. Both initialization modes use the same source files.

```bash
(
  set -eo pipefail
  source /home/n84449292/miniforge3/etc/profile.d/conda.sh
  conda activate four_anchor_ali
  unset DRAFT_INIT OUTPUT_DIR RESUME MAX_STEPS CHECK_ONLY
  INIT_MODE=scratch NPUS=0,1,2,3,4,5,6,7 \
  EPOCHS=1 MAX_LENGTH=9216 NUM_ANCHORS=768 LR=0.0006 \
    bash /home/n84449292/a84384644/dloop-official-ascend/run_new_server.sh
)
```

This older command uses a 9216-token **record cap**, not a token-budget batch.
It does not implement the requested packing. Existing seq3072 records remain
at most 3072 tokens; this entry point does not concatenate or pack them.
Use `run_packed_new_server.sh` for the corrected packing implementation.
These settings and scratch initialization differ
from the published recipe and must be reported in a comparison. One epoch is
retained explicitly here; scratch convergence has not been established.

`launch_provenance.json` records `initialization_mode`, the architecture hash,
and an empty `warm_start_files` list for scratch. Resume requires the same
initialization mode, source paths, world size and recipe. Resuming scratch
correctly restores your own saved training weights and optimizer; it does not
reinitialize the resumed run or load the released DFlash model.

Dependencies: an existing compatible Torch/torch_npu/CANN installation plus
Transformers, datasets, accelerate, safetensors, huggingface_hub, tqdm,
NumPy and TensorBoard. Upstream specifies Transformers >=5.8.1. Tiny DFlash
CPU checks also pass on 5.0.0, but other versions/Ascend combinations are not
certified here. Preflight imports the actual training dependencies before
allocating full models. No package upgrade is performed automatically.

## Outputs and resume

Each fresh run gets its own directory:

```text
dloop-official-ascend/output/dloop_native_TIMESTAMP_PID/
  train.log
  launch_provenance.json
  train_args.json
  train_command.txt
  source_snapshot/
  runs/                         # TensorBoard
  epoch_0_step_5000/
  epoch_0_step_10000/
  epoch_1_step_TOTAL/            # completed one-epoch run
  latest_checkpoint.txt
```

Each complete checkpoint contains the regular DFlash model/config, one
`training_state_rankR.pt` per rank, and a `COMPLETE.json` marker. The rank files
preserve optimizer/master parameters and RNG. The original upstream trainer
saved only rank 0's optimizer shards; the port corrects that checkpoint I/O.
Resume refuses incomplete checkpoints, changed rank counts, or changed recipes.

```bash
RESUME=1 OUTPUT_DIR=/absolute/path/to/original/dloop_native_run \
  bash /home/n84449292/a84384644/dloop-official-ascend/run_new_server.sh
```

For a scratch run, also set `INIT_MODE=scratch`, unset `DRAFT_INIT`, and restore
the original `EPOCHS`, `MAX_LENGTH`, `NUM_ANCHORS` and other launch settings.

For a two-update hardware check, set `MAX_STEPS=2`. This limits the run without
changing the full-data LR schedule; the resulting checkpoint can be resumed
by setting RESUME=1, using its original OUTPUT_DIR and removing MAX_STEPS.
Ctrl-C does not promise an additional checkpoint; resume from a complete one.

Optional `ACTIVATION_CHECKPOINTING=1` enables Transformer layer recomputation;
default is off, matching upstream. SDPA memory/kernel behavior on Ascend is
hardware-dependent; no full-size speed or memory guarantee is made. Changing
NUM_ANCHORS, EPOCHS, MAX_LENGTH or LR is recorded and is a recipe variation.

## Memory reduction for 768 sampled starts

The native loss forms full-vocabulary logits for every prediction row at
once. With 768 blocks, 15 prediction rows and vocabulary size 151936, one
BF16 logits tensor is 3.260 GiB. Both the main stage and the selected loop
stage contribute losses before backward. Cross-entropy intermediates, draft
activations across up to four forwards, and the rank-local frozen target
require additional memory. FSDP does not shard these per-rank activations.

Set `HEAD_LOSS_CHUNK_SIZE=128 ACTIVATION_CHECKPOINTING=1` to use the optional
memory adapter. This preserves the configured sampled starts, native target context,
loop depth, differentiable carriers, CE labels, exponential weights and
joint stage normalization. The vocabulary remains complete: the chunk size
is prediction rows, not vocabulary entries. A BF16 logits chunk is about
37.1 MiB at 128 rows. Checkpointing recomputes the head/loss in backward rather
than retaining all chunks' logits; simply splitting the forward would not
solve that retained-memory problem.

The adapter is `ascend_support/chunked_ce.py`, enabled on the native wrapper
before FSDP. It adds no trainable parameters. The original core/model source
files remain unchanged. `HEAD_LOSS_CHUNK_SIZE=0` restores the original dense
loss. The same mathematical loss/gradients are preserved, but floating-point
reduction ordering may differ; bitwise full-run equivalence is not claimed.
Recomputation costs extra compute. No measured NPU memory or speed guarantee
is available; attention masks/backbone allocations may still be limiting.

```bash
INIT_MODE=scratch HEAD_LOSS_CHUNK_SIZE=128 ACTIVATION_CHECKPOINTING=1 \
MAX_LENGTH=9216 NUM_ANCHORS=768 EPOCHS=1 \
  bash /path/to/dloop-official-ascend/run_new_server.sh
```

Use a fresh output directory after an OOM before any complete checkpoint.
Do not claim this resumes the interrupted first update. A seq3072 dataset
is still not packed through `run_new_server.sh`: with approximately 1.42 million rows and eight ranks,
one epoch has approximately 177614 optimizer updates regardless of the
larger length cap. Dense-vs-chunked tests cover full loop gradients at depths
1/2/3, BF16/FP32 loss, masks and out-of-range labels, zero supervision, layer
recomputation, and no retained vocabulary-sized loss tensors.

## Scope and verification

See `VALIDATION.md` for the executed tests and hardware limitations. This
package trains DLoop and exports regular DFlash checkpoints. It does not make
the previous four-anchor vLLM inference adapter a DLoop decoder. Upstream's
evaluation code is included as reference; its GPU paths still need separate
Ascend validation before reporting DLoop inference results.
