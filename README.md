# Four-anchor DFlash pipeline

Branch: **`four-anchor-dflash-pipeline`**. This root README is the branch landing page on GitHub.

This implements **one shared regular DFlash drafter**, four explicit anchors, and two draft passes. Each pass has **4 anchor slots + 16 masked generation slots = 20 query positions** and predicts **16 new tokens**. It includes data preparation, training and resume, greedy inference, evaluation, HTTP serving, CPU tests, and launch scripts.

The decoder layers are imported directly from `speculators.models.dflash.model_definitions.Qwen3DFlashDecoderLayer` in the repository's pinned submodule. No prefix-attention variant or MTP layer is instantiated. Implementation code lives in `four_anchor/`; existing submodule code and pins are unchanged.

Source base: `alipourranjbar1/attention_prefix_head`, branch `prefix-attention-qwen3-4b`, root commit `831471c2207fcf41493c73d126ca023ced6cc301`, speculators commit `89b5caa300f473c1e3127497e6f594ba024057d5`.

## Train on your Ascend setup

Activate the existing `attention_prefix_ali` environment, then run from this branch's checkout:

```bash
CHECK_ONLY=0 bash four_anchor/scripts/train_ascend.sh
```

This uses the paths found in the supplied repository's existing training scripts:

```bash
MODEL=/home/n84449292/m84379596/Huggingface/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c
DATASET=/home/n84449292/m84379596/Huggingface/datasets/open_perfectblend.qwen3-4b-rollout.qwen3.seq3072
```

Existing exported `MODEL` and `DATASET` values take precedence. The launcher validates both local paths on your server. It sources CANN/ATB in the ABI-aware order from your script and uses the existing environment; it does not install or downgrade packages. `VLLM_SOURCE_ROOT` defaults to your original `attention_prefix_head` checkout, which supplies the working vLLM/vLLM-Ascend code. Four-anchor trainer and regular DFlash source come from this branch.

| Setting | Ascend launcher |
|---|---|
| Target NPUs | `0,1`, vLLM data parallel size 2 |
| Trainer NPUs | `2,3,4,5,6,7`, six DDP ranks |
| Target model on trainer cards | Frozen embeddings, final norm and LM head; no full target transformer |
| Draft | Five regular bidirectional DFlash layers |
| Conditioning anchors | 4 |
| New tokens per pass | 16; 20 query positions total |
| Packed token budget | 3072 per trainer rank |
| Training / held-out split | 90% / 10%; `TRAIN_DATA_RATIO` overrides |
| Training blocks | Up to 512 per packed batch (`MAX_ANCHORS`) |
| Loss on both passes | **0.1 CE + 0.9 TV**, full target vocabulary |
| Optimizer | AdamW, unchanged; DSpark loss does not select Muon |
| Epochs | 5 shuffled passes over the training split |
| Learning rate / schedule | 0.0006 / cosine; 50 warmup optimizer steps by default |
| Precision / seed | BF16 autocast with FP32 trainable/master weights / 42 |
| Gradient accumulation | 1 packed batch per rank per update; `GRADIENT_ACCUMULATION` overrides |
| Checkpoints / logging | Every epoch plus final/early stop; TensorBoard and `train.log` |
| Progress bar | Rank 0 shows steps, percentage, elapsed time, ETA, loss and epoch |
| Target server port | 8080 |

The model remains five bidirectional regular DFlash layers. **Four** is the number of conditioning tokens inside each block; **512** is the maximum number of sampled prediction starts in one packed batch. Each start creates 4 anchor slots plus 16 generation slots. Anchor slots have no loss.

Training now uses the repository's `MultipackDistributedBatchSamplerV2`: it packs whole documents up to 3072 tokens per rank, resets their positions and blocks cross-document attention. Like that sampler, it can drop fewer than `world_size` trailing rows to keep ranks synchronized. Rows are shuffled each epoch. The 90% training split is deterministic; held-out rows are excluded from optimization. Set `TRAIN_DATA_RATIO=1` only if the comparison baseline also trains on the full dataset.

The launcher computes **actual steps for each epoch and all epochs** before starting the target server, and writes `checkpoints/packing_plan.json`. It does not hard-code 80,000 or shorten the dataset to force a count. The previous 1,183,640 bar counted all five epochs of an unpacked, one-row trainer. A packed count depends on token lengths, train split, worker count and accumulation. A 70–80k *per-epoch* count would mean 350–400k updates over five epochs. The 10–12 hour runtime must be measured on the NPUs.

The bar now reads `Epoch 1/5: ... current/steps_this_epoch`, with elapsed time and epoch ETA. Its postfix shows the global step and an estimated remaining time for the full run. JSON metrics and TensorBoard separately log first/second CE and TV (KL is diagnostic and has zero weight in the default loss). `SHOW_PROGRESS=0` hides only the bar. `MAX_STEPS` stops a trial without changing the five-epoch schedule.

Before packing, an Arrow scan filters rows without a supervised token after a nonempty prefix within 3072 tokens. This fixes the observed unsupervised-row crash without skipping individual DDP updates. Row indices, lengths and packing plans are cached under `output/.four_anchor_data_cache`; source data remains unchanged. Start a **fresh run** when switching from the old online objective; old optimizer checkpoints are not compatible with the new schedule/objective. Corrected packed runs support exact resume with per-rank RNG and FP32 master weights preserved.

The launcher requests vLLM capture slots `[2,10,18,26,34]` plus the final decoder output using `--include-last-layer`. The five drafter taps remain zero-based decoder indices `[1,9,17,25,33]`; vLLM numbers embedding as zero. The extra final state supplies the loss targets only: the frozen target final norm and LM head reconstruct its next-token distribution. The hidden state at token `t-1` supervises prediction of token `t`.

Each document needs **one teacher prefill**, reused across its sampled blocks and both views. CPU workers prefetch those requests; no per-block autoregressive teacher rollout is performed. A live contract check verifies token alignment and both feature shapes before allocating the six trainers. Full-vocabulary losses are computed in checkpointed chunks (`HEAD_CHUNK_SIZE=64`); draft blocks are processed in groups (`BLOCKS_PER_CHUNK=32`) to bound attention memory.

```bash
# Optional imports, configuration and actual packing counts; no server/training:
CHECK_ONLY=1 bash four_anchor/scripts/train_ascend.sh

# Optional short run, then resume using its checkpoint and the same recipe:
MAX_STEPS=10 CHECK_ONLY=0 bash four_anchor/scripts/train_ascend.sh
RESUME=/path/to/run/checkpoints/checkpoint-000010 CHECK_ONLY=0 \
  bash four_anchor/scripts/train_ascend.sh

# Optional regular full-vocabulary DFlash warm start:
DFLASH_INIT=/path/to/regular-dflash CHECK_ONLY=0 bash four_anchor/scripts/train_ascend.sh
```

The run directory is `output/four_anchor_dflash_<timestamp>_<pid>/`. Logs are `train.log`, `target_server.log`, and `checkpoints/logs/four_anchor_dflash/`; the resolved configuration is `train_config.json`. The launcher owns and shuts down only its target-server process group. The packed implementation is tested on CPU, including BF16 resume and exact greedy decoding after training. Full six-rank Ascend training and wall-clock throughput remain to be validated on the server.

## Exact algorithm

Let the fully processed, verified prefix occupy positions `0 … n-1`.

1. Pass one receives tokens at positions `n-4 … n-1`, their target features, and 16 masks at positions `n … n+15`.
2. It produces proposals `d[0:16]` and the final drafter state at each proposal position.
3. Select `l` with D-Loop's confidence boundary score:

   `l = argmax_{j=0..15} [ product_{k<j} c[k] * (1-c[j]) ]`.

   `c[k]` is the untempered softmax probability of the greedy first-pass token. The implementation computes the score in log space. `l` means keep exactly `l` proposals. The token at index `l` is the **predicted** first error, not an observed rejection. No target call occurs between draft passes.
4. Pass two receives exactly the four tokens at positions `n+l-4 … n+l-1`, their original positions and available features, then 16 masks at `n+l … n+l+15`. It reuses the **same backbone weights**.
5. Verify the concatenated candidate `d[:l] + second_pass[0:16]` with the target. Accept only a contiguous matching prefix, then append the target correction/bonus token. Crop rejected target KV entries before processing that token.

Example: if `n=100` and `l=6`, pass one anchors are `96,97,98,99`. Pass two anchors are `102,103,104,105`, using proposed tokens `d[2:6]`. Pass two generates positions `106 … 121`; final verification sees 22 candidates at `100 … 121`. It does not use `d[6]` as an anchor.

**Length convention:** `num_proposals=16` means 16 *new* slots per pass; it does not count anchors. The final candidate has `l+16` tokens (16–31), clipped to the remaining output budget. This extension is deliberate: regenerating only `16-l` tokens would not fulfill the fixed 20-input/16-output request. Consequently, increased raw acceptance can partly come from the longer candidate horizon; compare runtime as well as acceptance.

The 20 count refers to query/token positions. Regular DFlash also attends to projected target features from the verified prefix as KV memory; total key length is `prefix_length + 20`.

## Where the anchor states come from

| Anchor position | Token | State supplied to the shared drafter |
|---|---|---|
| In verified prefix | Verified token | Concatenated actual target decoder-layer outputs, projected by DFlash `fc` |
| In retained first-pass proposals | Proposed token | Detached first-pass drafter state through a learned `H → H` projection and RMS norm |
| Before sequence start | Padding | Masked out of attention; never supervised |

Only four explicit token anchors are supplied. Earlier retained proposals beyond those four are not re-embedded in pass two. The target feature memory remains the original verified context in both passes. Draft states are not represented as if they were target states. Target layer IDs are **zero-based decoder-layer indices**, captured using forward hooks before the target's final norm.

The projected anchor feature is added to its token embedding. The original regular DFlash attention and MLP layers remain bidirectional. The learned state bridge is the only added projection; there is no second transformer drafter.

All four initial anchors have materialized target features. Before another cycle, this reference implementation therefore processes the correction/bonus token with an additional one-token target forward after verification. Metrics count that call; it is skipped when generation has finished. Removing this cost while preserving the feature contract requires a separate serving optimization; reusing a rejected token's state would be incorrect.

## Relation to D-Loop

[D-Loop, arXiv:2610.06011](https://arxiv.org/abs/2610.06011), section 3.1 and Algorithm 1, supplies the boundary selection and shared-weight two-pass idea. Here `K=1`, matching the request for one boundary. This is **an extension**, not an exact paper reproduction: four moving anchors, first-pass state carry, a fresh 16-token second block, and the training objective below differ from D-Loop's fixed-horizon complementary prefix–suffix setup. No measured acceptance or speed improvement is claimed.

## Install

Use Python 3.10+ in an isolated environment. Install the PyTorch build appropriate to the machine **first**. The CPU validation environment uses PyTorch 2.10.0 and Transformers 5.0.0.

```bash
# Example CPU environment; use your CUDA or Ascend torch build on an accelerator.
python -m venv .venv-four-anchor
source .venv-four-anchor/bin/activate
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
bash four_anchor/scripts/install.sh
python -m pytest -q four_anchor/tests
OMP_NUM_THREADS=1 python -m four_anchor.smoke --output /tmp/four-anchor-smoke
```

The installer reuses the installed torch stack and installs the pinned local speculators and hs-connectors packages without their broad optional audio/vision dependency set. It does not install vLLM. On Ascend, use your compatible CANN / `torch_npu` environment and `--device npu:0`. Accelerator paths are provided but have not been validated on actual CUDA/NPU hardware here. CPU float32 and bfloat16 training were exercised. A two-worker distributed run was blocked during Gloo initialization by this environment's networking permissions; distributed training is unverified.

The smoke run creates a tiny random Qwen3 target locally, generates teacher data, trains two steps, resumes from step one, checks bitwise-equal final weights, decodes, evaluates on held-out prompts, and tests an HTTP completion request. It downloads no model weights. Its throughput and acceptance numbers are not research results.

## Data preparation

Raw input is JSONL, or an HF dataset saved with `save_to_disk`. Each record may contain `prompt`, `text`, `messages`, or `input_ids`. Plain prompts are used as supplied; chat records use the target's chat template with thinking disabled. Split training and evaluation prompts **before** teacher generation.

```json
{"prompt":"Explain why the sky appears blue."}
{"messages":[{"role":"user","content":"Write a Python binary search."}]}
```

```bash
dflash4a prepare --target /models/Qwen3-4B \
  --data /data/train_prompts.jsonl \
  --output four_anchor/runs/data/train.jsonl \
  --device cuda:0 --dtype bfloat16 --max-new-tokens 256
```

This regenerates continuations from the frozen target and writes `input_ids`, `loss_mask`, and `seq_len`. The prompt has loss mask zero. Existing tokenized JSONL or HF `save_to_disk` datasets with aligned `input_ids` / `loss_mask` can go directly to training. If `loss_mask` is absent, every token after the first is eligible. Use target-generated completions for draft/target alignment; arbitrary corpus labels are accepted but do not provide that guarantee. `prepare` refuses to overwrite an existing output.

## Train and resume

Edit `four_anchor/configs/qwen3_4b.json` or override paths on the command line. The example config uses five draft layers, four anchors, 16 proposals, full vocabulary, target layers `[1,9,17,25,33]`, and mask token `151669`. Validate those settings against the target/checkpoint you choose. Pin a model revision in the JSON config for reproducible remote loading.

```bash
bash four_anchor/scripts/train.sh \
  --target /models/Qwen3-4B --data /data/teacher_tokens \
  --output four_anchor/runs/experiment --device cuda:0

# Start from an existing REGULAR, full-vocabulary speculators DFlash checkpoint:
bash four_anchor/scripts/train.sh \
  --target /models/Qwen3-4B --data /data/teacher_tokens \
  --output four_anchor/runs/finetune --device npu:0 \
  --initialize-dflash /checkpoints/regular-dflash

# Resume optimizer and per-rank RNG state with the same config and world size:
bash four_anchor/scripts/train.sh \
  --target /models/Qwen3-4B --data /data/teacher_tokens \
  --output four_anchor/runs/experiment --device cuda:0 \
  --resume four_anchor/runs/experiment/checkpoint-000100

# Data parallel training; one frozen target replica per worker:
NPROC_PER_NODE=4 bash four_anchor/scripts/train.sh \
  --target /models/Qwen3-4B --data /data/teacher_tokens \
  --output four_anchor/runs/distributed --device npu
```

Regular DFlash warm starts accept local `config.json` plus safetensors (single or sharded). The importer checks model type, selected target layers, full vocabulary, and weight shapes. It rejects prefix-attention / MTP / other variant checkpoints. The new anchor interface and state projection require finetuning even when backbone weights are imported. Native four-anchor checkpoints use `--resume`, not `--initialize-dflash`.

Both the supplied JSON config and Ascend launcher select `training_mode="packed"`. Each sampled start trains the first block against frozen target next-token distributions. The second view uses the same drafter and four preceding tokens/states. Its initial boundary uses the D-Loop confidence score, with optional geometric exploration; **during training only**, the retained length is capped at the first mismatch with dataset tokens and the document end. This makes the reused teacher-forced labels consistent with the retained prefix. Inference uses confidence alone. This is a documented training approximation, not an exact D-Loop reproduction.

Both views minimize the requested **DSpark loss**, `0.1 * CE + 0.9 * TV`. CE uses the teacher distribution's argmax, exactly as the upstream loss does; TV is `1 - sum(min(p_target, p_draft))` over the complete vocabulary. Dataset loss masks and `exp(-position/7)` weight generation slots; the weighted sums of both views are normalized together. `LOSS_FN` can select `ce`, `tv`, or `kl_div` weights through the Ascend launcher. DSpark's RNN and confidence head are not included, so there is no confidence auxiliary loss. The optimizer remains AdamW at 0.0006.

Only the original verified prefix supplies target feature memory in both views. Retained proposal anchors use detached first-pass draft states, never teacher states of the retained proposals. If a group's retained lengths are all zero, the two views have exactly the same inputs and targets; the trainer reuses the first computation with both loss weights instead of repeating it. Nonzero retained prefixes run the second drafter pass.

DDP normalizes gradients by actual global supervised weight across all ranks and accumulation microbatches. Trainable weights and optimizer states remain FP32; BF16 uses autocast. Local-target training uses `num_workers=0`; the Ascend remote-teacher launcher uses four CPU workers per trainer rank. The previous `training_mode="online"` implementation remains available for controlled legacy experiments, but neither supplied default selects it.

Checkpoints include model/config, optimizer, per-rank RNG state, source versions/hashes, command settings, dataset fingerprint, diffs, and model SHA256. Resume uses the same dataset, world size, model configuration, and total learning-rate schedule. Keep those settings fixed for reproducibility.

## Inference and evaluation

```bash
dflash4a generate --checkpoint four_anchor/runs/experiment/checkpoint-001000 \
  --target /models/Qwen3-4B --device npu:0 --dtype bfloat16 \
  --prompt 'Explain speculative decoding.' --max-new-tokens 128

dflash4a evaluate --checkpoint four_anchor/runs/experiment/checkpoint-001000 \
  --target /models/Qwen3-4B --device npu:0 --dtype bfloat16 \
  --data /data/heldout_prompts.jsonl --output four_anchor/runs/eval \
  --limit 100 --max-new-tokens 256 --warmup 2
```

The evaluator runs ordinary target AR, one-pass four-anchor drafting, and two-pass four-anchor drafting on identical prompts and budgets. The one-pass mode is an ablation of this checkpoint, not an unmodified DFlash benchmark. It checks exact output-token equality and fails the run if equality does not hold. Hardware kernels can affect floating-point argmax ties; inspect a failure before interpreting speed measurements.

Results include per-prompt output IDs, boundary/anchor-source traces, accepted draft tokens per cycle, committed tokens per cycle (including correction/bonus), target calls including cache repair, elapsed time, and tokens/second. Summary throughput is total tokens divided by total time; warmup is excluded. Compare acceptance together with throughput: two draft passes, a longer verification block, and correction-state materialization all cost time.

Only greedy decoding (`temperature=0`), one sequence per call, one selected boundary (`K=1`), full vocabulary, and Qwen3 targets are implemented. Target KV caches are reused and rolled back correctly; projected draft context KV is recomputed each pass. No paged attention, continuous batching, graph capture, tensor parallelism, or custom vLLM proposer is included. These are deliberate limits of this executable reference backend. Do not load these new checkpoints into the existing vanilla/prefix vLLM launch scripts.

## HTTP serving

```bash
dflash4a serve --checkpoint four_anchor/runs/experiment/checkpoint-001000 \
  --target /models/Qwen3-4B --device npu:0 --dtype bfloat16 \
  --host 127.0.0.1 --port 8000 --provenance-dir four_anchor/runs/server

curl http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"four-anchor-dflash","prompt":"Explain speculative decoding.","temperature":0,"max_tokens":128}'
```

Endpoints: `/health`, `/v1/models`, `/v1/completions`, `/v1/chat/completions`. This is a serialized, non-streaming local research server. Unsupported sampling parameters are rejected instead of silently ignored. It is not a production vLLM replacement.

## Run the complete pipeline

```bash
TARGET=/models/Qwen3-4B \
TRAIN_PROMPTS=/data/train_prompts.jsonl \
EVAL_PROMPTS=/data/heldout_prompts.jsonl \
DEVICE=npu:0 NPROC_PER_NODE=4 \
DFLASH_INIT=/checkpoints/regular-dflash \
bash four_anchor/scripts/pipeline.sh
```

This prepares teacher data, trains, reads the final checkpoint path, evaluates, and generates a sample. Set `PREPARED_DATA=/data/existing_teacher_tokens` to skip preparation. Serving is a separate command because it is a long-running process. The installer, smoke run, pipeline commands, and full tests are included so the implementation can be reproduced and extended on the actual training hardware.
