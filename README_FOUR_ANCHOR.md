# Four-anchor DFlash pipeline

This implements **one shared regular DFlash drafter**, four explicit anchors, and two draft passes. Each pass has **4 anchor slots + 16 masked generation slots = 20 query positions** and predicts **16 new tokens**. It includes data preparation, training and resume, greedy inference, evaluation, HTTP serving, CPU tests, and launch scripts.

The decoder layers are imported directly from `speculators.models.dflash.model_definitions.Qwen3DFlashDecoderLayer` in the repository's pinned submodule. No prefix-attention variant or MTP layer is instantiated. All changes live in `four_anchor/`; existing submodule code and pins are unchanged.

Source base: `alipourranjbar1/attention_prefix_head`, branch `prefix-attention-qwen3-4b`, root commit `831471c2207fcf41493c73d126ca023ced6cc301`, speculators commit `89b5caa300f473c1e3127497e6f594ba024057d5`.

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

For each example, training runs the actual two-pass structure. The first block is trained against the dataset continuation. A boundary is selected by confidence, with geometric boundary exploration during training so an initially weak model does not always choose zero. The retained tokens and states are first-pass model outputs. A frozen target then rolls out a fresh continuation conditioned on that exact retained prefix, supplying the second-view labels. Those extra teacher states are **never passed to the drafter**. Both views use position-decayed CE in one normalized loss; no loss is applied to anchors or padding. First-pass states passed into the second view are detached.

This teacher rollout is intentionally expensive: up to 16 incremental target calls per training example, plus processing the retained prefix and initial prefill. It avoids silently supervising the redraft with a continuation conditioned on a different prefix. `boundary_exploration=0` uses only the inference boundary rule; the default `0.5` also samples from the paper-inspired geometric distribution with `beta=0.7`.

One example is processed per device per microstep. Gradient accumulation controls effective batch size. Distributed training uses DDP and gradient synchronization only on the final accumulation microstep. Trainable parameters/optimizer states stay float32; the bfloat16 setting uses autocast. Target weights remain frozen. Training is step-based with random example/start sampling, not a claim to reproduce the paper's five-epoch schedule.

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
