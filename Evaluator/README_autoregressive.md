# Qwen3-4B: same DFlash backbone, autoregressive drafting

Based on `acceptance-single-repo` commit `0d4d4d1`. This adds a distinct
`dflash_ar` model; existing DFlash and prefix-selector files are unchanged.
The comparison is against **vanilla DFlash**, not the extra prefix/linear
confidence heads. The AR model has exactly the vanilla backbone's parameter
names and shapes: same embedding, `fc`, `hidden_norm`, Qwen3 decoder layers,
normalization and target-owned vocabulary head. Target-layer IDs and decoder
shape are inherited from your original training recipe.

## What changes

* Teacher forcing: input tokens x[a:a+16] predict x[a+1:a+17].
* Causal attention inside each draft block; no mask-token inputs.
* Same target-feature visibility as DFlash: same document, strictly before
  the anchor. Future target states are inaccessible, even during training.
* Sixteen sequential draft proposals per round, with a fresh incremental draft
  KV cache. Target-prefix K/V enters this cache once, not once per token.
* Target verifies the full proposal; accept only the matching prefix. Use its
  correction at the first mismatch or its bonus token when all proposals match.
* `sample_from_anchor=True` means all 16 input positions predict a next token.
  This is not a diffusion sampling switch; it accompanies the new model code.

The original DFlash block16 uses 15 proposals plus the anchor. AR block16 uses
16 proposals; maximum acceptance length including bonus is 17. For a strictly
matched proposal-count experiment, use 15 AR proposals or train DFlash block17.
Do not interpret this proposed 16-versus-15 run as a controlled count-matched
comparison. No measured acceptance improvement is claimed before evaluation.

## Install patch

From the repository root, with the committed baseline and a clean working tree:

```bash
git switch -c dflash-autoregressive
git apply --check /path/to/dflash_autoregressive.patch
git apply /path/to/dflash_autoregressive.patch
export PYTHONPATH="$PWD/speculators/src:$PWD/speculators/hs_connectors/src:$PWD/vllm:$PWD/vllm-ascend:${PYTHONPATH:-}"
python -m unittest discover -s speculators/tests/autoregressive -v
```

The patch is additive except for model registration and training-default routing.
No existing Python filename is renamed. Do not load this model with `method=dflash`
in the existing vLLM server: that proposer still implements parallel diffusion.
Use the included reference evaluator for this first acceptance experiment.

## Train

Use your current activated Ascend environment and the same sourced CANN settings
as the working baseline. Supply the actual vanilla DFlash `run.yaml`; the launcher
will not silently guess the optimizer, batch size, decoder dimensions or split.
It preserves all resolved training settings and records changes in `changes.json`.
It inherits the data's online/offline cache policy unchanged. The launcher starts
an online target hidden-state server, so use your working online baseline config.
It does not resume an old diffusion checkpoint into the AR objective.

```bash
cd /home/n84449292/a84384644/attention_head_forked/attention_prefix_head
export BASELINE_CONFIG="/actual/path/to/dflash_qwen3_4b_openperfectblend_bs16/checkpoints/run.yaml"

# Optional only if paths in the baseline have moved:
# export MODEL="/actual/path/to/Qwen3-4B"
# export DATASET="/actual/path/to/open_perfectblend.qwen3-4b-rollout.qwen3.seq3072"

# Adjust these to your free devices. Defaults are shown.
export VLLM_NPUS=0,1 VLLM_DP=2
export TRAIN_NPUS=2,3,4,5,6,7 NUM_TRAIN_NPUS=6

CHECK_ONLY=1 OUTPUT_DIR="$PWD/output/ar_config_check" bash Evaluator/train_dflash_ar.sh
cat output/ar_config_check/changes.json

# Fresh output directory for actual training:
OUTPUT_DIR="$PWD/output/qwen3_4b_ar16" bash Evaluator/train_dflash_ar.sh

# Optional separate unchanged DFlash training baseline:
MODE=dflash OUTPUT_DIR="$PWD/output/qwen3_4b_dflash16_matched" bash Evaluator/train_dflash_ar.sh
```

The launcher inherits attention-layer/window settings too: it does not silently
convert a sliding-window baseline to full attention. If your reference uses full
attention, the AR model uses the same full-attention layers. It always makes the
within-block token mask causal. Device/port and model-identity checks precede
training. `CHECK_ONLY` resolves config; it is not an NPU execution test.

## Measure actual acceptance

Create a held-out JSONL file, one object per line, using either `{"prompt":"..."}`
or `{"input_ids":[...]}`. Text prompts use the Qwen chat template with thinking
disabled. Use identical held-out prompts and token budgets across checkpoints.

```bash
export PYTHONPATH="$PWD/speculators/src:$PWD/speculators/hs_connectors/src:${PYTHONPATH:-}"
ASCEND_RT_VISIBLE_DEVICES=0 python speculators/scripts/evaluate_dflash_ar.py \
  --target /actual/path/to/Qwen3-4B \
  --draft "$PWD/output/qwen3_4b_ar16/checkpoints/4" \
  --prompts /actual/path/to/heldout.jsonl \
  --device npu:0 --num-prompts 128 --max-new-tokens 2048 \
  --check-greedy 3 --output "$PWD/output/ar16_eval_epoch4"
```

This uses Hugging Face on one device and the repository's cached `LocalTarget`.
It is an **acceptance/correctness reference, not optimized vLLM serving or a
speed benchmark**. It keeps target-prefix features resident and must fit both
models and caches in memory. It currently supports one unpadded request at a time,
full vocabulary, and SDPA/eager draft inference. Auxiliary IDs use the repository's
hidden-state indexing (e.g. 1,9,17,25,33); final-normalized-layer IDs are rejected.

`summary.json` reports full-16-proposal rounds separately from budget-shortened
rounds, accepted-prefix histograms, draft acceptance rate and pooled AL including
one bonus. Raw verification counts can include candidates after EOS; emitted
counts and output IDs are retained separately. Initial/final standalone target
anchor tokens are not proposal rounds. Compare **mean full-block acceptance**,
not the maximum block observed. Training's `teacher_forced_eal` is diagnostic,
not measured free-running acceptance. Select the best epoch on validation prompts,
then report once on a separate held-out test set.

## Validation and limits

CPU tests cover parameter-layout equality, future-token/target-feature leakage,
shifted labels, packed-document boundaries, loss gradients, SDPA/eager cached
rollout equivalence, target-only greedy equivalence, EOS, all-accept and rejection
at every position, and configuration registration. Actual Ascend distributed
training, vLLM hidden-state-server interoperability and Qwen3-4B acceptance must
be run in your environment. The supplied patch does not change the NPU proposer
or claim an NPU speedup.
