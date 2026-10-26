# DFlash prefix attention + ReTrace + top-16 distillation

Based on `acceptance-single-repo` at
`0d4d4d196be53de29b8c7afdad79934a0744c7c9`.
The complete `retrace/` directory already matched MohammadMahdi1375/speculators
commit `89b5caa300f473c1e3127497e6f594ba024057d5` before this integration.
This change connects that implementation to the existing prefix attention selector.

## Training behavior

Start from a trained **dflash_prefix checkpoint**, including `prefix_head.*`.
Conversion strictly checks and copies every backbone and selector tensor; only
the three ReTrace matrices are new. Its residual projection starts at zero.
The full-attention architecture, block size, target layer IDs, and prefix-head
architecture/settings are copied from the source. The backbone is unfrozen for
joint retraining. Frozen target embeddings/output projection/norm remain frozen.

1. ReTrace conditions masked proposal embeddings using the preceding verified
   rejected suffix. The anchor is unchanged.
2. DFlash computes proposal states and full-vocabulary logits.
3. The existing prefix attention head selects the actual proposed tokens.
4. The target verifies that selected path. Only genuinely verified memory is
   carried forward; no clean future states are substituted for rejected states.
5. A successor trains only when the committed prefix agrees with the stored
   response. Document boundaries and response masks are respected.

The stored-response objective is:

`loss = full_vocab_KL + prefix_loss_alpha * selector_CE + 0.1 * top16_KL`

`top16_KL` means **KL(target || drafter)** on the target's top 16 token IDs.
Both distributions are renormalized over those same IDs, in FP32, temperature 1.
It matches relative probabilities within the target shortlist. The retained
full-vocabulary KL also supervises probability mass outside that shortlist.
The target is detached. The auxiliary losses use exp(-proposal_index / 4)
weighting and the valid-token denominator, preserving block-chunk invariance.
The selector's loss kind, reachability scope and damage weight come from the
source config; its valid-token denominator is fixed rather than a per-chunk
covered-token denominator. This differs from the original standalone prefix
trainer's selector normalization. The old joint trainer's schedule/guard is not
part of this stored-response Retrace recipe.

`distill_top_k=16` and `distill_kl_weight=0.1` are saved in the combined config.
Metrics include `selector_ce`, `top16_kl`, and `weighted_top16_kl`.
The prefix shortlist and the target top-16 distillation shortlist are independent.

## Launch on Ascend

Activate your existing environment and source CANN, or set `RETRACE_CANN_ROOT`.
Run from the root of this repository (which contains speculators/, vllm/, and
vllm-ascend/). Replace the three local input paths:

```bash
bash speculators/examples/train/train_retrace_prefix_stored.sh \
  --require-prefix \
  --target /path/to/Qwen3-4B \
  --base-dflash /path/to/trained/dflash_prefix/checkpoint \
  --data /path/to/open_perfectblend.qwen3-4b-rollout.qwen3.seq3072 \
  --output "$PWD/output/retrace_prefix_top16_smoke" \
  --server-npus 0,1 --trainer-npus 2,3,4,5,6,7 \
  --smoke
```

Use `--dry-run` to inspect the plan without starting NPUs. For a full run use a
fresh output directory and replace `--smoke` with your desired explicit budget,
for example `--epochs 8 --max-records 40000 --tokens-per-worker 3072`.
Those are example budgets, not inherited settings from your prior prefix run.
The launcher retains the source Retrace learning-rate default of 5e-5; use
`--lr` to override it. Use `bash RUN/resume_stored.sh` after interruption.

The run saves strict conversion fingerprints, initial and epoch checkpoints,
training provenance, target-server provenance and metrics. Check
`conditioned_pair_fraction` is nonzero; an epoch with no usable conditioned
pairs fails rather than silently training without ReTrace.

## Evaluation and limits

Combined checkpoints have algorithm/type `retrace_prefix` and runtime
architecture `ReTracePrefixDraftModel`; they must not be relabelled as vanilla
DFlash or dflash_prefix. The existing reference evaluator dispatches this type:

```bash
python -m speculators.models.retrace.evaluate \
  --target /path/to/Qwen3-4B --draft /path/to/combined/checkpoint \
  --device npu:0 --attention-backend sdpa \
  --data /path/to/held_out_prompts.jsonl \
  --output /path/to/new_evaluation.json
```

This integration covers stored-response retraining and reference greedy
proposal/verification. It does **not** add a combined native vLLM proposer or
change the independent paper_recipe/stored_recipe training entry points.
Use the launch command above. Native vLLM serving/graph-capture performance and
full NPU/HCCL training require separate validation. No throughput or acceptance
improvement is claimed before measurement.

The preserved attention head is `prefix_head` (token reranking). The separate
optional `acceptance_head.pt` sidecar used by Evaluator scripts predicts truncation
length; it is not trained or applied by this combined reference runtime and
must be recalibrated for a changed drafter before deployment.

## CPU tests

From speculators/ in an environment with its dependencies installed:

```bash
python -m pytest --confcutdir=tests/retrace_prefix tests/retrace_prefix -q
```

Tests cover loss direction/normalization/masking/gradients, all three selector
architectures, strict source-weight preservation, checkpoint roundtrips,
verified selected-path memory, BF16 gradients, future-context masking, block
chunk invariance, and unchanged vanilla ReTrace conversion.
