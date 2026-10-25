# Batched acceptance-length inference (eager)

This extension supports per-request acceptance predictions in synchronous padded
batches. It does not change the trained head or its features. Both local_prefix_v2
and compiled_prefix_v3 are supported. The head stays in FP32. Target and prefix
selector graphs are disabled for batches larger than one.

## Install

Apply `linear-confidence-batch.patch` over the original confidence patch AND its
local-v2 fix. It is an incremental patch, not a replacement for those patches.
Run `git apply --check` before `git apply`. Restart serving processes after applying.
Editable installations do not require reinstalling Python-only changes.

## Launch

```bash
# First smoke test with 4 active requests, then repeat with 16 and 128.
MAX_NUM_SEQS=4 NPU=0 PORT=8216 bash Evaluator/run_server_prefix_linear_batch.sh

# Original prediction policy: independently choose argmax length for each request.
MAX_NUM_SEQS=128 LINEAR_CONFIDENCE_POLICY=argmax \
    bash Evaluator/run_server_prefix_linear_batch.sh

# Experimental DSpark-style shared verification budget.
MAX_NUM_SEQS=128 LINEAR_CONFIDENCE_POLICY=budget LINEAR_CONFIDENCE_BUDGET=7 \
    bash Evaluator/run_server_prefix_linear_batch.sh

# Same capacity and execution settings, with the head disabled (baseline).
MAX_NUM_SEQS=128 LINEAR_CONFIDENCE=0 \
    bash Evaluator/run_server_prefix_linear_batch.sh
```

These commands run servers one at a time on card 0. Override NPU and PORT as needed.
Defaults use attention_prefix_ali, CANN 9.1.0, prefix_drafter_check/checkpoints/4,
and linear_head_selector/checkpoints/final.pt on the user's server.
`MAX_NUM_SEQS` is an active sequence limit, not a guarantee that all requests fit.
Model length, token budget and free KV memory may limit batching. Client concurrency
must also be raised. No batch-128 NPU test or speed improvement is claimed locally.

## Policies

- `argmax` (default): predicts one count 0..15 for each request independently.
- `budget`: fixed mean draft verification budget 1..15 across the active batch;
  at least one draft token per request. Remaining budget goes to requests with
  the largest predicted prefix-survival probabilities. The entire distribution
  is used, not only its argmax. A budget of 7 does NOT force every request to 7.
  For B requests the selected lengths sum to B*7 before scheduler clipping.
  With batch one this becomes a fixed length of 7, so use it to study batching.

For this categorical head, survival at position j is P(L >= j), a tail sum of
softmax probabilities. Do not take cumulative products of these tail sums.
Probability calibration is unverified; tune budgets on held-out performance data.

The draft still generates all 15 proposals; only target verification is truncated.
Target sampling and rejection verification remain unchanged. Prediction is not a
replacement for verification, and discarded proposals are never emitted directly.
Fewer verifier positions need not mean lower latency; use a measured sweep of budgets
5, 7, 9, 11, 15 with the same prompts, output limits and active batch size.

## What DSpark does in this repository

`vllm-ascend/vllm_ascend/spec_decode/utils.py:DynamicSpecScheduler`:

1. DSpark estimates conditional acceptance probabilities using its confidence
   head on draft hidden states and Markov embeddings.
2. A cumulative product gives the probability of accepting each full prefix.
3. A shared budget starts at 5 per request. By default it is recomputed every
   16 updates using survival >= 0.3, rounded upward to an odd draft budget where
   possible (the target also processes an anchor/bonus position).
4. Every request receives at least one proposal. Global top-k survival positions
   allocate the remaining budget, producing different verification lengths.
5. The runner slices each request's proposal list before passing it to scheduling.

This does not load multiple checkpoint block sizes: it generates one maximum block
and varies the verified prefix. Dynamic DSpark must be explicitly configured;
plain DSpark serving is not evidence that adaptive verification is enabled.
The code comments mention A3 measurements, not a validated 910B speedup.

The new `budget` policy adapts the same allocation idea to the existing trained
categorical head. It intentionally uses a fixed configurable budget for controlled
experiments; it does not copy DSpark's periodic heuristic or claim its speedup.

## Synchronization and request association

One batched head call produces [B,16] logits. The result and selected tokens are
copied to a reusable pinned CPU buffer on the runner's existing draft-copy stream,
before its existing completion event. The runner waits on that event before reading
both outputs. There is no separate blocking `.cpu()` copy per request or per head.
A snapshot of request IDs binds each row, including reordered/subset submissions.
Mismatched tokens, unknown IDs, duplicate IDs, invalid logits and missing predictions
fail closed. Pending state is reset on every proposal and warmup. Async scheduling,
TP/PP/DP > 1, unpadded drafter batches and collection mode remain unsupported.

## Validation

```bash
python vllm-ascend/tests/ut/spec_decode/test_linear_confidence_standalone.py
```

CPU tests exercise varying batch lengths, request reorder/replacement, budget
conservation, all-zero/full acceptance, invalid prediction state, the real runner
copy/event ordering with mocked streams, and actual local-v2 reference/fast score
parity on batched tensors. They do not execute Ascend kernels.

Live smoke test, same prompt set against baseline then head enabled:

```bash
python vllm-ascend/tests/e2e/linear_confidence/check_server.py \
    --concurrency 4 --num-prompts 16 --output baseline.json
# Restart server with head enabled, same max_num_seqs and execution settings.
python vllm-ascend/tests/e2e/linear_confidence/check_server.py \
    --concurrency 4 --num-prompts 16 --baseline baseline.json --output confidence.json
```

Repeat at 16 and 128 client concurrency with matching server capacity. Check server
metrics for `Running: N reqs` with N > 1, and compare token IDs. Also test context
limits, EOS, cancelled requests and repeated request turnover before production.
Keep launch.json, server.log and responses together for provenance.
Use INFO logging during performance measurement; DEBUG probes synchronize the NPU.
