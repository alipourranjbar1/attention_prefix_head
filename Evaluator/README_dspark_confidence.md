# DSpark confidence accuracy and verifier cutting

This uses the **built-in DSpark confidence head** stored in the draft checkpoint,
separate from the DFlash-Prefix `linear_head_selector/final.pt` MLP. The DSpark
backbone and Markov proposer stay enabled in every mode.

| `DSPARK_CONFIDENCE_MODE` | Compute confidence | Shorten verifier input |
| --- | --- | --- |
| `off` | No | No |
| `observe` | Yes | No: measure against full verification |
| `cut` | Yes | Yes: submit the selected proposal prefix |

The full draft is generated in every mode. `NUM_SPECULATIVE_TOKENS` sets its
maximum proposal count (default 15). Cutting only changes how many proposals the
target verifies; it never bypasses verification. The target also processes one
anchor/correction position. Checkpoint block size and KV-cache page size are
separate from this verification length.

## Install

The feature branch is `dspark-confidence-eval`, based directly on
`acceptance-single-repo` commit `0d4d4d1`. It contains the DSpark confidence
evaluation changes without the AR branch's additions.

When the feature branch is available on your remote:

```bash
git fetch origin dspark-confidence-eval
git switch dspark-confidence-eval
```

Alternatively, create the feature branch from the portable patch at the
repository root. Start from `acceptance-single-repo`:

```bash
git fetch origin acceptance-single-repo:refs/remotes/origin/acceptance-single-repo
git switch -c dspark-confidence-eval origin/acceptance-single-repo
git apply --check /path/to/dspark-confidence-eval.patch
git apply /path/to/dspark-confidence-eval.patch
```

Restart inference servers after applying. Launchers use the local Python sources
through `PYTHONPATH`; no compiled kernels are changed.

## Run

Activate the existing Ascend environment and select matching model directories:

```bash
export TARGET_MODEL=/path/to/Qwen3-4B
export DRAFT_MODEL=/path/to/dspark/checkpoints/4

# Smoke test: two accuracy phases plus three timing runs on an idle NPU.
NPU=6 CONCURRENCY=1 NUM_PROMPTS=4 MAX_NEW_TOKENS=256 \
    bash Evaluator/benchmark_dspark_confidence.sh

# Main comparison: same checkpoint, frozen GSM8K prompts, greedy decoding.
NPU=6 CONCURRENCY=1 NUM_PROMPTS=128 MAX_NEW_TOKENS=2048 \
    bash Evaluator/benchmark_dspark_confidence.sh

# Batched comparison: actual active batch size remains scheduler-dependent.
NPU=6 CONCURRENCY=128 NUM_PROMPTS=128 MAX_NEW_TOKENS=2048 \
    bash Evaluator/benchmark_dspark_confidence.sh
```

`ACTION=accuracy` runs full-block and cut-mode accuracy collection.
`ACTION=speed` runs only off/observe/cut timing. The default `ACTION=all` does both.
`DATASETS="gsm8k math500"` selects multiple datasets. `MANIFEST=/path/to/file.json`
reuses frozen prompts; otherwise a manifest is prepared from cached benchmark
datasets and checked against the target tokenizer/config. For the matching
Qwen3-4B snapshot, use `Evaluator/retrace_eval_prompts_all_v2.json` as `MANIFEST`.

Defaults: port 8217, target/draft eager, reference Markov head, maximum context
6400, maximum batched tokens 6400, memory utilization 0.85. Override `CANN_ROOT`
for other installations. Larger batches/context may need different memory
settings; every comparison mode reuses the same settings.

Batch one can also use the existing Markov graph, with identical target graph
buckets captured in all modes:

```bash
NPU=6 CONCURRENCY=1 TARGET_EXECUTION=graph HEAD_EXECUTION=fast \
    bash Evaluator/benchmark_dspark_confidence.sh
```

The batched launcher requires eager target and reference head. The confidence
path requires synchronous scheduling, unchunked prefill, full-vocabulary Qwen3
DSpark, default penalties, greedy decoding, and TP=PP=DP=1. Missing confidence
weights and conflicting dynamic policies are rejected.

## Choose cutting independently

```bash
# Predictions, full verification:
NPU=6 DSPARK_CONFIDENCE_MODE=observe bash Evaluator/run_server_dspark_confidence.sh

# Predictions, shortened verification:
NPU=6 DSPARK_CONFIDENCE_MODE=cut bash Evaluator/run_server_dspark_confidence.sh

# Full verification, no confidence computation:
NPU=6 DSPARK_CONFIDENCE_MODE=off bash Evaluator/run_server_dspark_confidence.sh
```

Set `DSPARK_CONFIDENCE_TRACE_DIR=/path/to/new/traces` to collect labels.
**Leave it unset for timing.** The integrated benchmark runs a separate traced
accuracy phase for each of observe/cut and disables tracing in all three timing phases.

## Predictions and metrics

DSpark returns conditional acceptance estimates `q[j]`. Prefix survival is
`s[j] = product(q[0:j+1])`, as in the existing scheduler. The report distinguishes:

- **Predicted accepted length:** the mode of `P(L=0)=1-s[0]`,
  `P(L=j)=s[j-1]-s[j]`, and `P(L=K)=s[K-1]`.
- **Expected accepted length:** `sum(s)`, also reported with one bonus added.
- **Selected verification budget:** the existing shared-budget policy,
  initially 5 proposals per request, updated every 16 proposal batches with
  survival threshold 0.3 and rounding up to odd budgets when possible. It keeps
  at least one proposal per request and distributes the rest across the batch.
  This cost policy is distinct from the modal prediction of acceptance.

Override `DSPARK_CONFIDENCE_INITIAL_BUDGET`,
`DSPARK_CONFIDENCE_UPDATE_INTERVAL`, and `DSPARK_CONFIDENCE_THRESHOLD` to sweep
the policy. Its existing periodic device synchronization remains included in
observe/cut timing.

`confidence_accuracy.json` contains:

- `full_block_accuracy`: exact, under, over, within-one percentages, MAE, signed
  error. Under means predicted < actual; over means predicted > actual.
- `selected_budget_vs_actual_full_length`: the same metrics for the budget.
- `before_cut`: mean predicted, expected and actual acceptance lengths.
- `counterfactual_cut_on_same_full_blocks`: `1 + min(actual, selected_budget)`,
  lost accepted tokens, and capped prediction accuracy on identical examples.
- `observed_after_verification`: actual verified length/acceptance, capped
  accuracy, and counts of censored and scheduler-clipped rounds.

Acceptance lengths include one target correction/bonus. Adding one to both
prediction and truth does not change exact/under/over classifications. Capped
accuracy is mechanically easier; use full-block accuracy to judge the head.
Counterfactual cutting on saved blocks is not end-to-end cut-mode performance.

`confidence_accuracy_after_cut.json` records the separate cut-mode diagnostic
run. Its `observed_after_verification` reports actual acceptance length and
prediction accuracy capped to each verified prefix. Unverified suffixes remain
censored; the full-block subset in this file is not an unbiased head-accuracy
estimate. Use `confidence_accuracy.json` for overall head accuracy.

Trace request IDs are matched to measured evaluator requests, excluding warmup
and unrelated traffic. EOS rounds are excluded from accuracy. If a shortened
prefix is fully accepted, its unverified suffix usually has no label. A target
bonus differing from the next proposal exposes an exact boundary; otherwise
the full label stays null/censored, unless the bonus confirms the last proposal
of the maximum block. Overall head accuracy must be measured in observe mode;
complete blocks in cut mode are a policy-selected subset.

`speed_comparison.json` and `.csv` report throughput, speedup versus head off,
pooled acceptance length, draft acceptance rate, TTFT and TPOT. A speedup ratio
below 1 is a slowdown. Weight fingerprints, source identities, prompt/output
hashes, sampling/runtime settings and metric isolation must match before a
speedup is reported. No speed gain is assumed; repeat measurements before
drawing performance conclusions.

## Validation

The existing dynamic call passed the whole backbone tensor, potentially
including anchor rows. It now uses `sample_hidden_states` aligned to proposals.
The new observer uses those same rows. Predictions and token identities are
copied before the existing draft-copy event and consumed after its existing
synchronization. Request IDs bind rows across reordering. No extra D2H event or
per-request device `.item()` is added for this transfer; trace I/O is diagnostic.

```bash
python vllm-ascend/tests/ut/spec_decode/test_dspark_confidence_standalone.py
python vllm-ascend/tests/ut/spec_decode/test_linear_confidence_standalone.py

# After an actual NPU ACTION=all run:
python vllm-ascend/tests/e2e/dspark_confidence/check_results.py \
    output/dspark_confidence_compare_<timestamp>_<pid>
```

The live result check requires complete measured labels, matched outputs and
valid timing comparisons. At concurrency one it also checks unchanged
speculative counts with head off versus observation. CPU tests do not certify
NPU kernels or establish a speedup.
