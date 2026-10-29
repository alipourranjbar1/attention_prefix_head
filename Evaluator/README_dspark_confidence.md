# DSpark confidence accuracy and verifier cutting

This uses the **built-in DSpark confidence head** stored in the draft checkpoint,
separate from the DFlash-Prefix `linear_head_selector/final.pt` MLP. The DSpark
backbone and Markov proposer stay enabled in every mode.

Both Speculators exports and native Hugging Face Qwen3 DSpark checkpoints are
supported, including the legacy `DSparkDraftModel` architecture spelling.
For the released `deepseek-ai/dspark_qwen3_4b_block7` checkpoint, the config's
`block_size=7` counts seven proposal positions (sampling from the anchor by
default); the target verifies eight positions including its anchor/bonus.
Set `NUM_SPECULATIVE_TOKENS=7`. Do not edit the checkpoint to add a Speculators
marker or change its block size. The Ascend Qwen3 DSpark subclass is supported.

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

### Full cached benchmark suite with repeated timing

Use `serve_dspark_full_suite.sh off` in the server terminal and then
`eval_dspark_full_suite.sh` in the client terminal. After the three OFF passes
finish, stop the server and start `serve_dspark_full_suite.sh cut`. Run the same
client again for the three ON passes. Server startup prepares one independent
full prompt manifest. It first reuses complete splits from
`Evaluator/retrace_eval_prompts.json`, checking the target configuration,
tokenizer fingerprints, thinking mode, token hashes and unique source-row
coverage. That file already holds all 880 LiveCodeBench v5 prompts, so its raw
JSONL files do not need to exist in the Hugging Face cache. Partial splits
are rebuilt in full from offline sources;
the 128-prompt manifest is never accepted as full coverage. Missing datasets
stop preparation before inference; none are silently omitted. Reuse provenance
and the original formatting choices are recorded in the new manifest.

The suite covers GSM8K test, MATH-500 test, HumanEval test, sanitized MBPP test,
LiveCodeBench release v5, AIME25, and MT-Bench first user turns. Alpaca is
excluded from preparation, evaluation, and reporting. The exact counts are
printed during preparation. All runs use greedy decoding, the configured
concurrency, seven maximum proposals, and a default output cap of 2,048 tokens.
The maximum context and unchunked-prefill token budget are chosen once from the
longest full-dataset prompt plus that cap and the speculative safety margin,
then kept identical in both modes. Prompts are not silently truncated.

`CONCURRENCY` defaults to 64 and sets both client concurrency and server
`max_num_seqs`. `SUITE_DIR` defaults to
`output/dspark_full_benchmark_c${CONCURRENCY}` under the repository.
`REPEATS` defaults to 3 and must be at least 1; `MAX_NEW_TOKENS` defaults to 2048.
With one pass, standard deviation and within-mode repeat agreement are unavailable.
Use the same preparation settings for both server launches. If full sources
require explicit paths, run the suite Python `prepare` command with repeated
`--source DATASET=LOCAL_PATH` arguments before starting the server. Both
launchers otherwise use the model, CANN, NPU 7, and port 8217 settings above.
Explicit sources take priority over saved splits. Use `REUSE_MANIFEST` to point
the server wrapper at another prepared manifest, or `--reuse-manifest` with the
Python preparation command. A model/tokenizer mismatch stops reuse rather than
silently mixing incompatible token IDs.

To repeat the experiment at concurrency 128, use
`CONCURRENCY=128 bash Evaluator/serve_dspark_full_suite.sh off` in the server
terminal and `CONCURRENCY=128 bash Evaluator/eval_dspark_full_suite.sh` in the
client terminal. After OFF finishes, stop the server, start it with `cut`, and
run the same client command again. Both modes retain three repeats, seven
proposals, and the same execution settings. The wrapper prefers the completed
concurrency-64 suite's frozen `prompts.json` when available, then falls back to
the repository's saved prompts. Concurrency-128 results go to a separate suite;
an explicit `SUITE_DIR` pointing at a suite with other settings is rejected.
AIME25 has only 30 prompts and MT-Bench has 80, so they cannot reach 128
simultaneous distinct requests. Their rows are not duplicated to fill slots.

The client resumes completed repeats from the suite's run index. Failed partial
attempts remain on disk and are not indexed as complete. No existing run is
overwritten. Change `SUITE_DIR` to start an independent experiment. Reports are:

- `suite_report.csv`: per-benchmark OFF/ON wall throughput means and standard
  deviations, descriptive throughput ratio, same-mode output agreement,
  cross-mode output agreement, and strict comparison status.
- `suite_report.json`: all repeat metrics, length-limited request counts,
  every mismatching request index, and the first token divergence for up to
  20 mismatch examples per paired repeat per benchmark.
- `suite_report.md`: compact results and limitations for review.
- `runs/*/generations/*.jsonl.gz`: generated token IDs, text, finish reasons,
  source IDs, and timings saved after each measured dataset interval.

Output differences remain visible even if the dataset is larger. Descriptive
ratios are separated from strict, identical-output-validated speedups. Means
and standard deviations from grouped-mode repeats are not confidence intervals
or a counterbalanced experiment. The geometric mean gives each benchmark
equal weight in the summary. HumanEval/MBPP/LCB
generated code is never executed; this suite measures performance and output
agreement, not answer correctness or pass@1.

### Two timing runs at concurrency 64

`run_dspark_speed_server.sh off` starts the full seven-proposal baseline.
`run_dspark_speed_server.sh cut` enables confidence computation and shortens
verification according to its policy. These use the local Qwen3-4B target and
native block7 checkpoint, NPU 7, port 8217, capacity 64, and identical eager
execution settings. Neither enables diagnostic tracing. The launcher creates
its own unique run directory and log; do not precreate the directory.

Keep the server in one terminal, then run `eval_dspark_speed.sh` in another
after startup. It submits 128 frozen GSM8K prompts with concurrency 64. Stop
the server, start the other mode, and run the same evaluator again. After both
complete, the second result directory contains `speed_comparison.json` and
`.csv`: `speedup` is head-on (`cut`) aggregate throughput divided by head-off
aggregate throughput. Head off still uses DSpark drafting and Markov sampling.
The client rejects traced accuracy servers and reports only validated matched
comparisons. Run directories and earlier results are retained.

### Separate server and evaluator for the released Qwen3-4B block7 checkpoint

`serve_dspark4b_confidence.sh` uses the local Qwen3-4B target and
`/share/canada_group_folder/ckpt/dspark_qwen3_4b_block7`, NPU 7, port 8217,
eager execution, and **exactly seven proposals**, overriding inherited values.
It runs the server in the foreground and displays its startup log. The
evaluator never starts or stops a server. Both scripts can be run from any
working directory; their default `ROOT` is the user's existing repository.

```bash
# Terminal 1: keep running until you want to switch modes.
bash Evaluator/serve_dspark4b_confidence.sh observe accuracy

# Terminal 2: run after server startup finishes.
bash Evaluator/evaluate_dspark4b_confidence.sh
```

The server takes two arguments: `off|observe|cut` and `accuracy|speed`.
Accuracy collection supports `observe` and `cut`; `off` is used for timing.
Stop the server with Ctrl+C between modes, then restart it with new arguments.
Run the same evaluator command for every mode. It discovers the exact launch
record and generates confidence metrics only when tracing is enabled.

For speed, run `off speed`, `observe speed`, and `cut speed` separately and
evaluate each. Once all three have completed, the evaluator compares the latest
timing results for that concurrency using the existing strict comparison checks.
Results are retained under `output/dspark4b_manual/c<concurrency>/`; only the
bookmarks to the latest timing results are updated. Accuracy results are never
used as timing baselines. The evaluator checks readiness once and tells you to
inspect the server terminal if it is not ready.

For concurrency 128, launch with `MAX_NUM_SEQS=128` and evaluate with
`CONCURRENCY=128`. Both defaults are 1. `NUM_PROMPTS` defaults to 128 and
`MAX_NEW_TOKENS` to 2048; use 4 and 256 for a short smoke run.

### Automated comparison for other checkpoints

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
and unrelated traffic. The report supports both the completion ID plus `-0`
and the additional eight-character hexadecimal suffix that vLLM appends to
internal request IDs. It rejects ambiguous mappings and duplicate trace rounds;
it never includes all traces as a fallback. `measured_requests` and
`requests_with_traces` show matching coverage in each report cell. Requests
that finish before a speculative round may have no traces.

If generation completed but confidence reporting failed, update
`dspark_confidence_report.py` and rerun only the report on the saved
`run_info.json`. The trace directory is recovered from that run's embedded
server manifest when `--trace-dir` is omitted. This needs neither a running
server nor an NPU and does not repeat inference. Keep the saved trace files.

EOS rounds are excluded from accuracy. If a shortened
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

Speed schema v2 uses `wall_tokens_per_second` (total output tokens divided by
elapsed benchmark wall time) for `speedup`. The evaluator's legacy
`tokens_per_second` divides by the sum of individual request durations, which
overlap under concurrency; it is not aggregate server throughput. That ratio
is retained separately as `request_time_speedup`. Existing `run_info.json`
files already contain both measurements, so comparison can be regenerated
without repeating inference. Diagnostic traced runs remain excluded.

The report's `compare` command accepts `--baseline` plus either `--observe`,
`--cut`, or both. Thus head-off versus head-on timing does not require a cut run.

## Validation

The existing dynamic call passed the whole backbone tensor, potentially
including anchor rows. It now uses `sample_hidden_states` aligned to proposals.
The new observer uses those same rows. Predictions and token identities are
copied before the existing draft-copy event and consumed after its existing
synchronization. Request IDs bind rows across reordering. No extra D2H event or
per-request device `.item()` is added for this transfer; trace I/O is diagnostic.

```bash
python Evaluator/tests/test_dspark_confidence_report.py
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
