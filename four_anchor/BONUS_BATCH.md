# Bonus20 inference with concurrency 32

This isolated inference branch uses the existing format-3 `target_four` checkpoint.
It does not modify the training checkout, checkpoint, Python environment, installed
vLLM/Ascend source, or the training evaluation queue. The launcher selects physical
NPU 7 only. No dependency installation or process termination is performed.

| Stage | Inputs per request | Hidden-state source |
|---|---|---|
| Draft round one | 4 processed anchors + known bonus + 15 masks = 20 | Same-position target features for the four anchors; embedding only for bonus |
| Draft round two | 4 preceding anchors + 15 masks = 19 | All four features from corresponding round-one drafter outputs |
| Native verification | Known bonus + 15 speculative candidates = 16 | One native target verification forward |

The four anchor positions never enter acceptance accounting. Reported acceptance
is `1 + accepted speculative tokens / request verification cycles`, capped at 16
before EOS/output-budget truncation. The newly sampled bonus has no own target
state until the next normal verification processes it. There are no extra target
forwards to materialize that state.

## Batching and timing

One vLLM engine uses native continuous batching with `max_num_seqs=32`. Each request
has a separate cache slot; slot lookup uses request IDs, not scheduler row numbers.
The target's packed query offsets separate accepted features for each request.
Both draft rounds run on batched tensors, with per-request boundaries, lengths,
position IDs and masks. Rejected suffixes, padded context and other requests'
states are excluded. Finished/preempted requests release cache slots; a missing
prefix raises an error instead of silently reusing unrelated states.

All benchmark prompts are submitted together; vLLM runs up to 32 simultaneously
and refills completed slots. Actual occupancy can be smaller for short datasets,
prefill scheduling, the end of a workload or KV-cache pressure. `max_draft_batch`
records the largest measured drafter batch, and `draft_requests / draft_batches`
gives average drafting occupancy. This is one NPU, not 32 engines or 32 NPUs.

Aggregate tokens/s is total generated tokens divided by elapsed time for the
entire queued workload, including prefill and excluding loading/warmup. Overlapping
request latencies are saved separately and never summed for throughput. Compare
speedup only against a baseline run with concurrency 32 and matching prompts and
settings. Previous concurrency-one throughput is not a matching baseline.

Target and drafter graphs are OFF. Greedy decoding, BF16, TP=PP=DP=1,
synchronous scheduling and prefix caching disabled are required. Eager PyTorch
SDPA executes the drafter; native vLLM-Ascend executes the target and verification.
The existing pinned vLLM and Ascend checkouts are read-only dependencies.

## Run

```bash
conda activate attention_prefix_ali
ROOT=/home/n84449292/a84384644/attention_head_forked/four-anchor-bonus20-c32-inference
EVAL_NPU=7 CONCURRENCY=32 bash "$ROOT/four_anchor/scripts/bonus_batch_ascend.sh" smoke
EVAL_NPU=7 CONCURRENCY=32 bash "$ROOT/four_anchor/scripts/bonus_batch_ascend.sh" benchmarks
```

`smoke` runs the first 32 GSM8K prompts with 128 output tokens maximum, exercising
multi-request inference. `evaluate` runs 128 GSM8K prompts with a 2048-token budget.
`benchmarks` runs up to 128 prompts from each of the eight frozen cached benchmarks;
smaller datasets use their existing available count. No baseline runs by default.
`WITH_BASELINE=1` enables a separate matching batched AR run and exact token check.

Defaults read checkpoint `checkpoint-022309` and frozen prompt plan from:
`four-anchor-15-train-benchmarks/output/target19_20261104_024656_3124841`.
Override `DRAFT`, `BENCHMARK_PLAN`, or `VLLM_SOURCE_ROOT` if their locations differ.
No network dataset downloads are attempted. `check` prints engine settings without
starting an engine; `prepare` checks/reuses the complete frozen benchmark plan.

Results are written only under this new checkout:
`output/bonus20_c32_<timestamp>_<pid>/`. For the suite, open `summary.md`,
`summary.csv`, `suite_summary.json`, and `<benchmark>/two_pass/summary.json`.
Per-prompt IDs, outputs and acceptance metrics are in `per_prompt.jsonl` there.
GSM8K-only runs use `two_pass/summary.json`. The launcher prints the exact path.

## Validation and limits

CPU tests compare batched outputs/hidden states against independent single-request
bonus20 inference for variable lengths and batches up to 32. Tests cover scheduler
row reordering, rejected-feature trimming, partial prefill, slot reuse, all
acceptance counts, pooled metrics, queued evaluation and installation isolation.
The package was not executed on an Ascend NPU here. Qwen3/Ascend correctness,
memory use, throughput and token parity require the on-device run.

The checkpoint was trained with 19 query positions. Adding the explicit bonus
slot and using drafter features for all four second-pass anchors is an inference
experiment, not a weight conversion or a guarantee of unchanged acceptance.
Batch shape can also change floating-point results. Raw throughput/acceptance
alone do not establish output parity or benchmark answer accuracy.

## Save this branch after review

```bash
cd "$ROOT"
python four_anchor/scripts/install_bonus_batch.py --stage
git diff --cached --check
git commit -m "Add isolated batched bonus20 inference with concurrency 32"
git push -u origin four-anchor-bonus20-c32-inference
```

The allowlist stages inference code and documentation only, excluding checkpoints,
training outputs and generated evaluation data. Training can continue on its
existing branch while this separate worktree is tested or committed.
