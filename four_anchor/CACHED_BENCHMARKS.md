> For the new **19-position training and inference recipe**, see [TRAIN_BONUS19.md](TRAIN_BONUS19.md). This document retains the earlier 20-position experiment and benchmark instructions.

# Offline evaluation across cached benchmarks

`scripts/run_cached_benchmarks.py` is a standalone controller for the recovered
`known_bonus_cached_v1` inference implementation at commit
`9a2671f3283015952c73265c6bf3c3741c22a045`. It validates the inference files before
running. An existing recovery worktree is reused; if necessary, a detached
worktree of that commit is created. Training files and checkpoints are not edited.

The suite covers GSM8K, MATH-500, HumanEval, sanitized MBPP, LiveCodeBench,
AIME25, MT-Bench, and Alpaca. It reads only cached sources, with no dataset Hub
requests. It prefers compatible frozen prompt manifests from the existing
Evaluator directories, otherwise reads HF dataset Arrow caches, local Hub
snapshots, or save_to_disk datasets. Manifest tokenizer identity and thinking
mode must match. Exact input token IDs and source information are saved.

Defaults:

- First checkpoint: `checkpoint-016699` of the existing four-anchor run.
- Up to 128 prompts per available benchmark; smaller sources use every row.
- Physical NPU 1, concurrency 1, two draft passes, 15 speculative candidates.
- Four past anchors plus the separate bonus embedding: 20 drafter query positions.
- Greedy BF16, thinking off, target and drafter graphs off, 2048 new tokens maximum.
- MT-Bench uses the first user turn only. Alpaca remains a separate benchmark.
- Each benchmark gets a fresh engine using the existing tested launcher.
- Context size is at least 4096 and automatically increased to fit selected
  prompts, generation budget and redraft positions. Inputs are not truncated.

Place the downloaded script in `/home/n84449292/a84384644/attention_head_forked/`,
activate `attention_prefix_ali`, and run:

```bash
python /home/n84449292/a84384644/attention_head_forked/run_cached_benchmarks.py \
  --npu 1 --limit 128 --max-new-tokens 2048
```

Run after the previous job on NPU 1 has finished. The CANN environment and pinned
vLLM submodules are set up by the existing launcher. The controller itself does
not initialize an accelerator while discovering or tokenizing datasets.

Useful options:

```bash
# Prepare all prompts and report missing caches, with no NPU engine.
python run_cached_benchmarks.py --prepare-only

# Add another local cache root or an existing frozen Evaluator manifest.
python run_cached_benchmarks.py --cache-dir /path/to/cache
python run_cached_benchmarks.py --manifest /path/to/retrace_eval_prompts_all_v2.json

# Explicit source override: JSON/JSONL, Parquet, Arrow, or save_to_disk directory.
python run_cached_benchmarks.py --source gsm8k=/path/to/gsm8k_test.jsonl

# Also measure matching AR baselines and check exact token agreement.
python run_cached_benchmarks.py --with-baseline

# Every row in each selected cached source; Alpaca may contain 52,002 rows.
python run_cached_benchmarks.py --limit 0
```

With `--limit 0`, raw cached datasets are preferred over frozen subsets. If only
a frozen subset is available, the runner reports its actual row count. Cached
LiveCodeBench shards up to release_v5 are used by default; missing shard names
are explicitly recorded and printed. This is not a claim to have evaluated
every example in a full release when only a subset is cached.

The default result directory is
`output/four_anchor_dflash_20261031_025300_2111006/evaluation/cached_suite_<timestamp>_<pid>/`.
Each benchmark contains `prompts.jsonl`, `two_pass/eval.log`,
`two_pass/per_prompt.jsonl`, and `two_pass/summary.json`. Optional AR results
are in the corresponding `ar/` directory. Root files are:

- `plan.json`: dataset discovery, selected row counts, context budgets, input
  hashes, runtime identity and current completion state.
- `summary.csv` and `summary.md`: per-benchmark acceptance and throughput table.
- `suite_summary.json`: full summaries, source metadata, failures and pooled
  metrics for completed benchmarks.

Acceptance includes one correction/bonus and excludes anchors, following
native vLLM accounting before EOS/output-budget truncation. Throughput is total
generated tokens divided by measured generation time, including prefill but
excluding model loading and warmup. No reference answers or private code tests
are placed in freshly formatted prompts; generated code is never executed.
GSM8K answer accuracy, code pass@1, and other task accuracy are not measured.
Speedup and exact token agreement are available only with `--with-baseline`.

Absent caches are reported as unavailable. Formatting errors and failed runs
are recorded separately; other prepared benchmarks continue. Interrupting a
run preserves completed summaries. Each invocation uses a fresh output folder.

Validation covers all eight prompt formats, exclusion of reference answers,
cache/config/split selection, corrupt/incomplete Arrow caches, cached LCB shard
reporting, tokenizer identity, tokenizers.Encoding normalization, context
budgets, complete offline preparation, failure continuation, launcher settings,
and rejection of the failed predictor-state runtime. A fresh-process preparation
test exercises the standalone downloadable script. No Ascend hardware benchmark
was run in the development environment.

Validation result: **28 tests passed**; Python compilation and Git whitespace
checks also passed. The existing inference and training implementations are
unchanged.
