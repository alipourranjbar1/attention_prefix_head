All cached benchmarks: acceptance-head evaluation
================================================
Upload and extract all_benchmarks_acceptance_eval.zip, then:
    cd all_benchmarks_acceptance_eval
    bash run_all_benchmarks.sh

Default selection:
GSM8K 128 test rows, MATH-500 128 test rows, HumanEval all 164,
MBPP 128 sanitized test rows, LiveCodeBench 128 rows,
AIME 2025 all 30, MT-Bench all 80 first-turn prompts, Alpaca 128 rows.
These are first-N selections; no random task selection is implied.
This includes all eight benchmark names, not every row of every dataset.

For EVERY ROW in all eight cached datasets (including the large Alpaca set):
    EVAL_ALL_ROWS=1 bash run_all_benchmarks.sh
For a different cap on the four larger benchmarks plus LiveCodeBench:
    EVAL_PROMPTS_PER_DATASET=256 bash run_all_benchmarks.sh
For a subset:
    BENCHMARKS="gsm8k humaneval" bash run_all_benchmarks.sh

Default hardware: target server NPU 0; draft/head evaluator NPU 2.
Only one target replica is started by default. A generation server is used
for all datasets, then stopped; a hidden-state server replaces it and is
reused for all evaluations. The two target servers are not simultaneous.
    EVAL_TARGET_NPUS=0 EVAL_NPU=6 bash run_all_benchmarks.sh
Choose unused devices. The script terminates only its own process groups.

Target, draft, sequence budget, and head architecture come from:
output/acceptance_multi_20261020_033003_1552425/checkpoints/final.pt
Default repository:
/home/n84449292/a84384644/attention_imp_code/vLLM_NPU_spec_main
Environment: /home/n84449292/a84384644/attention_modif; CANN 9.1.0.

All datasets are read from local Arrow caches without Hub access.
The named caches use known file/config paths from your listing, selecting
the newest matching raw split file by mtime. LiveCodeBench was not identifiable
by name in your listing. The resolver scans dataset_info.json for the
code-generation schema question_content/question_id plus associated columns.
It ignores cache-*.arrow transform/index files and reads the base split shards.
It chooses the newest revision within a single builder/config. If multiple
configs match, or none match, it stops BEFORE starting model servers and
prints the candidates. Specify the desired local source explicitly:
    EVAL_LCB_PATH=/absolute/path/to/lcb/shard_folder bash run_all_benchmarks.sh
Also accepts a single .arrow file or a Dataset.save_to_disk directory.
A DatasetDict root, JSON/JSONL file, and remote Hub ID are not accepted by
this override. Use the correct split directory/Arrow files.
The manifest records exactly which files and how many rows were selected.
All requested datasets must pass preflight; none is silently skipped.

Generation: one greedy response, non-thinking chat template, max 1024 new
tokens per prompt. HumanEval uses the same prompt wrapper as version 2.
Other adapters use task instructions; these may differ from the prompts in
your older benchmarking scripts. MT-Bench uses the FIRST turn only. MBPP
uses its task prompt; reference code and test lists are not appended. LCB
uses the problem statement and starter code, not hidden test cases.
Alpaca uses instruction plus input, never its reference output. Math datasets
use the question/problem without reference answers. This is acceptance
prediction evaluation on generated contexts, not benchmark answer accuracy.

Exact generated token IDs are preserved using vLLM token-ID logprobs.
The BatchEncoding conversion fix is included. Dataset.save_to_disk writes
input_ids, Boolean loss_mask, and seq_len as required by your ArrowDataset.
The head and draft remain frozen. Features and greedy-label verification
match the supplied trainer; checkpoint fingerprints must match.
Rows too long for the sequence budget cause an explicit failure, not silent
prompt truncation. Responses can reach the generation cap; counts are saved
in each prepared/summary.json. EOS and short-block exclusions match the
previous evaluator. All tasks are visited, but block positions are sampled
with the checkpoint's anchors-per-pack setting, not exhaustively evaluated.

Outputs: output/all_benchmarks_acceptance_<timestamp>_<pid>/
- manifest.json: selected paths, columns, and row counts
- <benchmark>/prepared/: generated responses and tokenized dataset
- <benchmark>/evaluation/metrics.json and confusion_matrix.csv
- <benchmark>/evaluation/directional_errors.txt, .csv, .json
- summary.csv: benchmark-level accuracy and signed error rates
- all_directional_errors.txt: ALL per-benchmark directional tables together

Send all_directional_errors.txt for analysis.
Exact, 1..K tokens LESS, and 1..K tokens MORE are reported separately.
Percentages use all evaluated blocks WITHIN EACH BENCHMARK. Counts refer to
block-level length predictions. LESS means predicted < actual; MORE means
predicted > actual. No combined either-direction error bins are used in
these detailed reports.

Overlap between evaluation data and PerfectBlend has NOT been checked;
these are separate benchmark sources, not proven contamination-free data.

Overrides: EVAL_HEAD, EVAL_REPO, BENCHMARK_OUTPUT, EVAL_CACHE_ROOT,
EVAL_LCB_PATH, BENCHMARKS, EVAL_ALL_ROWS, EVAL_PROMPTS_PER_DATASET,
EVAL_MAX_NEW_TOKENS (1024), EVAL_GENERATION_CONCURRENCY (4),
GENERATION_PORT (8116), EVAL_PORT (8115), EVAL_TARGET_NPUS (0), EVAL_NPU (2),
ACCEPTANCE_ENV_ROOT and ACCEPTANCE_CANN_ROOT.
Each run needs a fresh BENCHMARK_OUTPUT. Existing prepared files are preserved
on a failure, but this launcher does not automatically resume a partial run.

Validation: Python compilation, Bash syntax, prompt-adapter field tests,
LCB schema-discovery/ambiguity tests and signed-error metric tests.
End-to-end execution requires your Ascend models and environment.
