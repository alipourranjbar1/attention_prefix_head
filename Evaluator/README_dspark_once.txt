One ON run, then one OFF run, Qwen3-4B DSpark on NPU 7

Extract the ZIP at attention_prefix_head, activate attention_prefix_ali, then:
  bash Evaluator/run_dspark_on_then_off_once.sh

If the old evaluation server still occupies port 8217, stop that evaluation
server first. This script refuses to kill an unrelated process. It launches and
cleans up only its own evaluation server, including on error or Ctrl+C.

Fixed experiment:
  ON (confidence computation + verifier cutting), then OFF (no confidence).
  Exactly one pass per mode, concurrency 128, seven proposals, seed 42.
  Qwen3-4B target and /share/canada_group_folder/ckpt/dspark_qwen3_4b_block7.
  All saved rows from GSM8K, MATH-500, HumanEval, MBPP, LiveCodeBench v5,
  AIME25 and MT-Bench first turns; no Alpaca. Natural EOS; max output 2048.
  Identical frozen prompts; eager target/draft; tracing off; memory fraction .5.
  Fresh server for each mode. AIME25 (30) and MT-Bench (80) cannot fill 128 slots.

Reproducibility controls:
  - Greedy target/draft and explicit standard (not synthetic) verification.
  - Fixed generation and Python hash seeds.
  - VLLM_BATCH_INVARIANT=1, HCCL_DETERMINISTIC=strict, LCCL_DETERMINISTIC=1,
    ATB_MATMUL_SHUFFLE_K_ENABLE=0, ATB_LLM_LCOC_ENABLE=0 in both modes.
  - Mandatory AscendC batch_invariant_ops preflight. Triton-only fallback is
    rejected because this branch does not replace fused attention in that path.
  - Server log must confirm batch-invariant initialization before evaluation.
  - Kernel controls are not proof of output parity. Every final output hash must
    match before any speedup is marked strict/validated. A mismatch saves all
    results and exits status 2. It must be investigated, not hidden or ignored.

Prerequisite if preflight reports missing batch_invariant_ops:
  Activate the same Conda and source both CANN 9.1.0 set_env.sh files, then use
  the matching source checkout's documented installer for Ascend 910B:
    cd /home/n84449292/a84384644/attention_head_forked/attention_prefix_head/vllm-ascend
    bash csrc/build_batch_invariant_ops.sh ascend910b
  This installs device operators and their Python extension and needs access to
  its download URLs. The runner does not modify your installed packages itself.
  Re-run the main script after installation. If initialization or custom-kernel
  execution fails, retain preflight.log/server.log; do not disable the check and
  present the resulting experiment as batch invariant.

Saved files:
  /home/n84449292/a84384644/attention_head_forked/results/
    latest_on_then_off.txt  -> text containing the new experiment directory
    dspark_qwen3_4b_c128_on_then_off_TIMESTAMP_PID/
      ON/summary.csv, ON/run_info.json, ON/generations/*.jsonl.gz
      OFF/summary.csv, OFF/run_info.json, OFF/generations/*.jsonl.gz
      suite_report.csv, suite_report.json, suite_report.md
      status.json, runtime_preflight.json, prompts.json, suite.json
      run.log, run_script.sh, logs/, server_cut_*/, server_off_*/

ON and OFF are relative symlinks to the saved single-pass run directories.
The summary includes wall-clock throughput and acceptance length; the JSON
includes output agreement and first-divergence examples. The existing evaluator
stores token IDs/text after timing. This update reads chunked HTTP SSE in 1024-
byte chunks rather than byte by byte, and saves monotonic request/token timestamps
for subsequent occupancy diagnostics. These timestamps compare within a run,
not across different processes or machines.

With only one measurement per mode, this experiment cannot quantify timing
variation. Batch-invariant kernels can change performance; compare ON/OFF within
this experiment, not directly against earlier default-kernel measurements.

Optional dry run (no runtime preflight, model loading, server or evaluation):
  CHECK_ONLY=1 bash Evaluator/run_dspark_on_then_off_once.sh
It checks paths/configuration and writes launch records in a new results folder.
No NPU performance or Qwen3/DSpark parity is claimed from a dry run.
