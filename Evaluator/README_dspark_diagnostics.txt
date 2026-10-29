DSpark concurrency-128 saved-run diagnostics

Extract this ZIP at the attention_prefix_head repository root. It only adds
diagnostic files. It does not replace evaluator.py, change the running server,
install packages, import torch or use an NPU. Python's standard library suffices.

Run:
  bash Evaluator/diagnose_dspark_c128.sh

The wrapper reads these existing directories under the repository output/:
  dspark_full_benchmark_c128
  dspark_c128_cut_training_20261029_032419_2332808

Override ORIGINAL_SUITE and RERUN_SUITE if required. Both suite paths must have
completed-run indexes (runs_off.json and/or runs_cut.json), not just summary.csv.
All indexed repeats are read, including the previous ON runs. The diagnostic
verifies saved generation hashes and totals if generation files are available.

Results go into a new output/dspark_diagnostic_TIMESTAMP_PID/ directory:
  diagnostic.txt  - compact tables and interpretation
  diagnostic.json - identities, per-repeat comparisons and mismatch indices
  per_run.csv     - tokens, wall seconds, overlap, acceptance and length statistics

Read WITHIN EACH GROUP first: does OFF vary against OFF, and ON against ON?
Read BETWEEN GROUPS next: do settings and ordered prompts match, and do output
hashes agree? Every cross-group repeat combination is examined; these are not
time-matched pairs. CV is sample standard deviation / mean, not a confidence
interval. Do not turn a small three-repeat CV into a claim of statistical proof.

Average client requests in flight is summed request time / wall time; it is not
the server's actual NPU batch size. Request start/end timestamps were not saved,
so an occupancy timeline or the exact time spent draining the batch cannot be
reconstructed. Changed overlap alone does not establish its underlying cause.

Missing generation files limit length diagnostics. Corrupted or inconsistent
saved generations stop analysis. Outputs and input files are never overwritten.
No generated code from benchmark responses is executed.

After reading diagnostics:
  - If outputs drift within a mode, inspect the saved first-divergence examples;
    same-mode drift does not make cross-mode mismatches automatically harmless.
  - If outputs only change across modes, isolate correctness with a small greedy
    test at concurrency 1, then repeat at high concurrency.
  - If outputs match but time varies, profile the CPU client and server and record
    request timestamps/NPU activity before attributing the gap to the head.
  - For fresh performance measurements, keep settings and background load fixed
    and alternate OFF/ON/OFF. If changing the HTTP client, rerun BOTH modes with
    that client; do not compare it to old results from a different client.
  - A fixed-output-length test is a separate throughput experiment. It can reduce
    output-length/drain differences but does not replace natural-EOS evaluation.
