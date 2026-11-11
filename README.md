# DSpark training speed audit and isolated trials

This package compares concrete execution changes on the existing Ascend stack.
It does not claim that graphs, DDP, or any candidate is always fastest.
No active code directory, dependency checkout, process, or checkpoint is edited.

## What the supplied logs show

Excluding startup step 1, the mean of the logged maximum-rank phase timings for
steps 2–10 is:

| Phase | Seconds | Share of summed phase timings |
|---|---:|---:|
| Input wait and transfer | 0.421 | 5.5% |
| Drafter/objective forward | 2.448 | 31.8% |
| Backward | 4.805 | 62.5% |
| Gradient normalization/optimizer | 0.018 | 0.2% |

The sum is 7.692 seconds. Maxima may come from different ranks; this is a
bottleneck indicator, not an exact decomposition of critical-path latency.
Teacher work is prefetched by the data-loader workers. Input time is the
remaining visible wait plus transfer, not total teacher compute time.
Eliminating that visible 0.421 s alone would give about 1.06x, not a major
end-to-end improvement. Teacher graphs may affect overlap and contention, so
the actual improvement must be measured.

The supplied rank-0 peak allocated memory was about 16.13 GiB; reserved memory
was about 34.56 GiB. This motivates testing less recomputation. It does not
prove other ranks or longer documents will fit.

## Audit findings and implemented candidates

1. **Full-vocabulary loss recomputation.** The previous non-reentrant loss
   checkpoint recomputes the frozen LM projection, Markov projection and
   distribution loss during backward. `HEAD_LOSS_BACKEND=cached_vjp` computes
   first-order gradients while each vocabulary chunk is resident, then saves
   input/head-weight gradients. It avoids that recomputation at the cost of
   additional saved Markov-weight gradients. Full-vocabulary CE/TV and detached
   confidence labels remain; trainable Markov and confidence gradients remain.
   Higher-order derivatives are unsupported. BF16 rounding can differ.
2. **Proposal projection batching.** `PROPOSAL_BASE_BATCHED=1` computes the
   frozen base logits for all 15 positions together. The Markov term still uses
   the preceding generated token sequentially. There is no parallel or
   teacher-forced substitution for the proposal chain.
3. **Backbone recomputation.** `GRADIENT_CHECKPOINTING=0` trades activation
   memory for less backward recomputation. The benchmark tests whether it fits
   and improves latency. This is distinct from saved model checkpoints.
4. **Vocabulary and attention chunk sizes.** The suite tests head chunks of
   512 and block chunks of 64. Larger chunks are not automatically faster:
   the current packed attention computes dense masked attention, including
   work on masked cross-block/document positions. Smaller block chunks reduce
   some of this wasted work but increase launches. A true ragged/fused
   attention rewrite is not included or claimed tested here.
5. **FSDP versus DDP.** Root-level FSDP is already used, avoiding repeated
   data-dependent per-layer collectives. DDP is tested as a fresh candidate
   only; it may reduce sharding communication if the replicated state fits.
6. **Target serving graph.** `TARGET_EXECUTION=graph` removes the teacher's
   forced eager option and compile-disabling environment only in its child
   process. It requests VLLM_COMPILE=3 plus PIECEWISE ACLGraph, suitable for
   the prefill-heavy hidden-state workload. Attention remains outside graph
   pieces. The trainer's compile environment is unchanged. On Ascend these
   are NPU/ACL graphs, despite vLLM's `cudagraph` option names.
7. **Existing loss graph stays off.** The supplied real-device test measured
   that separate loss-only graph slower than eager. It is not enabled here.
8. **Packing, data and supervision stay the same.** Every trial uses 9,216
   tokens per training rank, up to 1,536 sampled blocks/rank, 15 proposals,
   four anchors, 20/19 query layouts, confidence-head boundary selection,
   exploration=0.5, the mismatch cap, five layers, CE=0.1/TV=0.9 and confidence
   weight 1. Up to 46,080 prediction positions/rank are projected over the
   151,936-token vocabulary across the two losses. The sequence budget is
   not the number of loss prediction positions. No dataset or epoch reduction
   is used to create an artificial speedup.

## Install without touching the running training

```bash
conda activate attention_prefix_ali
(
  set -euo pipefail
  FORK=/home/n84449292/a84384644/attention_head_forked
  package_dir="$(mktemp -d)"
  unzip -q "$FORK/dspark-training-speed.zip" -d "$package_dir"
  python "$package_dir/dspark-training-speed/install.py" \
    --source-root "$FORK/dspark-bonus20-confidence-train" \
    --destination "$FORK/dspark-training-speed"
)
```

The installer requires the corrected confidence-head source version. It copies
the source's local NPU-occupancy check and dependency locations unchanged. It
does not copy training outputs or create/switch Git branches. An existing
destination is refused.

## Measure on free NPUs

The trials allocate target NPUs 0,1 and trainer NPUs 2,3,4,5,6,7. Busy devices
cause refusal; no job is stopped. Run after the current job has finished or
after you have separately arranged a saved-checkpoint restart.

```bash
conda activate attention_prefix_ali
ROOT=/home/n84449292/a84384644/attention_head_forked/dspark-training-speed
bash "$ROOT/start_benchmark.sh"
```

This launches nine bounded trials: reference, cached loss, batched proposal
base, no backbone recomputation, larger head chunks, smaller block chunks,
DDP, target graphs, and confirmation of the selected settings. Each uses 24
real full-sized training updates from the same seed; the first eight are
excluded from steady timing. Configurations are adapted from faster successful
trials, requiring a 3% improvement to replace the current choice. This is a
bounded search, not an exhaustive search over all combinations.

CPU input/forward/backward phase synchronization is enabled consistently for
the trials. Performance can differ in a long unprofiled run. Startup, graph
capture, hardware checks and loading are recorded in trial wall time but
excluded from steady-step measurements. No full training checkpoint is saved
by these disposable trials; a tiny FSDP preflight checkpoint may be written.

An unsuccessful candidate is recorded with its log. If the reference fails,
no later trial starts. The cached-loss candidate first runs a real-device
gradient check; FSDP candidates also run a tiny distributed backward/save/
restore check. None of these checks has been run on NPUs in this workspace.

Results are under:

```text
dspark-training-speed/output/speed_trials_<timestamp>_<pid>/
  summary.md
  results.json
  selected.json
  selected.env
  01_reference.log
  01_reference/checkpoints/metrics.jsonl
  ...
```

`start_benchmark.sh` prints the exact path. Its job is launched with nohup;
Ctrl+C while viewing its `tail -F` stops the viewer, not the benchmark.

## Target graph evidence

`target_execution.json` records the requested mode. The graph candidate uses
capture sizes 128,256,512,1024,2048,3072; a 3,072-token target batch budget;
32 target sequences; and chunked prefill. This budget is independent of the
9,216-token/rank training packing budget. Consequently the candidate measures
this serving configuration as a whole, not graph execution in isolation.

`target_server.log` includes native vLLM graph metrics. The selector requires
observed PIECEWISE dispatch rows for real multi-token requests; seeing a graph
setting in the command or warmup message alone does not qualify. It records
dispatch evidence, not a profiler-level kernel-replay proof. Existing teacher
checks verify token IDs, tensor shapes and finiteness; they are not a numerical
comparison of eager and graph target hidden states.

Pinned dependencies are retained:
- vLLM `77b1d88c5b52abaf6a103292119f3435b51699d9`
- vLLM-Ascend `276a5c840b21c83b153d3ebd831720bc926dfc4e`

## Start a new full run with the measured settings

After inspecting `summary.md`, set the exact selected.env path printed above:

```bash
conda activate attention_prefix_ali
ROOT=/home/n84449292/a84384644/attention_head_forked/dspark-training-speed
SETTINGS="$ROOT/output/speed_trials_REPLACE_WITH_PRINTED_DIRECTORY/selected.env" \
  bash "$ROOT/start_training.sh"
```

This is explicitly a fresh run, not a resume or migration of the active run.
It uses a fresh `output/dspark_measured_<timestamp>_<pid>` directory. To continue
an existing optimizer checkpoint, the saved recipe/world size and sharding
strategy must be checked separately; do not use this fresh-run entrypoint.

## Validation limits

See VALIDATION.json for exact CPU test results. CPU tests cover FP32 objective
and gradient agreement; bounded BF16 error; two-round gradients; proposal
chain equivalence; checkpoint restore; installer isolation; and benchmark
accounting. CPU fake-process-group FSDP tests exercise the API at world size 1,
not multi-NPU HCCL performance. No NPU speedup, peak-memory fit, SDPA fused-kernel
selection, or teacher graph compatibility is claimed before the device trials.
