# Validation of packed training and DSpark loss

Validated on CPU with Python 3.12, PyTorch 2.10.0+cpu, Transformers 5.0.0,
and the pinned regular DFlash decoder implementation. **109 tests passed.**

The new packed tests verify:

- Batched block outputs and gradients agree with independent single-block forwards,
  with and without layer checkpointing.
- Attention cannot read another document or target states at/after the original
  prediction boundary, including after the second-pass anchors move forward.
- Four proposal anchors use detached first-pass states and train the state bridge.
- CE, TV and optional KL values/gradients agree with the upstream eager losses.
  The default weighted objective is 0.1 CE + 0.9 TV, not CE + KL.
- The frozen final decoder output, final norm and LM head reconstruct the target
  logits with the correct next-token alignment. Remote payload tests exercise
  the additional final-state slot, prompt alignment, file cleanup and one prefill.
- The actual upstream packing sampler produces equal batch counts across simulated
  ranks, respects the per-rank token budget and does not repeat packed rows.
- Actual two-epoch tiny-model packed training, partial gradient accumulation,
  mid-epoch checkpoint resume and subsequent exact greedy decoding work. Resume
  produces bitwise-identical final model weights in both FP32 and BF16 CPU runs.
- A nonzero second-pass training boundary exercises the real redraft computation,
  obtains bridge gradients and remains unchanged when unavailable input teacher
  states are perturbed.

The retained suite also covers D-Loop confidence selection, 20 query positions /
16 generation slots, source/position selection, upstream DFlash warm starts,
legacy training, data filtering/cache reuse, progress and TensorBoard, exact
cache rollback, EOS, output limits, evaluation and HTTP endpoints.

Python compilation, shell syntax and patch whitespace checks passed.

**Not validated on this hardware:** Qwen3-4B training, full Ascend execution,
HCCL/DDP execution, acceptance improvement or a 10–12 hour runtime. Distributed
sampler tests simulate ranks; they do not launch HCCL. An earlier Gloo launch
was blocked by this environment's network permissions. The tiny random target
is a correctness fixture, not a useful quality or throughput benchmark.

The new training objective reuses teacher-forced distributions and caps the
training retained prefix at a dataset-token mismatch. This is explicitly
documented in README.md; inference remains confidence-based. Old online
optimizer checkpoints cannot resume the packed schedule. No trained Qwen3-4B
weights are included.

## FSDP update

The Ascend defaults are now FULL_SHARD, 9216 packed tokens and up to 1536 training
blocks **per training NPU**. Four conditioning anchors and sixteen proposals per
pass are unchanged. Frozen teacher modules remain replicated.

Additional CPU checks exercise actual FSDP module hooks, full state dictionary
export, optimizer-state conversion, FP32/BF16 resumed continuation, and rejection
of invalid single-rank production launches. These use single-rank NO_SHARD with
an in-process fake collective backend; they do not validate multi-rank sharding.
A Gloo attempt on this host failed with `Operation not permitted` during socket
creation. No real distributed execution pass is claimed.

`four_anchor.fsdp_check` is automatically run with the real training world and
HCCL before loading Qwen3-4B on the server. It intentionally gives ranks different
numbers of chunks and checks FULL_SHARD storage, finite gradients, collective
checkpointing, and equivalent continuation after restore. Its result and logs
are saved under the run directory. Actual NPU compatibility and throughput
remain hardware checks, not results established by the CPU suite.

## Cached execution and timing update

The optimized path is compared with the reference for CE/TV/KL values and hidden
gradients in FP32/BF16, including the minimum-function tie case for TV. Tests
compare cached context forward outputs and gradients of all backbone parameters,
check exactly one full-context projection, and verify that a mixed group only
redrafts positive-boundary blocks while retaining both losses for every block.
Future/cross-document isolation, full tiny-model training and exact resume are
exercised under both reference and cached execution. The single-rank FSDP
checkpoint tests now use cached execution too.

These are CPU correctness tests. Neither the reported approximately 12-second
NPU step time nor a 10-hour target has been benchmarked on the updated code here.
The launcher records real synchronized phase timings for the first five server
training updates; this is necessary to establish hardware throughput.
