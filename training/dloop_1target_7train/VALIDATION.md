# Validation performed

Prepared on 2026-11-13 with Python 3.12, PyTorch 2.10.0+cpu,
Transformers 5.0.0 and Datasets 4.4.1. No Ascend device or server connection was
available in this environment.

21 CPU test cases passed: the 19-test suite plus two native DLoop decoding
checks added afterward. Shell scripts passed `bash -n`; the new Python modules
passed syntax parsing.

- Exact block-attention comparison against native dense SDPA, including
  output and Q/K/V gradients, invalid blocks, grouped-query heads, two chunk
  sizes, and carrier counts 0, 14, 29 and 44.
- Full native DLoop loss and trainable-parameter gradient comparison at each
  sampled loop depth 1, 2 and 3.
- Optimized native DLoop training with activation checkpointing and chunked CE:
  uninterrupted versus checkpoint/resume weights matched bit-for-bit on a tiny
  CPU lifecycle test. Distributed collectives and FSDP were stubbed for this
  test; this does not validate HCCL or real FSDP execution.
- Own bonus-add-on 20/19 cached training: uninterrupted versus checkpoint/resume
  weights matched bit-for-bit in a tiny single-rank CPU test.
- Own cached 20/19 decoder matched greedy target output on a tiny Qwen3 model.
- Native DLoop decoder matched greedy target output at loop gate -0.5 and at
  a test-only permissive gate forcing all three configured loop stages.
- Both Speculators-format and HF-format regular DFlash checkpoint conversion
  preserved every common draft tensor exactly and loaded successfully into
  both model implementations. Own bridge projection remained identity.
- Common record filtering preserved the original source dataset; row-based
  quarter planning, literal 0.25% planning, answer scoring, and full-run schedule
  arguments were checked.

The smoke test exposed a source-directory assumption in the native own-method
provenance recorder. The package sets `SPECULATORS_ROOT` to the bundled source
directory; no method or model source was edited to work around it.

The real user's checkpoint, dataset, GSM8K cache, NPU memory consumption, NPU
SDPA kernels, four-rank HCCL/FSDP execution, acceptance results and end-to-end
speedup remain unmeasured. The supplied launchers perform server-side checks
and record the actual outcomes. No speedup multiplier is claimed.

## Local GSM8K loader update

32 targeted CPU tests passed after the offline-input update (13 new local-input
cases plus the 19 relevant checkpoint, schedule, attention and decoding cases).
The two previously passed trainer lifecycle tests were not repeated because the
training implementation did not change.

The new tests verify saved-token identity and source-file order, no Hub/tokenizer
lookup for saved IDs, text and chat input formats, no reference-answer leakage,
invalid/padded/training-token rejection, optional reference-answer parsing, and
unavailable/partial/full accuracy aggregation. Acceptance aggregation remains
available when all reference answers are missing. The supplied server JSONL
has not been inspected from this workspace; its schema is validated during
local preparation on the server.

## Full-attention warm-start update

The user supplied the source settings: five `sliding_attention` draft layers,
`use_sliding_window=true`, window 2048, dropout 0.0, and explicitly requested
full attention for training. The converter now writes full-attention views
for both methods and records the deliberate connectivity change.

39 targeted CPU test cases passed after this change. Conversion tests cover
both HF and Speculators checkpoint formats, with full and five-layer sliding
source configurations. They verify exact tensor equality, unchanged source
config and weight bytes, both exported full-attention configs, and absence of
sliding windows in every loaded draft attention layer. Retry tests preserve
saved GSM8K prompts, reject conflicting prompts, and refuse directories that
already contain initialization/data/training outputs.

The user's real NPU execution and real checkpoint tensor conversion remain
server-side checks; the local conversion tests use small synthetic checkpoints.

## Matched budgets and one additional epoch

The corrected requested budget is 512 starts per 3,072-token unit, hence 1,536
per 9,216-token rank batch. Both methods now use the exact same document batch
sampler and retain their own model/objective. The prepared-run updater retains
one additional epoch from the shared base checkpoint, aligns the own-method
warmup and schedule to DLoop's existing schedule, preserves frozen inputs, and
refuses to alter running or started experiments.

17 targeted CPU checks passed for this update. They cover both checkpoint
formats and source attention modes; dataset filtering; launcher anchor/schedule
arguments; actual rank-batch equality and evaluation-boundary equality across
two epochs; installation of the shared sampler into both own-method scheduling
and training; metadata migration with preserved prompts and backup; idempotent
retry; and refusal to modify active or started jobs. No real NPU timing is
claimed. The existing method and decoder numerical checks were not rerun
because their implementations did not change.

## DLoop first: two target cards and six training cards

The new `prepare_dloop_8card.sh` / `run_dloop_8card.sh` mode uses 256 training
starts per 3,072-token budget unit (768 per 9,216-token packed rank batch), six
FSDP ranks, two separate HF target replicas, and one additional epoch. It reuses
the prepared eligible data and frozen prompts and rebuilds the document-based
quarter-epoch schedule for six ranks. Original model/loss/optimizer code is
unchanged; the remote target is installed through an adapter at entry.

21 targeted CPU checks passed after this change, including:

- Actual HTTP transfers through two serialized target services with six client
  rank assignments; exact float32 and BF16 hidden-state equality against the
  original local HF engine, padded documents, and nonzero RoPE offsets.
- Final hidden-state parity, response/file identity checks, transient-file
  cleanup, authorization and malformed-input rejection.
- Actual tiny DLoop training, CE, backward, activation checkpointing, optimized
  attention, checkpoint save, and resume with the remote target; final weights
  exactly match an uninterrupted run. FSDP/collectives are stubbed in this CPU
  lifecycle test, as in the existing native-trainer tests.
- Six-rank document coverage, budgets, quarter schedules, fresh preparation
  without changing original inputs, and idempotent preparation.
- Orchestration checks for target/trainer placement, six-rank training and
  evaluation, optimizer/RNG shard validation, phase continuation and skipping.
- Existing own/DLoop CPU lifecycle, matched budgets, and packing tests.

Real Ascend execution, six-rank HCCL/FSDP, NPU numerical behavior, memory usage,
and throughput have not been run in this workspace. The server launcher checks
both live target services before starting training. No real acceptance result
or acceleration measurement is claimed.

## One target card and seven training cards

The current launcher uses one target server on physical NPU 0 and seven FSDP
training ranks on NPUs 1–7. Preparation selects a separate default run directory,
reuses the original prepared data/prompts/base initialization, and recalculates
the document-based evaluation schedule for seven ranks. The budget remains 768
training starts and 9,216 padded tokens per rank/update, with one additional epoch.

16 targeted CPU tests passed for this update. They cover seven client ranks
using one real HTTP target service, exact float32/BF16 local-target parity with
RoPE offsets and document padding, seven-rank packing and quarter schedules,
idempotent preparation without changing source data, seven-rank training/eval
commands and checkpoint shard checks, explicit rejection of six-rank optimizer
checkpoints, and a mocked startup proving only NPU 0 receives a target process.
The real tiny-model remote-target CE/backward/save/resume lifecycle also passed;
distributed FSDP remains stubbed in that CPU test. Shell syntax checks passed.

The previous two-target/six-trainer job is not migrated or overwritten. The new
run starts from its original prepared base weights and new optimizer state.
Seven-rank Ascend/HCCL execution and end-to-end speed remain server-side checks.
