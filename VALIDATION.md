# Validation

The model, objective, original HF target engine, optimizer and scheduler are
byte-identical to NAVER DLoop commit
`775dedd71aa95e1d98e6780efe5c444a2a27575c`.

Executed CPU tests use PyTorch 2.10.0, Transformers 5.0.0 and tiny Qwen3 models.
Latest suite: **46 passed**, with 24 existing scheduler warnings, in 8.73 seconds.
Checks cover:

- Pinned source identity and the restricted list of port changes.
- Native weighted CE and parameter gradients at k=1,2,3, including position
  offsets and differentiable carriers across stages.
- Optional chunked/checkpointed CE versus the native dense implementation:
  FP32 and BF16 values, metrics and hidden gradients; end-of-sequence and
  masked labels; zero-weight supervision; full drafter/carrier gradients at
  k=1,2,3 with backbone recomputation; and saved-tensor inspection confirming
  the loss does not retain vocabulary-sized logits/log-softmax buffers.
- The upstream requirement to leave room for the original block and a loop.
- Saved-token IDs/loss masks and the unchanged upstream short-row replacement.
- Packed sampler coverage across eight simulated ranks, equal collective step
  counts, per-rank padded-token budgets, epoch reshuffling, cache fingerprints,
  short-row replacement parity and deterministic mid-epoch batch replay.
- Batch-wide anchor quotas, actual per-document end boundaries, bounded padded
  block work, and identical loop depths despite differing rank-local RNG use.
- Native DLoop forward on a right-padded batch versus separate unpadded
  documents: losses and draft parameter gradients at depths 1/2/3, including
  short-document boundaries and inactive block slots.
- HF Qwen3 selected target states before/after skipping its unused vocabulary
  head; independence from other documents and from padded token values.
- Rejection of custom four-anchor initialization and absent initial weights.
- Explicit config-only scratch mode, rejection of ambiguous/stale weight
  paths, and reproducible random matrices with different values for different
  seeds. The actual native model builder is tested with pretrained-draft loading
  forbidden in scratch mode.
- Occupied-device refusal, RNG restoration and incomplete checkpoint detection.
- The actual upstream training main loop, HF target, BF16 master optimizer,
  model export and save/resume on a tiny CPU example. FSDP/distributed
  collectives are stubbed in this test; uninterrupted and resumed final
  weights match exactly for both pretrained and scratch initialization, with
  and without chunked CE plus activation checkpointing, and with packing on/off.
- Shell launcher execution with fake environment scripts and Python probes:
  both CANN directory layouts, environment setup before imports, native flags,
  eight training ranks, and exclusion of inherited project PYTHONPATH/config.
  Both initialization modes are covered, including the requested 9216/768
  scratch launcher arguments and absence of --draft-init-path in that mode.
  Packed cases verify that planning runs before model launch, and that the
  index, 9,216-token budget and shared 768-start budget reach the trainer.

Shell syntax, Python compilation and source-diff boundaries are also checked.
Full output is in `tests/validation.log`.

Not established here: actual Ascend execution, HCCL collectives, full-model
memory use, training throughput, full-scale resume parity or inference parity.
No NPU training job was run in the authoring workspace. The launcher performs
dependency/input/device checks and supports MAX_STEPS=2 for a hardware check.
A local Gloo initialization probe was blocked by the workspace's socket
permissions. No real multi-process FSDP or HCCL test is claimed; distributed
sampler coverage is simulated and the trainer lifecycle test stubs FSDP.

Run CPU tests in an existing compatible development environment:

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -m pytest -q tests
```

`ASCEND_PORT.patch` lists changes to the original upstream tree. The additional
`ascend_support/` files implement only local-file loading, launch/provenance,
device checks, explicit initialization selection and rank-specific checkpoint I/O. They contain no new drafting
objective or suffix-redrafting policy.

Packed mode additionally binds a batch-wide sampler and uniform step-keyed
loop-depth RNG to the native wrapper, and uses the same Qwen3 backbone without
its unused vocabulary projection. Those are explicitly disclosed runtime and
batching adaptations; unchanged source hashes alone do not imply an identical
training recipe. See PACKING.md for quotas, padding, weighting and tail policy.

The optional `chunked_ce.py` adapter changes how the existing loss is executed,
not its mathematical objective. Float reduction ordering can differ from the
dense reference. These are CPU tests; NPU/FSDP memory behavior remains to be
checked on the user's hardware. Activations are recomputed to save memory;
faster training is not claimed.
