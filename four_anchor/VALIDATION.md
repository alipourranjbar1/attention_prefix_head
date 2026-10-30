# Validation

Validated on CPU with Python 3.12, PyTorch 2.10.0+cpu, Transformers 5.0.0,
and the repository's pinned regular DFlash decoder implementation.

- **56 tests passed.** These cover the D-Loop confidence score, four-anchor
  token/state/position selection, padding, 20 query positions / 16 outputs,
  training gradients, frozen target weights, upstream regular-DFlash warm starts,
  checkpoint validation, cache rollback at multiple rejection positions, full
  acceptance, EOS and output limits, and HTTP endpoints.
- A nonzero-boundary test checks that retaining five proposals yields a 21-token
  final candidate and moves the second pass's anchors to the correct positions.
- Greedy output IDs match both the reference autoregressive loop and Hugging
  Face `target.generate` on the small Qwen3 test model.
- The offline smoke pipeline prepares teacher data, trains two optimizer steps,
  loads the checkpoint, resumes from step one, and obtains **bitwise-identical
  final weights** versus uninterrupted training. Evaluation and HTTP completion
  also pass.
- Float32 and bfloat16 CPU training and evaluation were exercised. Both
  speculative modes matched target-only outputs on the smoke evaluation prompts.
- The actual `scripts/pipeline.sh` entry point completed training, evaluation,
  and sample generation using prepared tiny-model data.
- Python compilation, shell syntax, and patch whitespace checks passed.

**Not validated:** full Qwen3-4B training, acceptance improvements, accelerator
speedup, CUDA/NPU kernels, and multiworker distributed execution. A two-worker
CPU run was attempted, but Gloo initialization failed with `Operation not
permitted` under the environment's network restrictions. That result is not a
distributed-training pass.

The tiny target is randomly initialized. Its two-step training and timing
numbers test execution and correctness; they are not useful quality or
performance benchmarks. No trained Qwen3-4B weights are included.
