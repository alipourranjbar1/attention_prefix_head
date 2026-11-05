# Four-anchor DFlash: inference from cached accepted-prefix states

Branch: **`four-anchor-accepted-prefix-inference`**.
Base training commit: `01d7dea27438c6c0578215f51b19a7d89c843cea`.
Inference version: `accepted_prefix19_v1`.

This branch adds an inference-only policy for the existing format-3,
`target_four`, 15-proposal checkpoint. Each draft round has **19 query positions**:
**four anchors + fifteen masks**. Native vLLM receives **15 speculative candidates**;
maximum acceptance including one correction/bonus is **16**.

Round one uses the last four processed tokens and their cached target states.
Round two uses first-pass drafter states for generated anchors, and target
states for anchors still in the original prefix. The pending bonus ID is known,
but its target hidden state is not requested. Both draft rounds precede a single
normal target verification. Extra target forwards are measured and must be zero.

**[Exact token alignment, installation, NPU 7 evaluation, results and validation](four_anchor/ACCEPTED_PREFIX.md)**

The patch installs in a **separate sibling worktree**. Training model, losses,
data pipeline, launch scripts, epoch evaluation worker and checkpoints remain
unchanged. The previous training guide is preserved at
[four_anchor/TRAIN_BONUS19.md](four_anchor/TRAIN_BONUS19.md); it describes the
original materialized-bonus inference policy, not this new serving option.
Keep running training and its evaluator from their original checkout.

CPU correctness and isolation checks are included. Ascend execution, acceptance
and speedup must be validated on the server; no new benchmark results are claimed.
