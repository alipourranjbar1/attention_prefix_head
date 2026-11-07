# Four-anchor DFlash: bonus20 inference with concurrency 32

Branch: **`four-anchor-bonus20-c32-inference`**.

Inference-only extension for the existing `target_four` / 15-proposal checkpoint.
Training code and weights are unchanged. Install into a separate sibling checkout.

| Stage | Layout | Anchor hidden states |
|---|---|---|
| First draft | 4 past anchors + known bonus embedding + 15 masks = **20 queries** | Four target states at the past-anchor positions |
| Second draft | 4 preceding anchors + 15 masks = **19 queries** | All four from the first drafter pass |
| Target verification | Known bonus + 15 speculative candidates = **16 inputs** | One normal target forward; zero extra bonus-materialization calls |

Maximum acceptance including one correction/bonus is **16**. Conditioning
anchors are excluded. The bonus has no fabricated target hidden state.

[Full structure, installation, NPU 7 commands and result paths](four_anchor/BONUS_BATCH.md)

The launcher queues cached prompts in one vLLM engine with up to 32 active
requests on NPU 7, with no baseline run by default. It
saves a Markdown/CSV benchmark table in the new checkout. Existing checkpoint,
prompt plan, training output and epoch evaluation queue are read-only inputs.

This changes the inference input relative to training; performance must be
measured. CPU numerical and integration tests do not establish Ascend parity
or throughput. See the guide for validation limits and an optional parity run.

[Existing training guide](four_anchor/TRAIN_BONUS19.md)
