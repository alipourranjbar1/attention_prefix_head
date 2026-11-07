# Four-anchor DFlash: bonus-aware two-pass training

Branch: **`four-anchor-bonus20-train`**.

Train the layout used by the bonus20 batched inference pipeline:

- **Round one:** four past token IDs and target hidden states, one known bonus
  token embedding, and 15 masked speculative positions: **20 queries**.
- **Round two:** four preceding token IDs and their first-pass drafter hidden
  states, then 15 masks: **19 queries**.
- **Verification:** bonus plus 15 proposals, one native target forward.

Regular DFlash, five bidirectional layers, shared drafter, FSDP, and CE/TV loss.
Completed checkpoints are evaluated on cached benchmarks on NPU 7, with
concurrency 32; per-epoch result tables are maintained automatically.

Read [training, installation, results and constraints](four_anchor/TRAIN_BONUS20.md).
Start with `bash four_anchor/scripts/train_bonus20_ascend.sh` after setting
`MODEL`, `DATASET` and the Ascend environment for this server.

This branch saves new **format-4 `bonus_addon`** checkpoints. The original
format-3 training branch remains separate. The repository also contains legacy
launchers for historical experiments; the command above selects this objective.
