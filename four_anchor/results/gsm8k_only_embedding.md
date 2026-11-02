# Four-anchor DFlash: bonus embedding-only results

Branch: `four-anchor-only-embedding`

## Results

128 GSM8K prompts, concurrency 1, greedy decoding.
Target graphs and drafter graphs were off.
Throughput includes prefill and excludes model loading and warmup.

| Method | Output tokens | Time (seconds) | Tokens/second | Acceptance including bonus | Throughput speedup |
|---|---:|---:|---:|---:|---:|
| Qwen3-4B baseline | 38,533 | 3112.851 | 12.379 | N/A | 1.00x |
| Four-anchor two-pass, bonus embedding-only | 38,007 | 757.570 | 50.170 | 6.292 | **4.05x** |

Throughput speedup = 50.16960594660738 / 12.378684217201009
= 4.052902963377429.

The speculative run performed 6,020 verification cycles and accepted
5.2921926910299 draft tokens per cycle, plus one correction/bonus.
The four conditioning anchors are excluded from acceptance length.
Acceptance is counted before EOS/output-budget truncation.

Output agreement was not checked. Output lengths differ.
GSM8K answer accuracy was not measured.

## Preserved inference structure

- Inference version: `known_bonus_cached_v1`.
- Checkpoint: `checkpoint-016699`.
- Target: Qwen3-4B using vLLM-Ascend.
- One shared drafter with five bidirectional regular DFlash layers.
- First pass: four past anchors with their target hidden states,
  one known bonus embedding, and fifteen masks.
- No target hidden-state feature is added to the bonus in the first pass.
- Second pass: four preceding anchors; generated positions use projected
  first-pass draft features, including the bonus when it is an anchor.
- Exactly fifteen speculative candidates go to verification.
- Maximum acceptance including one correction/bonus: sixteen.
- Incremental context projection/KV caching enabled.
- Extra target forwards for bonus hidden states: zero.
- Maximum new tokens per prompt: 2048.
- Training layout: four anchors plus sixteen masks.

This branch preserves the embedding-only experiment before changing
bonus hidden-state conditioning.
