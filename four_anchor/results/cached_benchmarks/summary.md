# Cached benchmark results

Acceptance before EOS/budget truncation; throughput includes prefill, excludes loading/warmup. Accuracy not measured.

| Benchmark | Status | Prompts | Acceptance + bonus | Tokens/s | Speedup vs AR | Exact parity |
|---|---|---:|---:|---:|---:|---|
| gsm8k | completed | 128 | 6.292 | 50.360 | — | — |
| math500 | completed | 128 | 6.180 | 48.722 | — | — |
| humaneval | completed | 128 | 4.519 | 36.646 | — | — |
| mbpp | completed | 128 | 3.979 | 32.084 | — | — |
| lcb | completed | 128 | 4.065 | 32.514 | — | — |
| aime25 | completed | 30 | 5.151 | 39.832 | — | — |
| mtbench | completed | 80 | 2.599 | 20.435 | — | — |
| alpaca | completed | 128 | 2.371 | 19.125 | — | — |

Sources, unavailable caches, preparation errors and per-benchmark summaries: `suite_summary.json`.
MT-Bench uses first user turns; a frozen manifest may contain only a cached subset.
No new speedup or parity claim is available unless --with-baseline was used.
