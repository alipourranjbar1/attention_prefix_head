#!/usr/bin/env bash
# Run isolated trials only. Does not start a five-epoch run or stop other jobs.
set -euo pipefail
: "${CONDA_PREFIX:?Activate attention_prefix_ali first}"
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${RESUME:-}" ]]; then
    echo 'Unset RESUME before isolated speed trials.' >&2
    exit 1
fi
export CANN_ROOT="${CANN_ROOT:-/home/n84449292/m84379596/CANN/9.1.0}"
export MODEL="${MODEL:-/home/n84449292/m84379596/Huggingface/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c}"
export DATASET="${DATASET:-/home/n84449292/m84379596/Huggingface/datasets/open_perfectblend.qwen3-4b-rollout.qwen3.seq3072}"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
BENCH_OUT="${BENCH_OUT:-$ROOT/output/speed_trials_$(date +%Y%m%d_%H%M%S)_$$}"
mkdir -p "$(dirname -- "$BENCH_OUT")"
LOG="${BENCH_OUT}.log"
nohup python -u -m four_anchor.speed_benchmark --output "$BENCH_OUT" --steps 24 --warmup 8 \
    >"$LOG" 2>&1 < /dev/null &
printf 'Benchmark PID: %s\nLog: %s\nResults: %s/summary.md\n' "$!" "$LOG" "$BENCH_OUT"
tail -F "$LOG"
