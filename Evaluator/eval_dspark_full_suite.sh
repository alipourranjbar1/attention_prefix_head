#!/usr/bin/env bash
# Client only. Run all full datasets and resume remaining repeats for this mode.
set -euo pipefail
ROOT="${ROOT:-/home/n84449292/a84384644/attention_head_forked/attention_prefix_head}"
: "${CONDA_PREFIX:?Activate attention_prefix_ali first}"
CONCURRENCY="${CONCURRENCY:-64}"
[[ "$CONCURRENCY" =~ ^[1-9][0-9]*$ ]] || { echo 'CONCURRENCY must be a positive integer'; exit 2; }
SUITE_DIR="${SUITE_DIR:-$ROOT/output/dspark_full_benchmark_c${CONCURRENCY}}"
PORT="${PORT:-8217}"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export NO_PROXY=localhost,127.0.0.1,::1 no_proxy=localhost,127.0.0.1,::1
if [[ -r "$CONDA_PREFIX/lib/libstdc++.so.6" ]]; then
    export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"
fi
if ! curl --noproxy '*' --max-time 5 -fsS "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
    echo 'Server is not ready. Check the first terminal and rerun after startup completes.'
    exit 1
fi
"$CONDA_PREFIX/bin/python" "$ROOT/Evaluator/dspark_benchmark_suite.py" evaluate \
    --suite "$SUITE_DIR" --base-url "http://127.0.0.1:$PORT" \
    --server-manifest "$ROOT/output/evaluator_servers/port_$PORT.json" \
    --concurrency "$CONCURRENCY"
