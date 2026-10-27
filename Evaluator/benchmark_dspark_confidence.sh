#!/usr/bin/env bash
# Accuracy with full and cut verification, then matched off/observe/cut timing.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
: "${DRAFT_MODEL:?Set DRAFT_MODEL to your DSpark checkpoint}"
: "${TARGET_MODEL:?Set TARGET_MODEL to the matching Qwen3 target}"
: "${CONDA_PREFIX:?Activate your Ascend environment}"
export DRAFT_MODEL TARGET_MODEL
export ENV_ROOT="${ENV_ROOT:-$CONDA_PREFIX}"
export PYTHON_BIN="${PYTHON_BIN:-$ENV_ROOT/bin/python}"
if [[ -f "$ENV_ROOT/lib/libstdc++.so.6" ]]; then
    export LD_PRELOAD="$ENV_ROOT/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"
fi
export NPU="${NPU:-6}" PORT="${PORT:-8217}"
export TARGET_EXECUTION="${TARGET_EXECUTION:-eager}"
CONCURRENCY="${CONCURRENCY:-1}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-$CONCURRENCY}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-6400}" MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-6400}"
export PYTHONUNBUFFERED=1
ACTION="${ACTION:-all}"
case "$ACTION" in accuracy|speed|all) ;; *) echo 'ACTION must be accuracy, speed or all'; exit 2 ;; esac
EXPERIMENT="${EXPERIMENT:-$ROOT/output/dspark_confidence_compare_$(date +%Y%m%d_%H%M%S)_$$}"
mkdir -p "$(dirname -- "$EXPERIMENT")"
mkdir "$EXPERIMENT"
read -r -a datasets <<< "${DATASETS:-gsm8k}"
MANIFEST="${MANIFEST:-$EXPERIMENT/prompts.json}"
common=(--target "$TARGET_MODEL" --datasets "${datasets[@]}"
    --num-prompts "${NUM_PROMPTS:-128}" --manifest "$MANIFEST" --no-thinking
    --temperatures 0 --concurrencies "$CONCURRENCY"
    --max-new-tokens "${MAX_NEW_TOKENS:-2048}" --seed "${SEED:-42}")
"$PYTHON_BIN" "$ROOT/Evaluator/evaluator.py" "${common[@]}" --offline --prepare-only
server_pid=''
cleanup() {
    if [[ -n "$server_pid" ]]; then
        kill -TERM -- "-$server_pid" 2>/dev/null || true
        wait "$server_pid" 2>/dev/null || true
        server_pid=''
    fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
run_phase() {
    local phase="$1" mode="$2" trace="$3"
    export RUN_DIR="$EXPERIMENT/server_$phase" DSPARK_CONFIDENCE_MODE="$mode"
    if [[ "$trace" == 1 ]]; then
        export DSPARK_CONFIDENCE_TRACE_DIR="$EXPERIMENT/traces_$phase"
    else
        unset DSPARK_CONFIDENCE_TRACE_DIR
    fi
    # Never reuse a server belonging to another job.
    "$PYTHON_BIN" -c 'import socket,sys; s=socket.socket(); s.bind(("127.0.0.1",int(sys.argv[1]))); s.close()' "$PORT"
    echo "Starting $phase: NPU=$NPU, mode=$mode, concurrency=$CONCURRENCY"
    setsid bash "$ROOT/Evaluator/run_server_dspark_confidence.sh" > "$EXPERIMENT/$phase.launcher.log" 2>&1 &
    server_pid=$!
    local deadline=$((SECONDS + ${STARTUP_TIMEOUT:-1800}))
    while [[ ! -f "$RUN_DIR/launch.json" ]] || ! curl --noproxy '*' --max-time 3 -fsS "http://127.0.0.1:$PORT/v1/models" >/dev/null; do
        if ! kill -0 "$server_pid" 2>/dev/null || (( SECONDS >= deadline )); then
            tail -n 80 "$EXPERIMENT/$phase.launcher.log"
            return 1
        fi
        sleep 2
    done
    "$PYTHON_BIN" "$ROOT/Evaluator/evaluator.py" "${common[@]}" \
        --base-url "http://127.0.0.1:$PORT" --server-manifest "$RUN_DIR/launch.json" \
        --method "dspark_$phase" --output "$EXPERIMENT/eval_$phase" \
        2>&1 | tee "$EXPERIMENT/$phase.eval.log"
    cleanup
}
if [[ "$ACTION" == accuracy || "$ACTION" == all ]]; then
    run_phase accuracy observe 1
    "$PYTHON_BIN" "$ROOT/Evaluator/dspark_confidence_report.py" accuracy \
        --trace-dir "$EXPERIMENT/traces_accuracy" --run-info "$EXPERIMENT/eval_accuracy/run_info.json" \
        --output "$EXPERIMENT/confidence_accuracy.json"
    run_phase accuracy_cut cut 1
    "$PYTHON_BIN" "$ROOT/Evaluator/dspark_confidence_report.py" accuracy \
        --trace-dir "$EXPERIMENT/traces_accuracy_cut" --run-info "$EXPERIMENT/eval_accuracy_cut/run_info.json" \
        --output "$EXPERIMENT/confidence_accuracy_after_cut.json"
fi
if [[ "$ACTION" == speed || "$ACTION" == all ]]; then
    run_phase off off 0
    run_phase observe observe 0
    run_phase cut cut 0
    "$PYTHON_BIN" "$ROOT/Evaluator/dspark_confidence_report.py" compare \
        --baseline "$EXPERIMENT/eval_off/run_info.json" \
        --observe "$EXPERIMENT/eval_observe/run_info.json" \
        --cut "$EXPERIMENT/eval_cut/run_info.json" \
        --output "$EXPERIMENT/speed_comparison.json"
fi
printf 'Results: %s\n' "$EXPERIMENT"
