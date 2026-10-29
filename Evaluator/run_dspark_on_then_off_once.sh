#!/usr/bin/env bash
# One full confidence-CUT run, then one full confidence-OFF run, on one NPU.
set -euo pipefail
set +m

export ROOT="${ROOT:-/home/n84449292/a84384644/attention_head_forked/attention_prefix_head}"
RESULTS_ROOT="${RESULTS_ROOT:-/home/n84449292/a84384644/attention_head_forked/results}"
: "${CONDA_PREFIX:?Activate attention_prefix_ali first}"
export PYTHON_BIN="$CONDA_PREFIX/bin/python"
export CONCURRENCY=128 REPEATS=1 NUM_SPECULATIVE_TOKENS=7 SEED=42 PYTHONHASHSEED=42
export NPU="${NPU:-7}" PORT="${PORT:-8217}" MAX_NEW_TOKENS=2048
export CHECK_ONLY="${CHECK_ONLY:-0}"
export CANN_ROOT="${CANN_ROOT:-/home/n84449292/a84384644/CANN/9.1.0}"
export TARGET_MODEL="${TARGET_MODEL:-/home/n84449292/a84384644/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c}"
export DRAFT_MODEL="${DRAFT_MODEL:-/share/canada_group_folder/ckpt/dspark_qwen3_4b_block7}"
export VLLM_BATCH_INVARIANT=1 HCCL_DETERMINISTIC=strict LCCL_DETERMINISTIC=1
export ATB_MATMUL_SHUFFLE_K_ENABLE=0 ATB_LLM_LCOC_ENABLE=0
# The same policy settings are recorded in both modes. OFF never invokes it.
export DSPARK_CONFIDENCE_THRESHOLD=0.3 DSPARK_CONFIDENCE_INITIAL_BUDGET=5 DSPARK_CONFIDENCE_UPDATE_INTERVAL=16
unset DSPARK_CONFIDENCE_TRACE_DIR
export NO_PROXY=localhost,127.0.0.1,::1 no_proxy=localhost,127.0.0.1,::1
export PYTHONUNBUFFERED=1
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-1800}"
[[ "$NPU" =~ ^[0-7]$ && "$PORT" =~ ^[0-9]+$ && "$STARTUP_TIMEOUT" =~ ^[1-9][0-9]*$ ]] || {
    echo 'Invalid NPU, PORT or STARTUP_TIMEOUT'; exit 2;
}
(( PORT > 0 && PORT < 65536 )) || { echo 'PORT must be 1..65535'; exit 2; }
[[ "$CHECK_ONLY" == 0 || "$CHECK_ONLY" == 1 ]] || { echo 'CHECK_ONLY must be 0 or 1'; exit 2; }
[[ -x "$PYTHON_BIN" ]] || { echo "Missing Python: $PYTHON_BIN"; exit 1; }
for dependency in setsid flock curl ps awk; do
    command -v "$dependency" >/dev/null || { echo "Missing command: $dependency"; exit 1; }
done
for source in serve_dspark_full_suite.sh eval_dspark_full_suite.sh dspark_benchmark_suite.py dspark_once_control.py; do
    [[ -f "$ROOT/Evaluator/$source" ]] || { echo "Missing $ROOT/Evaluator/$source; extract the complete update ZIP."; exit 1; }
done
if [[ -r "$CONDA_PREFIX/lib/libstdc++.so.6" ]]; then
    export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"
fi

mkdir -p "$RESULTS_ROOT"
exec 9>"$RESULTS_ROOT/.dspark_npu${NPU}_port${PORT}.lock"
flock -n 9 || { echo 'Another ON/OFF run owns this NPU/port lock.'; exit 1; }
export SUITE_DIR="$RESULTS_ROOT/dspark_qwen3_4b_c128_on_then_off_$(date +%Y%m%d_%H%M%S)_$$"
mkdir "$SUITE_DIR"
mkdir "$SUITE_DIR/logs"
cp -- "${BASH_SOURCE[0]}" "$SUITE_DIR/run_script.sh"
printf '%s\n' "$SUITE_DIR" > "$RESULTS_ROOT/latest_on_then_off.txt"
exec > >(tee "$SUITE_DIR/run.log") 2>&1

SERVER_PID=""
SERVER_LOG=""
group_alive() {
    [[ -n "$SERVER_PID" ]] && ps -eo pgid=,stat= | awk -v group="$SERVER_PID" \
      '$1 == group && $2 !~ /^Z/ {found=1} END {exit !found}'
}
stop_server() {
    [[ -n "$SERVER_PID" ]] || return 0
    echo "Stopping this script's evaluation server (process group $SERVER_PID)."
    # Only the process group created below with setsid is signalled.
    kill -TERM -- "-$SERVER_PID" 2>/dev/null || true
    for ((attempt=0; attempt<60; attempt++)); do
        group_alive || break
        sleep 1
    done
    if group_alive; then
        echo 'Evaluation worker cleanup exceeded 60 seconds; stopping remaining owned workers.'
        kill -KILL -- "-$SERVER_PID" 2>/dev/null || true
    fi
    wait "$SERVER_PID" 2>/dev/null || true
    for ((attempt=0; attempt<10; attempt++)); do
        group_alive || break
        sleep 1
    done
    if group_alive; then
        echo "Owned process group $SERVER_PID did not stop. Results are preserved."
        return 1
    fi
    SERVER_PID=""
}
cleanup() {
    status=$?
    trap - EXIT INT TERM
    stop_server || status=1
    if (( status != 0 )); then
        echo "Run stopped with status $status. Saved files: $SUITE_DIR"
        [[ -z "$SERVER_LOG" || ! -f "$SERVER_LOG" ]] || tail -n 60 "$SERVER_LOG"
    fi
    exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

echo "Results: $SUITE_DIR"
echo 'Order: ON (cut), then OFF. One pass per mode. Concurrency=128, proposals=7, seed=42.'
echo 'Datasets: GSM8K, MATH-500, HumanEval, MBPP, LiveCodeBench, AIME25, MT-Bench.'
echo 'Both modes request batch-invariant kernels; equality is checked after generation.'

if [[ "$CHECK_ONLY" == 0 ]]; then
    "$PYTHON_BIN" "$ROOT/Evaluator/dspark_once_control.py" port-free --port "$PORT"
    # Check installed runtime/operator support in a separate process. No packages
    # are installed and no existing model or training process is touched.
    (
        set +u
        source "$CANN_ROOT/ascend-toolkit/set_env.sh"
        source "$CANN_ROOT/nnal/atb/set_env.sh"
        set -u
        # CANN/ATB setup may replace the values inherited from this runner.
        export VLLM_BATCH_INVARIANT=1 HCCL_DETERMINISTIC=strict LCCL_DETERMINISTIC=1
        export ATB_MATMUL_SHUFFLE_K_ENABLE=0 ATB_LLM_LCOC_ENABLE=0 PYTHONHASHSEED=42
        export ASCEND_RT_VISIBLE_DEVICES="$NPU"
        export PYTHONPATH="$ROOT/vllm:$ROOT/vllm-ascend:$ROOT/speculators/src:$ROOT/speculators/hs_connectors/src"
        unset TORCH_DEVICE_BACKEND_AUTOLOAD
        "$PYTHON_BIN" "$ROOT/Evaluator/dspark_once_control.py" preflight --root "$ROOT" \
          --output "$SUITE_DIR/runtime_preflight.json"
    ) 2>&1 | tee "$SUITE_DIR/logs/preflight.log"
fi

# Preserve the earlier full prompt file byte-for-byte when it is available.
PROMPTS_SOURCE="${PROMPTS_SOURCE:-$ROOT/output/dspark_full_benchmark_c128/prompts.json}"
if [[ -f "$PROMPTS_SOURCE" ]]; then
    cp -- "$PROMPTS_SOURCE" "$SUITE_DIR/prompts.json"
    echo "Frozen prompts copied from: $PROMPTS_SOURCE"
else
    export REUSE_MANIFEST="${REUSE_MANIFEST:-$ROOT/Evaluator/retrace_eval_prompts.json}"
    echo "Preparing complete saved splits from: $REUSE_MANIFEST"
fi

snapshot() {
    {
        date -Is
        [[ ! -r /proc/loadavg ]] || cat /proc/loadavg
        if command -v npu-smi >/dev/null && command -v timeout >/dev/null; then
            timeout 15s npu-smi info || true
        fi
    } > "$SUITE_DIR/logs/$1.hardware.txt" 2>&1
}

for confidence_mode in cut off; do
    SERVER_LOG="$SUITE_DIR/logs/server_${confidence_mode}.console.log"
    if [[ "$CHECK_ONLY" == 1 ]]; then
        bash "$ROOT/Evaluator/serve_dspark_full_suite.sh" "$confidence_mode" > "$SERVER_LOG" 2>&1
        echo "Validated launcher configuration: $confidence_mode (no server started)."
        continue
    fi
    "$PYTHON_BIN" "$ROOT/Evaluator/dspark_once_control.py" port-free --port "$PORT"
    snapshot "before_${confidence_mode}"
    setsid bash "$ROOT/Evaluator/serve_dspark_full_suite.sh" "$confidence_mode" > "$SERVER_LOG" 2>&1 &
    SERVER_PID=$!
    echo "Starting $confidence_mode server on NPU $NPU; log: $SERVER_LOG"
    deadline=$((SECONDS + STARTUP_TIMEOUT))
    ready=0
    while (( SECONDS < deadline )); do
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo 'Server exited during startup.'; exit 1
        fi
        if curl --noproxy '*' --connect-timeout 1 --max-time 3 -fsS "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
            ready=1
            break
        fi
        sleep 2
    done
    (( ready == 1 )) || { echo "Server startup timed out after $STARTUP_TIMEOUT seconds."; exit 1; }
    "$PYTHON_BIN" "$ROOT/Evaluator/dspark_once_control.py" ready \
      --manifest "$ROOT/output/evaluator_servers/port_$PORT.json" \
      --pid "$SERVER_PID" --mode "$confidence_mode" --suite "$SUITE_DIR"
    echo "Evaluating $confidence_mode once at concurrency 128."
    bash "$ROOT/Evaluator/eval_dspark_full_suite.sh" 2>&1 | tee "$SUITE_DIR/logs/eval_${confidence_mode}.log"
    snapshot "after_${confidence_mode}"
    stop_server
done

if [[ "$CHECK_ONLY" == 1 ]]; then
    echo "Dry run complete: $SUITE_DIR (no inference performed)."
    exit 0
fi
"$PYTHON_BIN" "$ROOT/Evaluator/dspark_once_control.py" finish --suite "$SUITE_DIR"
