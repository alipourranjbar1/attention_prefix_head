#!/usr/bin/env bash
# Full-dataset rollout collection with bounded concurrency; train only the head.
set -eo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export SPEC_MAIN="/home/n84449292/a84384644/attention_imp_code/vLLM_NPU_spec_main"
export EXPERIMENT="${EXPERIMENT:-$SPEC_MAIN/output/acceptance_fast_$(date +%Y%m%d_%H%M%S)_$$}"
export ENV_ROOT="${ENV_ROOT:-/home/n84449292/a84384644/attention_modif}"
export PYTHON_BIN="${PYTHON_BIN:-$ENV_ROOT/bin/python}"
export CANN_ROOT="${CANN_ROOT:-/home/n84449292/m84379596/CANN/9.1.0}"
export CANN_ENV_SH="${CANN_ENV_SH:-$CANN_ROOT/cann/set_env.sh}"
export ATB_ENV_SH="${ATB_ENV_SH:-$CANN_ROOT/nnal/atb/set_env.sh}"
export TARGET_MODEL="${TARGET_MODEL:-/home/n84449292/m84379596/Huggingface/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c}"
export DRAFT_MODEL="${DRAFT_MODEL:-$SPEC_MAIN/output/dflash_prefix_compiled_joint_bs16/joint_memory_fast_20261009_041643/checkpoints/4}"
export DATASET="${DATASET:-/home/n84449292/m84379596/Huggingface/datasets/open_perfectblend.qwen3-4b-rollout.qwen3.seq3072/}"
# 384 blocks on one device matches the old 128 x 3 global batch by default.
export BATCH_SIZE="${BATCH_SIZE:-384}" LR="${LR:-0.001}" SEED="${SEED:-42}"
export SAVE_EVERY="${SAVE_EVERY:-100}" LOG_EVERY="${LOG_EVERY:-10}"
export COLLECT_CONCURRENCY="${COLLECT_CONCURRENCY:-8}"
export MAX_SAMPLES="${MAX_SAMPLES:-0}"  # 0 = full dataset, no response-length cap
export REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-1800}" TRACE_STALL_TIMEOUT="${TRACE_STALL_TIMEOUT:-1800}"
export PORT="${PORT:-8113}" NPU="${NPU:-0}" NUM_SPECULATIVE_TOKENS=15
TRAIN_NPU="${TRAIN_NPU:-1}"
export SERVED_MODEL_NAME=qwen3-4b-prefix-dflash SPECULATIVE_METHOD=dflash
# Preserve the known trace execution mode. Do not guess graph-support flags.
export TARGET_EXECUTION=eager DRAFT_ENFORCE_EAGER=1 COMPARISON_PROFILE=custom
export HEAD_EXECUTION=reference MODE=reference
export MEMORY_UTILIZATION="${MEMORY_UTILIZATION:-0.85}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}" MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-4096}"
export VLLM_ASCEND_PREFIX_ACCEPTANCE_TRACE_DIR="$EXPERIMENT/traces"
export RUN_DIR="$EXPERIMENT/server_online_$(date +%Y%m%d_%H%M%S)_$$"
unset ACCEPTANCE_CONFIG VLLM_ASCEND_PREFIX_ACCEPTANCE_CONFIG
[[ "$NPU" =~ ^[0-9]+$ && "$TRAIN_NPU" =~ ^[0-9]+$ && "$NPU" != "$TRAIN_NPU" ]] || { echo 'Choose two distinct single NPU IDs'; exit 1; }
[[ ! -e "$EXPERIMENT" ]] || { echo "Use a new EXPERIMENT directory: $EXPERIMENT"; exit 1; }
[[ -f "$SCRIPT_DIR/acceptance_stream_fast.py" ]] || { echo 'Put acceptance_stream_fast.py beside this launcher'; exit 1; }
[[ -f "$SPEC_MAIN/Evaluator/run_server_dflash.sh" ]] || { echo 'Missing run_server_dflash.sh'; exit 1; }
source "$CANN_ENV_SH"
source "$ATB_ENV_SH"
export LD_LIBRARY_PATH="$ENV_ROOT/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$SPEC_MAIN/speculators/src:$SPEC_MAIN/vllm:$SPEC_MAIN/vllm-ascend:${PYTHONPATH:-}"
export NO_PROXY=localhost,127.0.0.1,::1 no_proxy=localhost,127.0.0.1,::1
export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
unset ASCEND_LAUNCH_BLOCKING
command -v setsid >/dev/null
command -v curl >/dev/null
"$PYTHON_BIN" - <<'PY'
import os, socket, torch, torch_npu, datasets, tqdm
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
for name in ('acceptance_head.py', 'acceptance_loss.py'):
    p = Path(os.environ['SPEC_MAIN'])/'speculators/src/speculators/models/dflash_prefix'/name
    if not p.is_file():
        raise SystemExit(f'Missing dependency: {p}')
for name in ('BATCH_SIZE', 'COLLECT_CONCURRENCY', 'MAX_MODEL_LEN'):
    if int(os.environ[name]) < 1:
        raise SystemExit(f'{name} must be positive')
if int(os.environ['MAX_SAMPLES']) < 0:
    raise SystemExit('MAX_SAMPLES must be zero or positive')
with socket.socket() as sock:
    sock.bind(('127.0.0.1', int(os.environ['PORT'])))
print('Dependencies checked; port is free.')
PY
mkdir -p "$EXPERIMENT/traces"
cp "$SCRIPT_DIR/acceptance_stream_fast.py" "$EXPERIMENT/trainer_source.py"
cp "${BASH_SOURCE[0]}" "$EXPERIMENT/launcher_source.sh"
SERVER_PID='' COLLECT_PID='' TRAIN_PID=''
cleanup() {
    local pid active=0
    for pid in "$TRAIN_PID" "$COLLECT_PID" "$SERVER_PID"; do
        if [[ -n "$pid" ]]; then kill -TERM -- "-$pid" 2>/dev/null || true; active=1; fi
    done
    if [[ "$active" == 1 ]]; then
        sleep 3
        for pid in "$TRAIN_PID" "$COLLECT_PID" "$SERVER_PID"; do
            if [[ -n "$pid" ]]; then kill -KILL -- "-$pid" 2>/dev/null || true; fi
        done
    fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
cd "$SPEC_MAIN"
ASCEND_RT_VISIBLE_DEVICES="$NPU" setsid bash Evaluator/run_server_dflash.sh > "$EXPERIMENT/server.log" 2>&1 &
SERVER_PID=$!
READY=0
for ((attempt=0; attempt<360; attempt++)); do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then tail -n 80 "$EXPERIMENT/server.log"; exit 1; fi
    if curl --noproxy '*' --connect-timeout 2 --max-time 3 -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null; then READY=1; break; fi
    sleep 5
done
[[ "$READY" == 1 ]] || { echo 'Server startup timed out'; exit 1; }
setsid "$PYTHON_BIN" "$SCRIPT_DIR/acceptance_stream_fast.py" collect > "$EXPERIMENT/collection.log" 2>&1 &
COLLECT_PID=$!
ASCEND_RT_VISIBLE_DEVICES="$TRAIN_NPU" setsid "$PYTHON_BIN" "$SCRIPT_DIR/acceptance_stream_fast.py" train > "$EXPERIMENT/train.log" 2>&1 &
TRAIN_PID=$!
echo "Collector: $COLLECT_CONCURRENCY concurrent requests; inference NPU $NPU; MLP NPU $TRAIN_NPU"
echo "Full dataset unless MAX_SAMPLES > 0 (current: $MAX_SAMPLES). Global head batch: $BATCH_SIZE"
echo "Collection: $EXPERIMENT/collection.log"
echo "Training:   $EXPERIMENT/train.log"
echo "TensorBoard: $EXPERIMENT/tensorboard"
echo "Watch logs: tail -f '$EXPERIMENT/collection.log' '$EXPERIMENT/train.log'"
# Monitor all children: a failed collector must not leave the trainer waiting.
while true; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then echo 'Server exited unexpectedly'; tail -n 80 "$EXPERIMENT/server.log"; exit 1; fi
    if [[ -n "$COLLECT_PID" ]] && ! kill -0 "$COLLECT_PID" 2>/dev/null; then
        if wait "$COLLECT_PID"; then COLLECT_PID=''; else status=$?; COLLECT_PID=''; tail -n 80 "$EXPERIMENT/collection.log"; exit "$status"; fi
    fi
    if ! kill -0 "$TRAIN_PID" 2>/dev/null; then
        if wait "$TRAIN_PID"; then TRAIN_PID=''; else status=$?; TRAIN_PID=''; tail -n 80 "$EXPERIMENT/train.log"; exit "$status"; fi
        [[ -f "$EXPERIMENT/collection.done" && -f "$EXPERIMENT/checkpoints/final.pt" ]] || { echo 'Trainer exited before completing the run'; exit 1; }
        if [[ -n "$COLLECT_PID" ]]; then wait "$COLLECT_PID"; COLLECT_PID=''; fi
        break
    fi
    sleep 2
done
echo "Finished: $EXPERIMENT/checkpoints/final.pt"
