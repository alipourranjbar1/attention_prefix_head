#!/usr/bin/env bash
# Four-anchor regular DFlash; five bidirectional layers; target 0,1 / trainers 2..7.
set -eo pipefail
FOUR_ANCHOR_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SPEC_DIR="$FOUR_ANCHOR_ROOT/speculators"
: "${CONDA_PREFIX:?Activate attention_prefix_ali first}"
if [[ -z "${CANN_ROOT:-}" ]]; then
    for candidate in /home/n84449292/a84384644/CANN/9.1.0 /home/n84449292/m84379596/CANN/9.1.0; do
        if [[ -f "$candidate/cann/set_env.sh" ]]; then CANN_ROOT="$candidate"; break; fi
    done
fi
if [[ -n "${CANN_ROOT:-}" ]]; then
    source "$CANN_ROOT/cann/set_env.sh"
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
    DFLASH_TORCH_ABI=$(TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -c 'import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))')
    source "$CANN_ROOT/nnal/atb/set_env.sh" --cxx_abi="$DFLASH_TORCH_ABI"
fi
set -u
export MODEL="${MODEL:-/home/n84449292/m84379596/Huggingface/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c}"
export DATASET="${DATASET:-/home/n84449292/m84379596/Huggingface/datasets/open_perfectblend.qwen3-4b-rollout.qwen3.seq3072}"
VLLM_SOURCE_ROOT="${VLLM_SOURCE_ROOT:-/home/n84449292/a84384644/attention_head_forked/attention_prefix_head}"
export PYTHONPATH="$FOUR_ANCHOR_ROOT:$SPEC_DIR/src:$SPEC_DIR/hs_connectors/src:$VLLM_SOURCE_ROOT/vllm:$VLLM_SOURCE_ROOT/vllm-ascend:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${ATB_HOME_PATH:?ATB environment missing}/lib:$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export TORCH_COMPILE_DISABLE=1 TORCHDYNAMO_DISABLE=1
unset ASCEND_LAUNCH_BLOCKING DFLASH_TP_GATHER
export no_proxy="localhost,127.0.0.1,::1" NO_PROXY="localhost,127.0.0.1,::1"
export SOC_VERSION="${SOC_VERSION:-ascend910_9372}"
export TARGET_NPUS="${TARGET_NPUS:-0,1}" TRAIN_NPUS="${TRAIN_NPUS:-2,3,4,5,6,7}"
export PORT="${PORT:-8080}" SEQ_LENGTH="${SEQ_LENGTH:-3072}" BLOCK_SIZE="${BLOCK_SIZE:-16}"
export EPOCHS="${EPOCHS:-5}" LR="${LR:-0.0006}" SEED="${SEED:-42}"
export GRADIENT_ACCUMULATION="${GRADIENT_ACCUMULATION:-1}" LOG_FREQ="${LOG_FREQ:-10}"
export MAX_ANCHORS="${MAX_ANCHORS:-512}"  # sampled blocks per packed batch; four anchors inside each block
export NUM_WORKERS="${NUM_WORKERS:-4}" TRAIN_DATA_RATIO="${TRAIN_DATA_RATIO:-0.9}"
export BLOCKS_PER_CHUNK="${BLOCKS_PER_CHUNK:-32}" HEAD_CHUNK_SIZE="${HEAD_CHUNK_SIZE:-64}"
if [[ -z "${LOSS_FN:-}" ]]; then export LOSS_FN='{"ce":0.1,"tv":0.9}'; fi
export OUTPUT_DIR="${OUTPUT_DIR:-$FOUR_ANCHOR_ROOT/output/four_anchor_dflash_$(date +%Y%m%d_%H%M%S)_$$}"
export DATA_CACHE_DIR="${DATA_CACHE_DIR:-$FOUR_ANCHOR_ROOT/output/.four_anchor_data_cache}"
cd "$FOUR_ANCHOR_ROOT"
python - <<'PY'
import importlib.util
import os
import torch
import torch_npu
from torch_npu.op_plugin.atb._atb_ops import _register_atb_extensions
print('ATB path:', os.environ['ATB_HOME_PATH'], flush=True)
print('torch:', torch.__version__, 'torch_npu:', torch_npu.__version__, flush=True)
print('PyTorch CXX11 ABI:', torch._C._GLIBCXX_USE_CXX11_ABI, flush=True)
_register_atb_extensions()
from speculators.models.dflash.model_definitions import Qwen3DFlashDecoderLayer
from four_anchor.remote_teacher import RemoteTeacher
from torch.utils.tensorboard import SummaryWriter
assert importlib.util.find_spec('vllm') is not None, 'vLLM is not importable in this environment'
print('Preflight imports OK: regular DFlash, remote teacher client, TensorBoard', flush=True)
PY
python -m four_anchor.ascend_config
printf 'Packed token budget: %s per rank; up to %s sampled training blocks; loss=%s\n' "$SEQ_LENGTH" "$MAX_ANCHORS" "$LOSS_FN"
printf 'Training layout: 4 conditioning anchors + %s new slots; all five draft layers bidirectional.\n' "$BLOCK_SIZE"
IFS=',' read -r -a train_devices <<< "$TRAIN_NPUS"
# Count real packed updates before starting either the target server or trainers.
python -m four_anchor.packed_data --config "$OUTPUT_DIR/train_config.json" --world-size "${#train_devices[@]}"
if [[ "${CHECK_ONLY:-0}" == 1 ]]; then exit 0; fi
mkdir -p "$OUTPUT_DIR/hidden_states"
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"
export OUTPUT_DIR
IFS=',' read -r -a target_devices <<< "$TARGET_NPUS"
IFS=',' read -r -a train_devices <<< "$TRAIN_NPUS"
python - "$PORT" <<'PY'
import socket,sys
with socket.socket() as sock:
    sock.bind(('127.0.0.1',int(sys.argv[1])))
PY
server_pid=''
cleanup() {
    if [[ -n "$server_pid" ]]; then kill -TERM -- "-$server_pid" 2>/dev/null || true; fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
# Our saved layer IDs name decoder outputs 0-based. vLLM numbers embedding=0,
# output of decoder i=i+1. Do NOT silently request 1,9,... for this checkpoint.
setsid env ASCEND_RT_VISIBLE_DEVICES="$TARGET_NPUS" \
    python "$SPEC_DIR/scripts/launch_vllm.py" train "$MODEL" \
    --target-layer-ids 2 10 18 26 34 --include-last-layer \
    --hidden-states-path "$OUTPUT_DIR/hidden_states" \
    --provenance-dir "$OUTPUT_DIR" -- \
    --served-model-name four-anchor-teacher --host 127.0.0.1 \
    --data-parallel-size "${#target_devices[@]}" \
    --api-server-count 1 --renderer-num-workers 2 \
    --port "$PORT" --max-model-len "$((SEQ_LENGTH + 256))" \
    --gpu-memory-utilization 0.85 >"$OUTPUT_DIR/target_server.log" 2>&1 &
server_pid=$!
ready=0
for ((elapsed=0; elapsed<${BOOT_TIMEOUT:-1800}; elapsed+=5)); do
    if ! kill -0 "$server_pid" 2>/dev/null; then tail -n 80 "$OUTPUT_DIR/target_server.log"; exit 1; fi
    if curl --noproxy '*' --connect-timeout 2 --max-time 5 -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null; then ready=1; break; fi
    sleep 5
done
if [[ "$ready" != 1 ]]; then tail -n 80 "$OUTPUT_DIR/target_server.log"; exit 1; fi
# Check the real server's file/token-ID contract before allocating six drafters.
python -m four_anchor.check_teacher --config "$OUTPUT_DIR/train_config.json" \
    2>&1 | tee "$OUTPUT_DIR/teacher_check.log"
ASCEND_RT_VISIBLE_DEVICES="$TRAIN_NPUS" python -m torch.distributed.run \
    --nnodes=1 --nproc_per_node="${#train_devices[@]}" \
    --master_addr=127.0.0.1 --master_port="${MASTER_PORT:-29637}" \
    -m four_anchor.cli train --config "$OUTPUT_DIR/train_config.json" \
    2>&1 | tee "$OUTPUT_DIR/train.log"
printf 'Training finished: %s\nTensorBoard logs: %s/checkpoints/logs/four_anchor_dflash\n' "$OUTPUT_DIR" "$OUTPUT_DIR"
