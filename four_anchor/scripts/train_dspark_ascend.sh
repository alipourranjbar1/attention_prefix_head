#!/usr/bin/env bash
# Isolated 20/19-query DSpark training. NPU 7 is not allocated by default.
set -eo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
: "${CONDA_PREFIX:?Activate attention_prefix_ali first}"
BASE=/home/n84449292
FORK="$BASE/a84384644/attention_head_forked"
if [[ -f "$ROOT/dependency_paths.json" ]]; then
    installed_dependencies="$(python - "$ROOT/dependency_paths.json" <<'PY_PATH'
import json, sys
print(json.load(open(sys.argv[1]))['dependencies'])
PY_PATH
)"
    installed_speculators="$(python - "$ROOT/dependency_paths.json" <<'PY_SPEC'
import json, sys
print(json.load(open(sys.argv[1]))['speculators'])
PY_SPEC
)"
fi
export VLLM_SOURCE_ROOT="${VLLM_SOURCE_ROOT:-${installed_dependencies:-$FORK/four-anchor-dflash-pipeline}}"
export SPECULATORS_ROOT="${SPECULATORS_ROOT:-${installed_speculators:-$VLLM_SOURCE_ROOT/speculators}}"
export CANN_ROOT="${CANN_ROOT:-$BASE/m84379596/CANN/9.1.0}"
if [[ -f "$CANN_ROOT/cann/set_env.sh" ]]; then
    source "$CANN_ROOT/cann/set_env.sh"
else
    source "$CANN_ROOT/ascend-toolkit/set_env.sh"
fi
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
ABI=$(TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -c 'import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))')
source "$CANN_ROOT/nnal/atb/set_env.sh" --cxx_abi="$ABI"
set -u
export MODEL="${MODEL:-$BASE/m84379596/Huggingface/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c}"
export DATASET="${DATASET:-$BASE/m84379596/Huggingface/datasets/open_perfectblend.qwen3-4b-rollout.qwen3.seq3072}"
export PYTHONPATH="$ROOT:$SPECULATORS_ROOT/src:$SPECULATORS_ROOT/hs_connectors/src:$VLLM_SOURCE_ROOT/vllm:$VLLM_SOURCE_ROOT/vllm-ascend:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${ATB_HOME_PATH:?ATB environment missing}/lib:$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TORCH_COMPILE_DISABLE=1 TORCHDYNAMO_DISABLE=1
# These disable torch.compile, not explicit torch.npu.NPUGraph capture.
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export no_proxy="localhost,127.0.0.1,::1" NO_PROXY="localhost,127.0.0.1,::1"
export SOC_VERSION="${SOC_VERSION:-ascend910_9372}"
unset ASCEND_LAUNCH_BLOCKING DFLASH_TP_GATHER TORCH_DEVICE_BACKEND_AUTOLOAD
export TARGET_NPUS="${TARGET_NPUS:-0,1}" TRAIN_NPUS="${TRAIN_NPUS:-2,3,4,5,6}"
export PORT="${PORT:-8096}" MASTER_PORT="${MASTER_PORT:-29657}" RPC_PORT="${RPC_PORT:-29658}"
export SEQ_LENGTH="${SEQ_LENGTH:-9216}" MAX_ANCHORS="${MAX_ANCHORS:-1536}"
export EPOCHS="${EPOCHS:-5}" LR="${LR:-0.0006}" TRAIN_DATA_RATIO="${TRAIN_DATA_RATIO:-1.0}"
export BLOCKS_PER_CHUNK="${BLOCKS_PER_CHUNK:-128}" HEAD_CHUNK_SIZE="${HEAD_CHUNK_SIZE:-256}"
export GRADIENT_ACCUMULATION="${GRADIENT_ACCUMULATION:-1}" NUM_WORKERS="${NUM_WORKERS:-4}"
export DISTRIBUTED_STRATEGY="${DISTRIBUTED_STRATEGY:-fsdp}" PACKED_EXECUTION=cached
export MARKOV_RANK="${MARKOV_RANK:-256}" CONFIDENCE_ALPHA="${CONFIDENCE_ALPHA:-1.0}"
export BOUNDARY_CONFIDENCE="${BOUNDARY_CONFIDENCE:-confidence_head}"
export BOUNDARY_EXPLORATION="${BOUNDARY_EXPLORATION:-0.5}" BOUNDARY_BETA="${BOUNDARY_BETA:-0.7}"
# The supplied real-NPU loss check measured the graph slower than eager.
# Keep this experimental loss-only graph opt-in; it is not whole-step capture.
export LOSS_GRAPH="${LOSS_GRAPH:-off}" PROFILE_STEPS="${PROFILE_STEPS:-5}"
export TARGET_EXECUTION="${TARGET_EXECUTION:-eager}"
export HEAD_LOSS_BACKEND="${HEAD_LOSS_BACKEND:-checkpoint}"
export GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-1}"
export PROPOSAL_BASE_BATCHED="${PROPOSAL_BASE_BATCHED:-0}"
export LOG_FREQ="${LOG_FREQ:-10}" EVAL_EACH_EPOCH="${EVAL_EACH_EPOCH:-0}"
export ANCHOR_LAYOUT=bonus_addon BLOCK_SIZE=15
if [[ -z "${LOSS_FN:-}" ]]; then export LOSS_FN='{"ce":0.1,"tv":0.9}'; fi
if [[ -n "${RESUME:-}" && -z "${OUTPUT_DIR:-}" ]]; then
    export OUTPUT_DIR="$(cd -- "$RESUME/../.." && pwd)"
fi
export OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/output/dspark_bonus20_$(date +%Y%m%d_%H%M%S)_$$}"
export DATA_CACHE_DIR="${DATA_CACHE_DIR:-$ROOT/output/.four_anchor_data_cache}"
cd "$ROOT"
flags=()
if [[ "${CHECK_ONLY:-0}" == 1 ]]; then flags+=(--metadata-only); fi
python -m four_anchor.dspark_preflight --dependencies "$VLLM_SOURCE_ROOT" --speculators "$SPECULATORS_ROOT" \
    --targets "$TARGET_NPUS" --trainers "$TRAIN_NPUS" "${flags[@]}"
python - <<'PY'
import torch, torch_npu
from torch_npu.op_plugin.atb._atb_ops import _register_atb_extensions
_register_atb_extensions()
from four_anchor.dspark_model import FourAnchorDSpark
from torch.utils.tensorboard import SummaryWriter
print('DSpark imports OK:', torch.__version__, torch_npu.__version__, flush=True)
PY
python -m four_anchor.dspark_ascend_config
IFS=',' read -r -a trainers <<< "$TRAIN_NPUS"
IFS=',' read -r -a targets <<< "$TARGET_NPUS"
WORLD_SIZE="${#trainers[@]}" python -m four_anchor.dspark_train --config "$OUTPUT_DIR/train_config.json" --plan
if [[ "${CHECK_ONLY:-0}" == 1 ]]; then exit 0; fi
python - "$PORT" "$MASTER_PORT" "$RPC_PORT" <<'PY'
import socket, sys
ports = [int(p) for p in sys.argv[1:]]
if len(ports) != len(set(ports)):
    raise SystemExit('PORT, MASTER_PORT and RPC_PORT must differ')
for port in ports:
    with socket.socket() as sock:
        try:
            sock.bind(('0.0.0.0', port))
        except OSError as error:
            raise SystemExit(f'Port {port} is busy. Choose another port: {error}')
PY
if [[ "$LOSS_GRAPH" == npu ]]; then
    ASCEND_RT_VISIBLE_DEVICES="${trainers[0]}" python -m four_anchor.check_dspark_graph \
        --rows "$HEAD_CHUNK_SIZE" --vocab 151936 --output "$OUTPUT_DIR/loss_graph_check.json" \
        2>&1 | tee "$OUTPUT_DIR/loss_graph_check.log"
fi
if [[ "$HEAD_LOSS_BACKEND" == cached_vjp ]]; then
    ASCEND_RT_VISIBLE_DEVICES="${trainers[0]}" python -m four_anchor.check_cached_vjp \
        --device npu:0 --output "$OUTPUT_DIR/cached_vjp_check.json" \
        2>&1 | tee "$OUTPUT_DIR/cached_vjp_check.log"
fi
if [[ "$DISTRIBUTED_STRATEGY" == fsdp ]]; then
    ASCEND_RT_VISIBLE_DEVICES="$TRAIN_NPUS" python -m torch.distributed.run \
        --nnodes=1 --nproc_per_node="${#trainers[@]}" --master_addr=127.0.0.1 --master_port="$MASTER_PORT" \
        -m four_anchor.fsdp_check --device npu --dtype bfloat16 --draft-type dspark \
        --head-loss-backend "$HEAD_LOSS_BACKEND" \
        --loss-graph "$LOSS_GRAPH" --anchor-layout bonus_addon --packed-execution cached \
        --output "$OUTPUT_DIR/fsdp_check" 2>&1 | tee "$OUTPUT_DIR/fsdp_check.log"
fi
if [[ "${HARDWARE_CHECK_ONLY:-0}" == 1 ]]; then
    echo "Hardware checks passed. No teacher server or full training was started."
    exit 0
fi
# Recheck after preflight before allocating full models, leaving all other jobs alone.
python -m four_anchor.dspark_preflight --dependencies "$VLLM_SOURCE_ROOT" --speculators "$SPECULATORS_ROOT" \
    --targets "$TARGET_NPUS" --trainers "$TRAIN_NPUS"
mkdir -p "$OUTPUT_DIR/hidden_states"
server_pid=''
cleanup() {
    if [[ -n "$server_pid" ]]; then
        kill -TERM -- "-$server_pid" 2>/dev/null || true
        wait "$server_pid" 2>/dev/null || true
    fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
setsid env ASCEND_RT_VISIBLE_DEVICES="$TARGET_NPUS" \
    python -m four_anchor.target_serving \
    >"$OUTPUT_DIR/target_server.log" 2>&1 &
server_pid=$!
ready=0
for ((elapsed=0; elapsed<${BOOT_TIMEOUT:-1800}; elapsed+=5)); do
    if ! kill -0 "$server_pid" 2>/dev/null; then tail -n 80 "$OUTPUT_DIR/target_server.log"; exit 1; fi
    if curl --noproxy '*' --connect-timeout 2 --max-time 5 -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null; then ready=1; break; fi
    sleep 5
done
if [[ "$ready" != 1 ]]; then tail -n 80 "$OUTPUT_DIR/target_server.log"; exit 1; fi
python -m four_anchor.check_teacher --config "$OUTPUT_DIR/train_config.json" 2>&1 | tee "$OUTPUT_DIR/teacher_check.log"
ASCEND_RT_VISIBLE_DEVICES="$TRAIN_NPUS" python -m torch.distributed.run \
    --nnodes=1 --nproc_per_node="${#trainers[@]}" --master_addr=127.0.0.1 --master_port="$MASTER_PORT" \
    -m four_anchor.dspark_train --config "$OUTPUT_DIR/train_config.json" 2>&1 | tee "$OUTPUT_DIR/train.log"
printf 'Checkpoints: %s/checkpoints\nTraining log: %s/train.log\n' "$OUTPUT_DIR" "$OUTPUT_DIR"
