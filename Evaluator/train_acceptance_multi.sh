#!/usr/bin/env bash
# New block-training path. Does not call run_server_dflash.sh or acceptance_stream.
set -eo pipefail
bash -n -- "${BASH_SOURCE[0]}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# Derive repository from this installed Evaluator/ directory, ignoring stale exports.
export SPEC_MAIN="$(cd -- "$SCRIPT_DIR/.." && pwd)"
ENV_ROOT="${ACCEPTANCE_ENV_ROOT:-/home/n84449292/a84384644/attention_modif}"
PYTHON_BIN="$ENV_ROOT/bin/python"
CANN_ROOT="${ACCEPTANCE_CANN_ROOT:-/home/n84449292/m84379596/CANN/9.1.0}"
TARGET="${ACCEPTANCE_TARGET_MODEL:-/home/n84449292/m84379596/Huggingface/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c}"
DRAFT="${ACCEPTANCE_DRAFT_MODEL:-$SPEC_MAIN/output/dflash_prefix_compiled_joint_bs16/joint_memory_fast_20261009_041643/checkpoints/4}"
DATA="${ACCEPTANCE_DATASET:-/home/n84449292/m84379596/Huggingface/datasets/open_perfectblend.qwen3-4b-rollout.qwen3.seq3072/}"
OUT="${ACCEPTANCE_OUTPUT:-$SPEC_MAIN/output/acceptance_multi_$(date +%Y%m%d_%H%M%S)_$$}"
PORT="${ACCEPTANCE_PORT:-8114}"
TARGET_NPUS="${ACCEPTANCE_TARGET_NPUS:-0,1}"
TRAIN_NPU="${ACCEPTANCE_TRAIN_NPUS:-2,3,4,5}"
SEQ_LEN="${ACCEPTANCE_SEQ_LEN:-8192}"
ANCHORS="${ACCEPTANCE_ANCHORS:-256}"
HEAD_BATCH="${ACCEPTANCE_HEAD_BATCH:-384}"
WORKERS="${ACCEPTANCE_WORKERS:-2}"
VERIFY_WORKERS="${ACCEPTANCE_VERIFY_WORKERS:-4}"
MAX_SAMPLES="${ACCEPTANCE_MAX_SAMPLES:-0}"
EPOCHS="${ACCEPTANCE_EPOCHS:-1}"
LR="${ACCEPTANCE_LR:-0.001}"
[[ "$TARGET_NPUS" =~ ^[0-9]+(,[0-9]+)*$ && "$TRAIN_NPU" =~ ^[0-9]+(,[0-9]+)*$ ]] || { echo 'Invalid NPU list'; exit 1; }
IFS=, read -ra TARGET_DEVICES <<< "$TARGET_NPUS"
DP="${#TARGET_DEVICES[@]}"
IFS=, read -ra TRAIN_DEVICES <<< "$TRAIN_NPU"
NPROC="${#TRAIN_DEVICES[@]}"
[[ "$((HEAD_BATCH % NPROC))" == 0 ]] || { echo 'HEAD_BATCH must be divisible by training worker count'; exit 1; }
for ((i=0;i<NPROC;i++)); do
    for ((j=i+1;j<NPROC;j++)); do
        [[ "${TRAIN_DEVICES[i]}" != "${TRAIN_DEVICES[j]}" ]] || { echo 'Duplicate training NPU'; exit 1; }
    done
done
for id in "${TARGET_DEVICES[@]}"; do
    [[ ",$TRAIN_NPU," != *",$id,"* ]] || { echo 'Target and trainer NPUs must be separate'; exit 1; }
done
[[ ! -e "$OUT" ]] || { echo "Choose a new ACCEPTANCE_OUTPUT: $OUT"; exit 1; }
for file in "$PYTHON_BIN" "$SCRIPT_DIR/train_acceptance_multi.py" "$SCRIPT_DIR/acceptance_block_labels.py" "$SPEC_MAIN/speculators/scripts/launch_vllm.py"; do
    [[ -f "$file" ]] || { echo "Missing file: $file"; exit 1; }
done
source "$CANN_ROOT/cann/set_env.sh"
source "$CANN_ROOT/nnal/atb/set_env.sh"
export PATH="$ENV_ROOT/bin:$PATH"
export LD_LIBRARY_PATH="$ENV_ROOT/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$SPEC_MAIN/speculators/src:$SPEC_MAIN/speculators/hs_connectors/src:$SPEC_MAIN/vllm:$SPEC_MAIN/vllm-ascend:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
unset TORCH_COMPILE_DISABLE TORCHDYNAMO_DISABLE
export NO_PROXY=localhost,127.0.0.1,::1 no_proxy=localhost,127.0.0.1,::1
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY ftp_proxy FTP_PROXY
unset ASCEND_LAUNCH_BLOCKING RANK LOCAL_RANK WORLD_SIZE
unset VLLM_ASCEND_PREFIX_ACCEPTANCE_TRACE_DIR ACCEPTANCE_CONFIG VLLM_ASCEND_PREFIX_ACCEPTANCE_CONFIG
command -v curl >/dev/null
command -v setsid >/dev/null
"$PYTHON_BIN" -m py_compile "$SCRIPT_DIR/train_acceptance_multi.py" "$SCRIPT_DIR/acceptance_block_labels.py"
"$PYTHON_BIN" - "$DRAFT" "$TARGET" "$DATA" "$PORT" <<'PY'
import sys, json, socket
from pathlib import Path
import acl
from acl.rt import memcpy
import torch, torch_npu, datasets, transformers, openai, safetensors
from hs_connectors import FileTransfer
from torch.utils.tensorboard import SummaryWriter
from speculators.models.dflash_prefix.acceptance_features import build_acceptance_features
from speculators.models.dflash_prefix.acceptance_loss import acceptance_boundary_loss
from speculators.models.dflash_prefix.acceptance_head import AcceptedLengthHead
from speculators.data_generation.vllm_client import generate_hidden_states
for path in sys.argv[1:4]:
    if not Path(path).is_dir(): raise SystemExit(f'Missing model/data directory: {path}')
cfg=json.loads((Path(sys.argv[1])/'config.json').read_text())
if cfg.get('speculators_model_type')!='dflash_prefix': raise SystemExit('Expected your DFlash-Prefix checkpoint')
if not cfg.get('aux_hidden_state_layer_ids'): raise SystemExit('No target layers in draft config')
with socket.socket() as sock: sock.bind(('127.0.0.1',int(sys.argv[4])))
print('Preflight passed; source imports and port checked.')
PY
read -ra LAYERS <<< "$("$PYTHON_BIN" - "$DRAFT/config.json" <<'PY'
import json,sys
print(' '.join(map(str,json.load(open(sys.argv[1]))['aux_hidden_state_layer_ids'])))
PY
)"
mkdir -p "$OUT"
cp "$SCRIPT_DIR/train_acceptance_multi.py" "$SCRIPT_DIR/acceptance_block_labels.py" "$OUT/"
cp "${BASH_SOURCE[0]}" "$OUT/launcher_source.sh"
SERVER_PID='' TRAIN_PID=''
cleanup() {
    local pid
    trap '' INT TERM
    for pid in "$TRAIN_PID" "$SERVER_PID"; do
        [[ -z "$pid" ]] || kill -TERM -- "-$pid" 2>/dev/null || true
    done
    sleep 3
    for pid in "$TRAIN_PID" "$SERVER_PID"; do
        [[ -z "$pid" ]] || kill -KILL -- "-$pid" 2>/dev/null || true
    done
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
cd "$SPEC_MAIN/speculators"
echo "Starting hidden-state target server on NPUs $TARGET_NPUS; port $PORT"
echo "Output: $OUT"
ASCEND_RT_VISIBLE_DEVICES="$TARGET_NPUS" setsid "$PYTHON_BIN" scripts/launch_vllm.py "$TARGET" \
    --target-layer-ids "${LAYERS[@]}" -- \
    --data-parallel-size "$DP" --port "$PORT" \
    --max-model-len "$((SEQ_LEN+256))" --gpu-memory-utilization 0.85 \
    > "$OUT/server.log" 2>&1 &
SERVER_PID=$!
ready=0
for ((attempt=0;attempt<360;attempt++)); do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then tail -n 100 "$OUT/server.log"; exit 1; fi
    if curl --noproxy '*' --connect-timeout 2 --max-time 3 -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null; then ready=1; break; fi
    sleep 5
done
[[ "$ready" == 1 ]] || { tail -n 100 "$OUT/server.log"; echo 'Target server startup timed out'; exit 1; }
extra=()
[[ -z "${ACCEPTANCE_RESUME_HEAD:-}" ]] || extra+=(--resume-head "$ACCEPTANCE_RESUME_HEAD")
echo "Frozen draft + trainable MLP on NPU $TRAIN_NPU; $ANCHORS blocks per pack; head batch $HEAD_BATCH"
echo "Full dataset unless ACCEPTANCE_MAX_SAMPLES > 0; current: $MAX_SAMPLES"
echo "Training log: $OUT/train.log"
echo "TensorBoard: $OUT/tensorboard"
TORCH_COMPILE_DISABLE=1 TORCHDYNAMO_DISABLE=1 ASCEND_RT_VISIBLE_DEVICES="$TRAIN_NPU" setsid "$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node="$NPROC" "$SCRIPT_DIR/train_acceptance_multi.py" \
    --target "$TARGET" --draft "$DRAFT" --dataset "$DATA" --output "$OUT" \
    --endpoint "http://127.0.0.1:$PORT/v1" --seq-len "$SEQ_LEN" \
    --anchors "$ANCHORS" --head-batch "$HEAD_BATCH" --workers "$WORKERS" \
    --verify-workers "$VERIFY_WORKERS" --max-samples "$MAX_SAMPLES" \
    --epochs "$EPOCHS" --lr "$LR" "${extra[@]}" \
    > >(tee "$OUT/train.log") 2>&1 &
TRAIN_PID=$!
while kill -0 "$TRAIN_PID" 2>/dev/null; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then tail -n 100 "$OUT/server.log"; echo 'Target server exited'; exit 1; fi
    sleep 2
done
if wait "$TRAIN_PID"; then
    TRAIN_PID=''
else
    status=$?
    TRAIN_PID=''
    echo "Training failed with exit code $status"
    tail -n 80 "$OUT/train.log"
    exit "$status"
fi
[[ -f "$OUT/checkpoints/final.pt" ]] || { echo 'No final checkpoint produced'; exit 1; }
echo "Finished: $OUT/checkpoints/final.pt"
