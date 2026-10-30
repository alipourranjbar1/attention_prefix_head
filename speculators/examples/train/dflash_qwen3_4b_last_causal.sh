#!/usr/bin/env bash
# Five plain DFlash layers: four bidirectional + one causal. No prefix module.
# Run inside the existing Ascend training environment; optionally set CANN_ROOT.
set -eo pipefail
SPEC_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
SPEC_MAIN="$(cd -- "$SPEC_DIR/.." && pwd)"
if [[ -n "${CANN_ROOT:-}" ]]; then
    source "$CANN_ROOT/cann/set_env.sh"
    export LD_LIBRARY_PATH="${CONDA_PREFIX:?Activate the training Conda environment}/lib:${LD_LIBRARY_PATH:-}"
    DFLASH_TORCH_ABI=$(TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -c 'import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))')
    source "$CANN_ROOT/nnal/atb/set_env.sh" --cxx_abi="$DFLASH_TORCH_ABI" 
fi
set -u
: "${MODEL:?Set MODEL to your local Qwen3-4B directory}"
: "${DATASET:?Set DATASET to your preprocessed HF save_to_disk dataset}"
export PYTHONPATH="$SPEC_DIR/src:$SPEC_DIR/hs_connectors/src:$SPEC_MAIN/vllm:$SPEC_MAIN/vllm-ascend:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${ATB_HOME_PATH:?ATB environment missing}/lib:${CONDA_PREFIX:-/usr}/lib:${LD_LIBRARY_PATH:-}"
export TORCH_COMPILE_DISABLE=1 TORCHDYNAMO_DISABLE=1
unset ASCEND_LAUNCH_BLOCKING DFLASH_TP_GATHER
export no_proxy="localhost,127.0.0.1,::1" NO_PROXY="localhost,127.0.0.1,::1"
export SOC_VERSION="${SOC_VERSION:-ascend910_9372}"
TARGET_NPUS="${TARGET_NPUS:-0,1}"
TRAIN_NPUS="${TRAIN_NPUS:-2,3,4,5,6,7}"
PORT="${PORT:-8080}"
SEQ_LENGTH="${SEQ_LENGTH:-3072}"
MAX_ANCHORS="${MAX_ANCHORS:-512}"
BLOCK_SIZE="${BLOCK_SIZE:-16}"
EPOCHS="${EPOCHS:-5}"
LR="${LR:-0.0006}"
SEED="${SEED:-42}"
OUTPUT_DIR="${OUTPUT_DIR:-$SPEC_MAIN/output/dflash_5layer_last_causal_$(date +%Y%m%d_%H%M%S)_$$}"
export MODEL DATASET OUTPUT_DIR TARGET_NPUS TRAIN_NPUS
cd "$SPEC_DIR"
python - <<'PY'
import json, os
from pathlib import Path
import torch
import torch_npu
from torch_npu.op_plugin.atb._atb_ops import _register_atb_extensions
print("ATB path:", os.environ.get("ATB_HOME_PATH"), flush=True)
print("torch:", torch.__version__, "torch_npu:", torch_npu.__version__, flush=True)
print("PyTorch CXX11 ABI:", torch._C._GLIBCXX_USE_CXX11_ABI, flush=True)
_register_atb_extensions()
print("ATB library load: OK", flush=True)
from speculators.models.dflash.config import DFlashSpeculatorConfig
from speculators.train.config import TrainConfig
assert 'last_layer_causal' in DFlashSpeculatorConfig.model_fields, 'Apply the patch first'
assert TrainConfig.from_flat({'verifier_name_or_path': os.environ['MODEL'],
    'speculator_type': 'dflash', 'last_layer_causal': True}).flatten()['last_layer_causal']
cfg = json.loads((Path(os.environ['MODEL']) / 'config.json').read_text())
cfg = cfg.get('text_config', cfg)
assert cfg['vocab_size'] == 151936, 'Expected Qwen3-4B full vocabulary'
assert cfg['num_hidden_layers'] > 33, 'Target must expose layers 1,9,17,25,33'
data = Path(os.environ['DATASET'])
assert (data / 'state.json').exists() and (data / 'dataset_info.json').exists(), 'Expected a preprocessed save_to_disk dataset'
features = json.loads((data / 'dataset_info.json').read_text())['features']
assert {'input_ids', 'loss_mask'} <= set(features)
target = os.environ['TARGET_NPUS'].split(',')
train = os.environ['TRAIN_NPUS'].split(',')
assert all(x.isdigit() for x in target + train), 'Use comma-separated numeric NPU IDs'
assert len(set(target + train)) == len(target + train), 'NPU IDs must be distinct and disjoint'
for path in Path(os.environ['OUTPUT_DIR']).glob('checkpoints/**/config.json'):
    saved = json.loads(path.read_text())
    tl = saved.get('transformer_layer_config', {})
    assert saved.get('last_layer_causal') is True, f'Wrong attention config: {path}'
    assert tl.get('num_hidden_layers') == 5, f'Wrong layer count: {path}'
    assert tl.get('layer_types') == ['full_attention'] * 5, f'Wrong context config: {path}'
print('Preflight OK: plain DFlash, 5 layers, masks = [bidir,bidir,bidir,bidir,causal]')
PY
python scripts/check_dflash_last_causal.py
if [[ "${CHECK_ONLY:-0}" == 1 ]]; then exit 0; fi
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"
IFS=',' read -r -a target_devices <<< "$TARGET_NPUS"
IFS=',' read -r -a train_devices <<< "$TRAIN_NPUS"
# Refuse to attach to another job's server.
python - "$PORT" <<'PY'
import socket, sys
with socket.socket() as s:
    s.bind(('127.0.0.1', int(sys.argv[1])))
PY
server_pid=''
cleanup() {
    if [[ -n "$server_pid" ]]; then
        kill -TERM -- "-$server_pid" 2>/dev/null || true
    fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
setsid env ASCEND_RT_VISIBLE_DEVICES="$TARGET_NPUS" \
    python scripts/launch_vllm.py train "$MODEL" \
    --target-layer-ids 1 9 17 25 33 \
    --provenance-dir "$OUTPUT_DIR" -- \
    --data-parallel-size "${#target_devices[@]}" \
    --port "$PORT" --max-model-len "$((SEQ_LENGTH + 256))" \
    --gpu-memory-utilization 0.85 >"$OUTPUT_DIR/target_server.log" 2>&1 &
server_pid=$!
ready=0
for ((elapsed=0; elapsed<${BOOT_TIMEOUT:-1800}; elapsed+=5)); do
    if ! kill -0 "$server_pid" 2>/dev/null; then
        tail -n 80 "$OUTPUT_DIR/target_server.log"; exit 1
    fi
    if curl --noproxy '*' --connect-timeout 2 --max-time 5 -sf \
        "http://127.0.0.1:$PORT/v1/models" >/dev/null; then ready=1; break; fi
    sleep 5
done
if [[ "$ready" != 1 ]]; then
    tail -n 80 "$OUTPUT_DIR/target_server.log"; exit 1
fi
extra=()
if [[ -n "${MAX_STEPS:-}" ]]; then extra+=(--max-steps "$MAX_STEPS"); fi
set -o pipefail
ASCEND_RT_VISIBLE_DEVICES="$TRAIN_NPUS" torchrun --standalone \
    --nproc_per_node "${#train_devices[@]}" -m speculators.train \
    --verifier-name-or-path "$MODEL" --data-path "$DATASET" \
    --vllm-endpoint "http://127.0.0.1:$PORT/v1" \
    --save-path "$OUTPUT_DIR/checkpoints" \
    --speculator-type dflash --num-layers 5 \
    --full-attention-indices 0 1 2 3 4 --last-layer-causal \
    --draft-arch qwen3 --draft-hidden-act silu --draft-attn-impl sdpa \
    --draft-vocab-size 151936 --mask-token-id 151669 \
    --target-layer-ids 1 9 17 25 33 \
    --block-size "$BLOCK_SIZE" --max-anchors "$MAX_ANCHORS" \
    --total-seq-len "$SEQ_LENGTH" --epochs "$EPOCHS" --lr "$LR" \
    --checkpoint-freq 1 --scheduler-type cosine --seed "$SEED" \
    --logger tensorboard --log-dir "$OUTPUT_DIR/logs" \
    --run-name dflash_5layer_last_causal \
    --on-missing generate --on-generate delete \
    --request-timeout 180 --max-retries 8 --log-freq 10 \
    "${extra[@]}" 2>&1 | tee "$OUTPUT_DIR/train.log"
printf 'Training finished: %s\n' "$OUTPUT_DIR"
