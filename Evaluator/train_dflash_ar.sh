#!/usr/bin/env bash
# Run in the same activated Ascend/CANN environment as the original training.
set -euo pipefail
SPEC_MAIN="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
: "${BASELINE_CONFIG:?Set BASELINE_CONFIG to your original vanilla DFlash checkpoints/run.yaml}"
MODE="${MODE:-ar}"
OUTPUT_DIR="${OUTPUT_DIR:-$SPEC_MAIN/output/dflash_${MODE}_qwen3_4b_$(date +%Y%m%d_%H%M%S)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
VLLM_NPUS="${VLLM_NPUS:-0,1}"
TRAIN_NPUS="${TRAIN_NPUS:-2,3,4,5,6,7}"
NUM_TRAIN_NPUS="${NUM_TRAIN_NPUS:-6}"
VLLM_PORT="${VLLM_PORT:-8096}"
VLLM_RPC_PORT="${VLLM_RPC_PORT:-29696}"
export PYTHONPATH="$SPEC_MAIN/speculators/src:$SPEC_MAIN/speculators/hs_connectors/src:$SPEC_MAIN/vllm:$SPEC_MAIN/vllm-ascend:${PYTHONPATH:-}"
export TORCH_COMPILE_DISABLE=1 TORCHDYNAMO_DISABLE=1
export no_proxy='localhost,127.0.0.1,::1' NO_PROXY='localhost,127.0.0.1,::1'
cd "$SPEC_MAIN/speculators"
export VLLM_PORT VLLM_RPC_PORT VLLM_NPUS TRAIN_NPUS NUM_TRAIN_NPUS
"$PYTHON_BIN" - <<'PY_CHECK'
import os, socket
server = set(os.environ['VLLM_NPUS'].split(','))
trainer = os.environ['TRAIN_NPUS'].split(',')
if server.intersection(trainer) or len(trainer) != len(set(trainer)):
    raise SystemExit('Server/trainer devices must be disjoint and unique')
if len(trainer) != int(os.environ['NUM_TRAIN_NPUS']):
    raise SystemExit('NUM_TRAIN_NPUS must match TRAIN_NPUS')
ports = [int(os.environ[k]) for k in ('VLLM_PORT', 'VLLM_RPC_PORT')]
if len(set(ports)) != 2:
    raise SystemExit('Use distinct HTTP and RPC ports')
for port in ports:
    with socket.socket() as sock:
        sock.bind(('0.0.0.0', port))
PY_CHECK
extra=()
[[ -z "${MODEL:-}" ]] || extra+=(--target "$MODEL")
[[ -z "${DATASET:-}" ]] || extra+=(--dataset "$DATASET")
"$PYTHON_BIN" scripts/prepare_dflash_ar.py --baseline-config "$BASELINE_CONFIG" \
    --output "$OUTPUT_DIR" --mode "$MODE" --endpoint "http://127.0.0.1:$VLLM_PORT/v1" "${extra[@]}"
if [[ "${CHECK_ONLY:-0}" == 1 ]]; then
    echo "Review $OUTPUT_DIR/changes.json; use a fresh OUTPUT_DIR for training."
    exit 0
fi
MODEL="$(cat "$OUTPUT_DIR/target.txt")"
read -r -a layers < "$OUTPUT_DIR/target_layers.txt"
mkdir -p "$OUTPUT_DIR/hidden_states" "$OUTPUT_DIR/logs" "$OUTPUT_DIR/provenance"
setsid env ASCEND_RT_VISIBLE_DEVICES="$VLLM_NPUS" "$PYTHON_BIN" scripts/launch_vllm.py "$MODEL" \
    --target-layer-ids "${layers[@]}" --hidden-states-path "$OUTPUT_DIR/hidden_states" \
    --provenance-dir "$OUTPUT_DIR/provenance" -- \
    --served-model-name "$MODEL" --tensor-parallel-size 1 --data-parallel-size "${VLLM_DP:-2}" \
    --data-parallel-rpc-port "$VLLM_RPC_PORT" --port "$VLLM_PORT" \
    --max-model-len "${MAX_MODEL_LEN:-3328}" --gpu-memory-utilization 0.85 \
    > "$OUTPUT_DIR/logs/target.log" 2>&1 &
server_pid=$!
cleanup() { kill -TERM -- "-$server_pid" 2>/dev/null || true; }
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
deadline=$((SECONDS + 1800))
until curl --noproxy '*' --connect-timeout 3 --max-time 5 -fsS "http://127.0.0.1:$VLLM_PORT/v1/models" >/dev/null; do
    if ! kill -0 "$server_pid" 2>/dev/null || ((SECONDS >= deadline)); then
        tail -80 "$OUTPUT_DIR/logs/target.log"
        exit 1
    fi
    sleep 3
done
"$PYTHON_BIN" - "$MODEL" "$VLLM_PORT" <<'PY_ID'
import json, sys, urllib.request
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
with opener.open(f'http://127.0.0.1:{sys.argv[2]}/v1/models', timeout=15) as response:
    entries = json.load(response)['data']
if not entries or entries[0]['id'] != sys.argv[1]:
    raise SystemExit('Hidden-state server model identity does not match the trainer')
PY_ID
ASCEND_RT_VISIBLE_DEVICES="$TRAIN_NPUS" "$PYTHON_BIN" -m torch.distributed.run \
    --standalone --nproc_per_node "$NUM_TRAIN_NPUS" scripts/train_dflash_ar.py \
    --config "$OUTPUT_DIR/train_config.yaml" 2>&1 | tee "$OUTPUT_DIR/logs/train.log"
