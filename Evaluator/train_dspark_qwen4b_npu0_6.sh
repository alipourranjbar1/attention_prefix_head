#!/usr/bin/env bash
set -eo pipefail

ROOT="${ROOT:-/home/n84449292/a84384644/attention_head_forked/attention_prefix_head}"
: "${CONDA_PREFIX:?Activate attention_prefix_ali first}"

CANN_ROOT="${CANN_ROOT:-/home/n84449292/a84384644/CANN/9.1.0}"
source "$CANN_ROOT/ascend-toolkit/set_env.sh"
source "$CANN_ROOT/nnal/atb/set_env.sh"
set -u

PYTHON_BIN="$CONDA_PREFIX/bin/python"
CXX_LIB="$CONDA_PREFIX/lib/libstdc++.so.6"
[[ -r "$CXX_LIB" ]] || {
    echo "Missing Conda C++ library: $CXX_LIB"
    exit 1
}

export LD_PRELOAD="$CXX_LIB${LD_PRELOAD:+:$LD_PRELOAD}"
export PYTHONPATH="$ROOT/speculators/src:$ROOT/speculators/hs_connectors/src:$ROOT/vllm:$ROOT/vllm-ascend:${PYTHONPATH:-}"
export NO_PROXY=localhost,127.0.0.1,::1
export no_proxy="$NO_PROXY"
export USE_TORCH=1 USE_TF=0
export TORCH_COMPILE_DISABLE=1 TORCHDYNAMO_DISABLE=1
unset TORCH_DEVICE_BACKEND_AUTOLOAD
unset DFLASH_TP_GATHER

# Fixed physical-device allocation.
SERVER_NPUS=0,1
TRAIN_NPUS=2,3,4,5,6
TRAIN_WORKERS=5

MODEL="${MODEL:-/home/n84449292/a84384644/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c}"
DATASET="${DATASET:-/home/n84449292/a84384644/attention_head_forked/dataset/open_perfectblend_full.qwen3.seq3072}"

EPOCHS="${EPOCHS:-5}"
LR="${LR:-0.0006}"
SEQ_LENGTH="${SEQ_LENGTH:-3072}"
BLOCK_SIZE="${BLOCK_SIZE:-7}"
MAX_ANCHORS="${MAX_ANCHORS:-512}"

VLLM_PORT="${VLLM_PORT:-8098}"
VLLM_RPC_PORT="${VLLM_RPC_PORT:-29698}"
MAX_MODEL_LEN=$((SEQ_LENGTH + 256))
LAYERS=(1 9 17 25 33)

RUN="${OUTPUT_DIR:-$ROOT/output/dspark_qwen3_4b_base_$(date +%Y%m%d_%H%M%S)_$$}"
export ROOT MODEL DATASET VLLM_PORT VLLM_RPC_PORT

cd "$ROOT"
printf 'Branch: %s\n' "$(git branch --show-current)"

# Check paths, Qwen3-4B configuration, prepared data and ports.
"$PYTHON_BIN" - <<'PY'
import json
import os
import socket
from pathlib import Path

model = Path(os.environ["MODEL"])
data = Path(os.environ["DATASET"])
cfg = json.loads((model / "config.json").read_text())
cfg = cfg.get("text_config", cfg)

if (
    cfg.get("model_type") != "qwen3"
    or cfg.get("hidden_size") != 2560
    or cfg.get("vocab_size") != 151936
):
    raise SystemExit("MODEL must point to the expected Qwen3-4B checkpoint")

info = json.loads((data / "dataset_info.json").read_text())
if not (data / "state.json").is_file():
    raise SystemExit("DATASET must be an existing save_to_disk dataset")
if not {"input_ids", "loss_mask"} <= set(info.get("features", {})):
    raise SystemExit("Prepared dataset requires input_ids and loss_mask")

ports = [int(os.environ[k]) for k in ("VLLM_PORT", "VLLM_RPC_PORT")]
if len(set(ports)) != len(ports):
    raise SystemExit("HTTP and DP RPC ports must differ")
for port in ports:
    with socket.socket() as sock:
        try:
            sock.bind(("127.0.0.1", port))
        except OSError as exc:
            raise SystemExit(f"Port {port} is unavailable: {exc}")

print("Model, dataset and ports checked")
PY

# Refuse to reuse an existing run directory.
mkdir -p "$(dirname "$RUN")"
mkdir "$RUN"
mkdir -p "$RUN/logs" "$RUN/checkpoints" "$RUN/hidden_states" "$RUN/provenance"
cp -- "${BASH_SOURCE[0]}" "$RUN/provenance/launcher.sh"

TRAIN_ARGS=(
    --speculator-type dspark
    --verifier-name-or-path "$MODEL"
    --data-path "$DATASET"
    --vllm-endpoint "http://127.0.0.1:$VLLM_PORT/v1"
    --hidden-states-backend file
    --hidden-states-path "$RUN/hidden_states"
    --hidden-states-dtype bfloat16
    --on-missing generate
    --on-generate delete
    --request-timeout 180
    --max-retries 8

    --num-layers 5
    --draft-arch qwen3
    --draft-hidden-act silu
    --draft-attn-impl sdpa
    --draft-vocab-size 151936
    --mask-token-id 151669
    --target-layer-ids "${LAYERS[@]}"

    --block-size "$BLOCK_SIZE"
    --sample-from-anchor
    --max-anchors "$MAX_ANCHORS"
    --total-seq-len "$SEQ_LENGTH"

    --markov-rank 256
    --markov-head-type vanilla
    --enable-confidence-head
    --confidence-head-with-markov
    --confidence-head-alpha 1.0
    --loss-fn '{"ce":0.1,"tv":0.9}'
    --per-position-loss-weight fixed-exp-decay

    --optimizer muon
    --lr "$LR"
    --scheduler-type cosine
    --epochs "$EPOCHS"
    --checkpoint-freq 1
    --no-resume-from-checkpoint
    --save-path "$RUN/checkpoints"
    --num-workers "${NUM_WORKERS:-12}"
    --seed 42

    --logger tensorboard
    --log-dir "$RUN/logs"
    --run-name "$(basename "$RUN")"
    --log-freq 10
)

# Validate against this branch's real parser and save the resolved config.
ASCEND_RT_VISIBLE_DEVICES=2 "$PYTHON_BIN" - \
    "$RUN/train_config.yaml" "${TRAIN_ARGS[@]}" <<'PY'
import sys
from pathlib import Path

import sqlite3
import torch
import torch_npu
from speculators.train.config import TrainConfig

cfg = TrainConfig.resolve(sys.argv[2:])
Path(sys.argv[1]).write_text(cfg.dump_yaml())
print("Imports and training configuration checked")
PY

printf '\nRun: %s\nServing NPUs: %s\nTraining NPUs: %s (%s workers)\n' \
    "$RUN" "$SERVER_NPUS" "$TRAIN_NPUS" "$TRAIN_WORKERS"

if [[ "${CHECK_ONLY:-0}" == 1 ]]; then
    echo "Preflight complete; server and training were not started."
    exit 0
fi

SERVER_PID=""
cleanup() {
    if [[ -n "$SERVER_PID" ]]; then
        kill -TERM -- "-$SERVER_PID" 2>/dev/null || true
        for ((i=0; i<10; i++)); do
            if ! kill -0 -- "-$SERVER_PID" 2>/dev/null; then
                break
            fi
            sleep 1
        done
        kill -KILL -- "-$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

SERVER_LOG="$RUN/logs/hidden_states_server.log"

setsid env ASCEND_RT_VISIBLE_DEVICES="$SERVER_NPUS" \
    "$PYTHON_BIN" "$ROOT/speculators/scripts/launch_vllm.py" train "$MODEL" \
    --target-layer-ids "${LAYERS[@]}" \
    --hidden-states-path "$RUN/hidden_states" \
    --provenance-dir "$RUN/provenance/server" \
    -- \
    --host 127.0.0.1 \
    --port "$VLLM_PORT" \
    --served-model-name "$MODEL" \
    --data-parallel-size 2 \
    --data-parallel-rpc-port "$VLLM_RPC_PORT" \
    --tensor-parallel-size 1 \
    --pipeline-parallel-size 1 \
    --dtype bfloat16 \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-batched-tokens "$MAX_MODEL_LEN" \
    --max-num-seqs 64 \
    --gpu-memory-utilization 0.85 \
    --no-enable-prefix-caching \
    --enforce-eager \
    >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!

echo "Waiting for Qwen3-4B server. Log: $SERVER_LOG"
deadline=$((SECONDS + ${VLLM_BOOT_TIMEOUT:-1800}))
next_notice=$((SECONDS + 30))

until curl --noproxy '*' --connect-timeout 3 --max-time 5 -fsS \
    "http://127.0.0.1:$VLLM_PORT/v1/models" >/dev/null 2>&1; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null || ((SECONDS >= deadline)); then
        echo "Server exited or startup timed out."
        tail -n 100 "$SERVER_LOG"
        exit 1
    fi
    if ((SECONDS >= next_notice)); then
        echo "Still waiting; see $SERVER_LOG"
        next_notice=$((SECONDS + 30))
    fi
    sleep 3
done

# Confirm the trainer will contact the intended target model.
"$PYTHON_BIN" - <<'PY'
import json
import os
import urllib.request

opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
url = f"http://127.0.0.1:{os.environ['VLLM_PORT']}/v1/models"
with opener.open(url, timeout=15) as response:
    entries = json.load(response).get("data", [])
if not entries or entries[0].get("id") != os.environ["MODEL"]:
    raise SystemExit("Server model ID does not match the training target")
print("Qwen3-4B server ready")
PY

echo "Starting five training workers. Log: $RUN/logs/train.log"

ASCEND_RT_VISIBLE_DEVICES="$TRAIN_NPUS" \
    "$PYTHON_BIN" -m torch.distributed.run \
    --standalone \
    --nproc_per_node "$TRAIN_WORKERS" \
    -m speculators.train \
    --config "$RUN/train_config.yaml" \
    2>&1 | tee "$RUN/logs/train.log"

printf '\nTraining completed. Checkpoints: %s/checkpoints\n' "$RUN"