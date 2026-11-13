#!/usr/bin/env bash
# DLoop from scratch: per-NPU budget 9216 tokens, <=768 sampled block starts.
set -eo pipefail
PACKAGE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "${CONDA_PREFIX:-}" ]]; then
    source /home/n84449292/miniforge3/etc/profile.d/conda.sh
    conda activate four_anchor_ali
fi
export INIT_MODE="${INIT_MODE:-scratch}"
export PACKED_TOKEN_BUDGET="${PACKED_TOKEN_BUDGET:-9216}"
export MAX_LENGTH="${MAX_LENGTH:-3072}"
export NUM_ANCHORS="${NUM_ANCHORS:-768}"
export EPOCHS="${EPOCHS:-1}" LR="${LR:-0.0006}"
export HEAD_LOSS_CHUNK_SIZE="${HEAD_LOSS_CHUNK_SIZE:-128}"
export ACTIVATION_CHECKPOINTING="${ACTIVATION_CHECKPOINTING:-1}"
export NUM_WORKERS="${NUM_WORKERS:-4}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-500}" LOG_INTERVAL="${LOG_INTERVAL:-10}"
export NPUS="${NPUS:-0,1,2,3,4,5,6,7}"
export OUTPUT_DIR="${OUTPUT_DIR:-$PACKAGE_ROOT/output/dloop_packed_$(date +%Y%m%d_%H%M%S)_$$}"
if [[ "$PACKED_TOKEN_BUDGET" == 0 ]]; then
    echo 'This entry point requires packing; PACKED_TOKEN_BUDGET must be positive.' >&2
    exit 2
fi
printf 'DLoop packing: <=%s tokens/rank, <=%s starts/pack; max record %s tokens.\n' \
    "$PACKED_TOKEN_BUDGET" "$NUM_ANCHORS" "$MAX_LENGTH"
exec bash "$PACKAGE_ROOT/train_ascend.sh"
