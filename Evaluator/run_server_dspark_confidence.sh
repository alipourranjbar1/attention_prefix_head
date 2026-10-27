#!/usr/bin/env bash
# Same DSpark checkpoint/Markov proposal path; choose whether confidence cuts verification.
set -eo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export SPEC_MAIN="${SPEC_MAIN:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
: "${DRAFT_MODEL:?Set DRAFT_MODEL to your trained DSpark checkpoint directory}"
: "${TARGET_MODEL:?Set TARGET_MODEL to the matching Qwen3 target directory}"
export DRAFT_MODEL TARGET_MODEL
export ENV_ROOT="${ENV_ROOT:-${CONDA_PREFIX:?Activate your Ascend environment}}"
export CANN_ROOT="${CANN_ROOT:-/home/n84449292/a84384644/CANN/9.1.0}"
export DSPARK_CONFIDENCE_MODE="${DSPARK_CONFIDENCE_MODE:-observe}"
export SPECULATIVE_METHOD=dspark
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
export TARGET_EXECUTION="${TARGET_EXECUTION:-eager}"
# Keep the Markov path identical in every mode, including the baseline.
export DRAFT_ENFORCE_EAGER=1
export HEAD_EXECUTION="${HEAD_EXECUTION:-reference}"
if [[ "$HEAD_EXECUTION" == reference ]]; then
    export COMPARISON_PROFILE=matched_reference
else
    export COMPARISON_PROFILE=matched_fast
fi
export NUM_SPECULATIVE_TOKENS="${NUM_SPECULATIVE_TOKENS:-15}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-6400}" MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-6400}"
export MEMORY_UTILIZATION="${MEMORY_UTILIZATION:-0.85}"
export NPU="${NPU:-6}" PORT="${PORT:-8217}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3-dspark-confidence}"
export RUN_DIR="${RUN_DIR:-$SPEC_MAIN/output/dspark_confidence_${DSPARK_CONFIDENCE_MODE}_$(date +%Y%m%d_%H%M%S)_$$}"
export LINEAR_CONFIDENCE=0
unset ACCEPTANCE_CONFIG VLLM_ASCEND_PREFIX_ACCEPTANCE_CONFIG VLLM_ASCEND_PREFIX_ACCEPTANCE_TRACE_DIR
unset VLLM_ASCEND_PREFIX_LINEAR_CONFIDENCE_HEAD TORCH_DEVICE_BACKEND_AUTOLOAD
export VLLM_LOGGING_LEVEL=INFO
exec bash "$SCRIPT_DIR/run_server_dflash.sh"
