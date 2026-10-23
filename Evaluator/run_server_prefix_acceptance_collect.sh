#!/usr/bin/env bash
# Collect full-block verification records for acceptance-head training.
set -eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export SPEC_MAIN="$(cd -- "$SCRIPT_DIR/.." && pwd)"

# Use the currently activated Python environment.
export ENV_ROOT="${ENV_ROOT:-${CONDA_PREFIX:-/home/n84449292/a84384644/attention_modif}}"
export PYTHON_BIN="${PYTHON_BIN:-$ENV_ROOT/bin/python}"

# Change this if your trained checkpoint is elsewhere.
export DRAFT_MODEL="${DRAFT_MODEL:-$SPEC_MAIN/output/dflash_prefix_compiled_joint_bs16/joint_memory_fast_20261009_041643/checkpoints/4}"

# TARGET_MODEL and CANN paths are inherited from your existing launcher.
# Export TARGET_MODEL / CANN_ROOT before running if you need to override them.

export SPECULATIVE_METHOD="dflash"
export NUM_SPECULATIVE_TOKENS="${NUM_SPECULATIVE_TOKENS:-15}"
export SERVED_MODEL_NAME="qwen3-4b-prefix-dflash"
export NPU="${NPU:-4}"
export PORT="${PORT:-8113}"

# Collection configuration: eager execution and complete draft blocks.
export TARGET_EXECUTION="eager"
export DRAFT_ENFORCE_EAGER="1"
export COMPARISON_PROFILE="custom"
export HEAD_EXECUTION="reference"
export MODE="reference"

export VLLM_ASCEND_PREFIX_ACCEPTANCE_TRACE_DIR="$SPEC_MAIN/output/acceptance_experiment/traces"

# Remove configuration belonging to the earlier patch implementation.
unset ACCEPTANCE_CONFIG
unset VLLM_ASCEND_PREFIX_ACCEPTANCE_CONFIG

if [[ ! -f "$DRAFT_MODEL/config.json" ]]; then
    echo "Draft checkpoint not found: $DRAFT_MODEL/config.json" >&2
    echo "Set DRAFT_MODEL to your trained checkpoint directory." >&2
    exit 1
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python executable not found: $PYTHON_BIN" >&2
    exit 1
fi

mkdir -p "$VLLM_ASCEND_PREFIX_ACCEPTANCE_TRACE_DIR"

echo "Draft checkpoint: $DRAFT_MODEL"
echo "Trace directory:  $VLLM_ASCEND_PREFIX_ACCEPTANCE_TRACE_DIR"
echo "NPU: $NPU"
echo "Port: $PORT"

exec bash "$SCRIPT_DIR/run_server_dflash.sh"