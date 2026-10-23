#!/usr/bin/env bash
# DFlash-Prefix defaults for the same fast launcher used by vanilla and DSpark.
set -eo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export SPEC_MAIN="${SPEC_MAIN:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
export SPECULATIVE_METHOD="${SPECULATIVE_METHOD:-dflash}"
export DRAFT_MODEL="${DRAFT_MODEL:-/home/n84449292/m84379596/DFlash/vLLM_NPU_spec_main/output/dflash_prefix_compiled_joint_bs16/joint_memory_fast_20261009_041643/checkpoints/4}"
export NUM_SPECULATIVE_TOKENS="${NUM_SPECULATIVE_TOKENS:-15}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3-4b-prefix-dflash}"
export NPU="${NPU:-4}"
export PORT="${PORT:-8216}"
export TARGET_EXECUTION="${TARGET_EXECUTION:-graph}"
exec bash "$SCRIPT_DIR/run_server_dflash.sh"
