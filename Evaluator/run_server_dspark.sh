#!/usr/bin/env bash
# DSpark defaults for the shared configurable fast launcher.
set -eo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export SPEC_MAIN="${SPEC_MAIN:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
export SPECULATIVE_METHOD="${SPECULATIVE_METHOD:-dspark}"
# Select a DSpark checkpoint with the training budget intended for this comparison.
# No epoch-1 default beside the epoch-5 DFlash/Prefix defaults.
export DRAFT_MODEL="${DRAFT_MODEL:-/home/n84449292/m84379596/DFlash/vLLM_NPU_spec_main/output/dspark_qwen3_4b_openperfectblend_bs16/dspark_qwen3_4b_openperfectblend/checkpoints/4}"
[[ -n "$DRAFT_MODEL" ]] || { echo "Set DRAFT_MODEL to your DSpark checkpoint." >&2; exit 2; }
export DRAFT_MODEL
export NUM_SPECULATIVE_TOKENS="${NUM_SPECULATIVE_TOKENS:-15}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3-4b-dspark}"
export NPU="${NPU:-6}"
export PORT="${PORT:-8214}"
export TARGET_EXECUTION="${TARGET_EXECUTION:-graph}"
# Environment paths and all other settings are editable in run_server_dflash.sh
# or can be supplied here as environment overrides.
# DRAFT_MODEL is required so an older training epoch is not silently selected.
exec bash "$SCRIPT_DIR/run_server_dflash.sh"
