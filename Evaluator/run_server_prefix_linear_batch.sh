#!/usr/bin/env bash
# Batched prefix proposals with optional trained acceptance-length scheduling.
set -eo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export SPEC_MAIN="$(cd -- "$SCRIPT_DIR/.." && pwd)"
export ENV_ROOT="${ENV_ROOT:-/home/n84449292/.conda/envs/attention_prefix_ali}"
export PYTHON_BIN="$ENV_ROOT/bin/python"
export CANN_ROOT="${CANN_ROOT:-/home/n84449292/a84384644/CANN/9.1.0}"
export TARGET_MODEL="${TARGET_MODEL:-/home/n84449292/m84379596/Huggingface/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c}"
export DRAFT_MODEL="${DRAFT_MODEL:-/home/n84449292/a84384644/prefix_drafter_check/checkpoints/4}"
export LINEAR_CONFIDENCE="${LINEAR_CONFIDENCE:-1}"
export LINEAR_CONFIDENCE_HEAD="${LINEAR_CONFIDENCE_HEAD:-/home/n84449292/a84384644/linear_head_selector/checkpoints/final.pt}"
export LINEAR_CONFIDENCE_POLICY="${LINEAR_CONFIDENCE_POLICY:-argmax}"
export LINEAR_CONFIDENCE_BUDGET="${LINEAR_CONFIDENCE_BUDGET:-7}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-128}"
export NPU="${NPU:-0}"
export PORT="${PORT:-8216}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3-4b-prefix-dflash}"
export NUM_SPECULATIVE_TOKENS=15
export SPECULATIVE_METHOD=dflash
export TARGET_EXECUTION=eager DRAFT_ENFORCE_EAGER=1
export HEAD_EXECUTION=reference COMPARISON_PROFILE=matched_reference
export VLLM_LOGGING_LEVEL=INFO GRAPH_METRICS=0
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
export MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-32768}"
export MEMORY_UTILIZATION="${MEMORY_UTILIZATION:-0.96}"
export RUN_DIR="${RUN_DIR:-$SPEC_MAIN/output/linear_batch${MAX_NUM_SEQS}_${LINEAR_CONFIDENCE_POLICY}_$(date +%Y%m%d_%H%M%S)_$$}"
unset TORCH_DEVICE_BACKEND_AUTOLOAD ACCEPTANCE_CONFIG
unset VLLM_ASCEND_PREFIX_ACCEPTANCE_CONFIG VLLM_ASCEND_PREFIX_ACCEPTANCE_TRACE_DIR
unset VLLM_ASCEND_PREFIX_LINEAR_CONFIDENCE_HEAD
export NO_PROXY=localhost,127.0.0.1,::1
export no_proxy="$NO_PROXY"
echo "Linear enabled=$LINEAR_CONFIDENCE policy=$LINEAR_CONFIDENCE_POLICY max_num_seqs=$MAX_NUM_SEQS"
echo "Log: $RUN_DIR/server.log"
exec bash "$SCRIPT_DIR/run_server_prefix_dflash.sh"
