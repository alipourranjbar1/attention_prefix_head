#!/usr/bin/env bash
# Foreground server: off = full verification; cut = confidence-selected verification.
set -eo pipefail
MODE="${1:-off}"
case "$MODE" in off|cut) ;; *) echo 'Usage: bash run_dspark_speed_server.sh off|cut'; exit 2 ;; esac
ROOT="${ROOT:-/home/n84449292/a84384644/attention_head_forked/attention_prefix_head}"
: "${CONDA_PREFIX:?Activate attention_prefix_ali first}"
export SPEC_MAIN="$ROOT" ENV_ROOT="$CONDA_PREFIX" PYTHON_BIN="$CONDA_PREFIX/bin/python"
export CANN_ROOT="${CANN_ROOT:-/home/n84449292/a84384644/CANN/9.1.0}"
export TARGET_MODEL="${TARGET_MODEL:-/home/n84449292/a84384644/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c}"
export DRAFT_MODEL="${DRAFT_MODEL:-/share/canada_group_folder/ckpt/dspark_qwen3_4b_block7}"
export NPU="${NPU:-7}" PORT="${PORT:-8217}"
export NUM_SPECULATIVE_TOKENS=7 MAX_NUM_SEQS=64
export MAX_MODEL_LEN=5120 MAX_BATCHED_TOKENS=5120 MEMORY_UTILIZATION=0.5
export TARGET_EXECUTION=eager HEAD_EXECUTION=reference
export SERVED_MODEL_NAME=qwen3-dspark-confidence
export DSPARK_CONFIDENCE_MODE="$MODE"
export CHECK_ONLY="${CHECK_ONLY:-0}"
unset DSPARK_CONFIDENCE_TRACE_DIR
export RUN_DIR="$ROOT/output/dspark_c64_speed_${MODE}_$(date +%Y%m%d_%H%M%S_%N)_$$"
# The launcher creates RUN_DIR and server.log itself. Do not precreate either.
printf 'Speed server: confidence=%s, proposals=7, capacity=64, tracing=off\n' "$MODE"
exec bash "$ROOT/Evaluator/run_server_dspark_confidence.sh"
