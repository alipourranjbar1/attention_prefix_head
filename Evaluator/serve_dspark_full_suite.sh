#!/usr/bin/env bash
# Prepare full cached splits once, then serve OFF or confidence-cut ON.
set -eo pipefail
MODE="${1:-off}"
case "$MODE" in off|cut) ;; *) echo 'Usage: bash serve_dspark_full_suite.sh off|cut'; exit 2 ;; esac
ROOT="${ROOT:-/home/n84449292/a84384644/attention_head_forked/attention_prefix_head}"
: "${CONDA_PREFIX:?Activate attention_prefix_ali first}"
export SPEC_MAIN="$ROOT" ENV_ROOT="$CONDA_PREFIX" PYTHON_BIN="$CONDA_PREFIX/bin/python"
export CANN_ROOT="${CANN_ROOT:-/home/n84449292/a84384644/CANN/9.1.0}"
export TARGET_MODEL="${TARGET_MODEL:-/home/n84449292/a84384644/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c}"
export DRAFT_MODEL="${DRAFT_MODEL:-/share/canada_group_folder/ckpt/dspark_qwen3_4b_block7}"
CONCURRENCY="${CONCURRENCY:-64}"
[[ "$CONCURRENCY" =~ ^[1-9][0-9]*$ ]] || { echo 'CONCURRENCY must be a positive integer'; exit 2; }
SUITE_DIR="${SUITE_DIR:-$ROOT/output/dspark_full_benchmark_c${CONCURRENCY}}"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
if [[ -r "$CONDA_PREFIX/lib/libstdc++.so.6" ]]; then
    export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"
fi
PREP_ARGS=()
if [[ -z "${REUSE_MANIFEST:-}" ]]; then
    REUSE_MANIFEST="$ROOT/output/dspark_full_benchmark_c64/prompts.json"
    if [[ ! -f "$REUSE_MANIFEST" ]]; then
        REUSE_MANIFEST="$ROOT/Evaluator/retrace_eval_prompts.json"
    fi
fi
if [[ -f "$REUSE_MANIFEST" ]]; then
    PREP_ARGS+=(--reuse-manifest "$REUSE_MANIFEST")
fi
"$PYTHON_BIN" "$ROOT/Evaluator/dspark_benchmark_suite.py" prepare \
    --suite "$SUITE_DIR" --target "$TARGET_MODEL" --draft "$DRAFT_MODEL" \
    --repeats "${REPEATS:-3}" --max-new-tokens "${MAX_NEW_TOKENS:-2048}" \
    --concurrency "$CONCURRENCY" \
    "${PREP_ARGS[@]}"
MAX_MODEL_LEN="$("$PYTHON_BIN" -c 'import json,sys; print(json.load(open(sys.argv[1]))["max_model_len"])' "$SUITE_DIR/suite.json")"
export MAX_MODEL_LEN MAX_BATCHED_TOKENS="$MAX_MODEL_LEN"
export NPU="${NPU:-7}" PORT="${PORT:-8217}" NUM_SPECULATIVE_TOKENS=7 MAX_NUM_SEQS="$CONCURRENCY"
export MEMORY_UTILIZATION=0.5 TARGET_EXECUTION=eager HEAD_EXECUTION=reference
export SERVED_MODEL_NAME=qwen3-dspark-confidence DSPARK_CONFIDENCE_MODE="$MODE"
export CHECK_ONLY="${CHECK_ONLY:-0}"
unset DSPARK_CONFIDENCE_TRACE_DIR
export RUN_DIR="$SUITE_DIR/server_${MODE}_$(date +%Y%m%d_%H%M%S_%N)_$$"
printf 'Serving full suite: mode=%s, K=7, capacity=%s, trace=off. Keep this terminal open.\n' "$MODE" "$CONCURRENCY"
exec bash "$ROOT/Evaluator/run_server_dspark_confidence.sh"
