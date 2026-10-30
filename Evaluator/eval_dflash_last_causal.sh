#!/usr/bin/env bash
# A dedicated local server, 15 draft tokens, greedy generation, concurrency 1.
set -eo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
: "${CONDA_PREFIX:?Activate attention_prefix_ali first}"
: "${NPU:?Set NPU to one free physical NPU index}"
PYTHON_BIN="${PYTHON_BIN:-$CONDA_PREFIX/bin/python}"
if [[ -z "${DRAFT_MODEL:-}" ]]; then
    DRAFT_MODEL=$("$PYTHON_BIN" - "$ROOT" <<'PY'
from pathlib import Path
import sys
root = Path(sys.argv[1])
candidates = sorted({p.parent.resolve() for p in root.glob('output/dflash_5layer_last_causal_*/checkpoints/0/config.json')})
if len(candidates) != 1:
    raise SystemExit('Set DRAFT_MODEL to your completed first checkpoint directory. Candidates:\n'
                     + ('\n'.join(map(str, candidates)) or '(none found under output/)'))
print(candidates[0])
PY
)
fi
CANN_ROOT="${CANN_ROOT:-/home/n84449292/m84379596/CANN/9.1.0}"
source "$CANN_ROOT/cann/set_env.sh"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
DFLASH_TORCH_ABI=$(TORCH_DEVICE_BACKEND_AUTOLOAD=0 "$PYTHON_BIN" -c 'import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))')
source "$CANN_ROOT/nnal/atb/set_env.sh" --cxx_abi="$DFLASH_TORCH_ABI"
export LD_LIBRARY_PATH="${ATB_HOME_PATH:?ATB environment missing}/lib:$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
set -u
export PYTHONPATH="$ROOT/speculators/src:$ROOT/speculators/hs_connectors/src:$ROOT/vllm:$ROOT/vllm-ascend:${PYTHONPATH:-}"
export ASCEND_RT_VISIBLE_DEVICES="$NPU"
export no_proxy="localhost,127.0.0.1,::1" NO_PROXY="localhost,127.0.0.1,::1"
unset TORCH_DEVICE_BACKEND_AUTOLOAD
TARGET_MODEL="${TARGET_MODEL:-${MODEL:-/home/n84449292/m84379596/Huggingface/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c}}"
PORT="${PORT:-8214}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
CHECK_ONLY="${CHECK_ONLY:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/output/eval_last_causal_$(date +%Y%m%d_%H%M%S)_$$}"
mkdir -p "$(dirname -- "$OUTPUT_DIR")"
mkdir "$OUTPUT_DIR"
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"
cd "$ROOT"

checks=(--checkpoint "$DRAFT_MODEL")
if [[ "$CHECK_ONLY" != 1 ]]; then checks+=(--npu-check); fi
"$PYTHON_BIN" Evaluator/check_dflash_last_causal_runtime.py "${checks[@]}" 2>&1 | tee "$OUTPUT_DIR/preflight.log"

server=("$PYTHON_BIN" "$ROOT/speculators/scripts/dflash_prefix_compiled/target_graph_server.py"
    --spec-main "$ROOT" --target "$TARGET_MODEL" --draft "$DRAFT_MODEL"
    --speculative-method dflash --num-speculative-tokens 15
    --target-execution eager --draft-enforce-eager 1
    --comparison-profile matched_reference --head-execution reference --mode reference
    --npu "$NPU" --port "$PORT" --served-model-name qwen3-4b-last-causal
    --seed 42 --kv-block-size 128 --max-model-len "$MAX_MODEL_LEN"
    --max-batched-tokens "$MAX_MODEL_LEN" --memory-utilization "${MEMORY_UTILIZATION:-0.90}"
    --graph-metrics 0 --run-dir "$OUTPUT_DIR/server" --register-evaluator 1)
if [[ "$CHECK_ONLY" == 1 ]]; then
    "${server[@]}" --check-only 1
    echo "Check-only passed; no benchmark or NPU kernel check was run. Records: $OUTPUT_DIR"
    exit 0
fi

server_pid=''
cleanup() {
    if [[ -n "$server_pid" ]]; then kill -TERM -- "-$server_pid" 2>/dev/null || true; fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
setsid "${server[@]}" >"$OUTPUT_DIR/bootstrap.log" 2>&1 &
server_pid=$!
echo "Server starting; log: $OUTPUT_DIR/server/server.log"
ready=0
for ((elapsed=0; elapsed<${BOOT_TIMEOUT:-1800}; elapsed+=5)); do
    if ! kill -0 "$server_pid" 2>/dev/null; then
        tail -n 100 "$OUTPUT_DIR/bootstrap.log"; exit 1
    fi
    if curl --noproxy '*' --connect-timeout 2 --max-time 5 -sf \
        "http://127.0.0.1:$PORT/v1/models" >/dev/null; then ready=1; break; fi
    sleep 5
done
if [[ "$ready" != 1 ]]; then tail -n 100 "$OUTPUT_DIR/bootstrap.log"; exit 1; fi

read -r -a datasets <<< "${BENCHMARKS:-gsm8k math500 humaneval mbpp lcb aime25 mtbench alpaca}"
"$PYTHON_BIN" "$ROOT/Evaluator/evaluator.py" \
    --base-url "http://127.0.0.1:$PORT" --model qwen3-4b-last-causal \
    --target "$TARGET_MODEL" --method dflash_5layer_last_causal_eager \
    --server-manifest "$OUTPUT_DIR/server/launch.json" \
    --datasets "${datasets[@]}" --num-prompts "${NUM_PROMPTS:-0}" \
    --manifest "${PROMPT_MANIFEST:-$ROOT/Evaluator/dflash_last_causal_prompts.json}" \
    --temperatures 0 --concurrencies 1 --top-p 1 --top-k -1 \
    --max-new-tokens "${MAX_NEW_TOKENS:-2048}" --seed 42 --prompt-seed 42 \
    --warmup-requests 1 --selector-warmup-tokens 128 --selector-warmup-steps 0 \
    --no-thinking --output "$OUTPUT_DIR/results" 2>&1 | tee "$OUTPUT_DIR/evaluator.log"
echo "Results: $OUTPUT_DIR/results/summary.csv"
