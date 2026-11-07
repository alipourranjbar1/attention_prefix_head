#!/usr/bin/env bash
# Native vLLM target + in-worker four-anchor proposer. Default: first checkpoint.
set -eo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
: "${CONDA_PREFIX:?Activate attention_prefix_ali first}"
RUN="${RUN:-$ROOT/output/four_anchor_dflash_20261031_025300_2111006}"
MODEL="${MODEL:-/home/n84449292/m84379596/Huggingface/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c}"
DRAFT="${DRAFT:-$RUN/checkpoints/checkpoint-016699}"
VLLM_SOURCE_ROOT="${VLLM_SOURCE_ROOT:-$ROOT}"
if [[ -n "${CANN_ROOT:-}" ]]; then
if [[ -f "$CANN_ROOT/cann/set_env.sh" ]]; then
    source "$CANN_ROOT/cann/set_env.sh"
else
    source "$CANN_ROOT/ascend-toolkit/set_env.sh"
fi
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
DFLASH_TORCH_ABI=$(TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -c 'import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))')
source "$CANN_ROOT/nnal/atb/set_env.sh" --cxx_abi="$DFLASH_TORCH_ABI"
fi
set -u
SPECULATORS_ROOT="${SPECULATORS_ROOT:-$VLLM_SOURCE_ROOT/speculators}"
export PYTHONPATH="$ROOT:$SPECULATORS_ROOT/src:$SPECULATORS_ROOT/hs_connectors/src:$VLLM_SOURCE_ROOT/vllm:$VLLM_SOURCE_ROOT/vllm-ascend:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${ATB_HOME_PATH:?ATB environment missing}/lib:$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export ASCEND_RT_VISIBLE_DEVICES="${EVAL_NPU:-7}"
export VLLM_USE_V2_MODEL_RUNNER=0 TORCH_COMPILE_DISABLE=1 TORCHDYNAMO_DISABLE=1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
export no_proxy="localhost,127.0.0.1,::1" NO_PROXY="localhost,127.0.0.1,::1"
export SOC_VERSION="${SOC_VERSION:-ascend910_9372}"
unset ASCEND_LAUNCH_BLOCKING DFLASH_TP_GATHER
python - "$VLLM_SOURCE_ROOT" <<'PY'
from pathlib import Path
import subprocess, sys
root = Path(sys.argv[1])
expected = {'vllm': '77b1d88c5b52abaf6a103292119f3435b51699d9',
            'vllm-ascend': '276a5c840b21c83b153d3ebd831720bc926dfc4e'}
for name, commit in expected.items():
    actual = subprocess.check_output(['git', '-C', str(root / name), 'rev-parse', 'HEAD'], text=True).strip()
    if actual != commit:
        raise SystemExit(f'{name}: expected pinned revision {commit}, found {actual}. '
                         'Set VLLM_SOURCE_ROOT to a checkout with the pinned submodules.')
print('Pinned vLLM/Ascend source revisions verified', flush=True)
PY
command="${1:-evaluate}"
if [[ "${TRACE_STEPS:-0}" != 0 && "${TRACE_STEPS:-0}" != 1 ]]; then
    echo "TRACE_STEPS must be 0 or 1" >&2; exit 2
fi
args=("$command" --target "$MODEL" --checkpoint "$DRAFT" --mode "${MODE:-two_pass}"
      --anchor-policy bonus_addon --concurrency "${CONCURRENCY:-32}"
      --proposals "${NUM_SPECULATIVE_TOKENS:-15}" --max-model-len "${MAX_MODEL_LEN:-4096}"
      --memory-fraction "${MEMORY_FRACTION:-0.8}")
if [[ "${DRAFT_GRAPH:-0}" != 0 && "${DRAFT_GRAPH:-0}" != 1 ]]; then
    echo "DRAFT_GRAPH must be 0 or 1" >&2; exit 2
fi
args+=(--profile-steps "${PROFILE_STEPS:-0}")
if [[ "${DRAFT_GRAPH:-0}" == 1 ]]; then args+=(--draft-graph); fi
if [[ "${TRACE_STEPS:-0}" == 1 ]]; then args+=(--trace-steps); fi
if [[ "$command" == evaluate ]]; then
    DATA="${DATA:-$RUN/evaluation/epoch1_gsm8k_20261102_230727/prompts.jsonl}"
    OUTPUT_DIR="${OUTPUT_DIR:-$RUN/evaluation/vllm_${MODE:-two_pass}_$(date +%Y%m%d_%H%M%S)_$$}"
    test -s "$DATA"
    args+=(--data "$DATA" --output "$OUTPUT_DIR" --limit "${LIMIT:-128}"
           --max-new-tokens "${MAX_NEW_TOKENS:-2048}" --warmup "${WARMUP:-2}")
    if [[ -n "${BASELINE:-}" ]]; then args+=(--baseline "$BASELINE"); fi
    mkdir -p "$OUTPUT_DIR"
    python -m four_anchor.vllm_batch_entry "${args[@]}" 2>&1 | tee "$OUTPUT_DIR/eval.log"
    printf 'Results: %s/summary.json\n' "$OUTPUT_DIR"
else
    args+=(--host "${HOST:-127.0.0.1}" --port "${PORT:-8095}")
    exec python -m four_anchor.vllm_batch_entry "${args[@]}"
fi
