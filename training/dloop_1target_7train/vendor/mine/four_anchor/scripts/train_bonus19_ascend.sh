#!/usr/bin/env bash
# Four processed target anchors (including the bonus) + fifteen masks.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
: "${CONDA_PREFIX:?Activate attention_prefix_ali first}"
export VLLM_SOURCE_ROOT="${VLLM_SOURCE_ROOT:-/home/n84449292/a84384644/attention_head_forked/four-anchor-dflash-pipeline}"
export SPECULATORS_ROOT="${SPECULATORS_ROOT:-$VLLM_SOURCE_ROOT/speculators}"
export MODEL="${MODEL:-/home/n84449292/m84379596/Huggingface/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c}"
export CANN_ROOT="${CANN_ROOT:-/home/n84449292/m84379596/CANN/9.1.0}"
export ANCHOR_LAYOUT=target_four BLOCK_SIZE=15
export TARGET_NPUS="${TARGET_NPUS:-0,1}" TRAIN_NPUS="${TRAIN_NPUS:-2,3,4,5,6}"
export EVAL_NPU="${EVAL_NPU:-7}"
export SEQ_LENGTH="${SEQ_LENGTH:-9216}" MAX_ANCHORS="${MAX_ANCHORS:-1536}"
export EPOCHS="${EPOCHS:-5}" TRAIN_DATA_RATIO="${TRAIN_DATA_RATIO:-1.0}"
if [[ -n "${RESUME:-}" && -z "${OUTPUT_DIR:-}" ]]; then
    export OUTPUT_DIR="$(cd -- "$RESUME/../.." && pwd)"
fi
export OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/output/bonus19_$(date +%Y%m%d_%H%M%S)_$$}"
export EPOCH_BENCHMARK_DIR="$OUTPUT_DIR/evaluation"
export PYTHONPATH="$ROOT:$SPECULATORS_ROOT/src:$SPECULATORS_ROOT/hs_connectors/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export PYTHONUNBUFFERED=1
cd "$ROOT"
python - <<'PY'
import os
import json
from pathlib import Path
import subprocess
parts = [os.environ['TARGET_NPUS'].split(','), os.environ['TRAIN_NPUS'].split(','), [os.environ['EVAL_NPU']]]
devices = sum(parts, [])
if not all(x.isdigit() for x in devices) or len(devices) != len(set(devices)):
    raise SystemExit('Target, trainers and evaluator need distinct physical NPU IDs')
dependencies = Path(os.environ['VLLM_SOURCE_ROOT'])
for name, revision in {
    'vllm': '77b1d88c5b52abaf6a103292119f3435b51699d9',
    'vllm-ascend': '276a5c840b21c83b153d3ebd831720bc926dfc4e',
}.items():
    actual = subprocess.check_output(['git', '-C', str(dependencies / name), 'rev-parse', 'HEAD'], text=True).strip()
    if actual != revision:
        raise SystemExit(f'{name}: expected {revision}, found {actual}; use the pinned working VLLM_SOURCE_ROOT.')
if not (Path(os.environ['SPECULATORS_ROOT']) / 'scripts/launch_vllm.py').is_file():
    raise SystemExit('SPECULATORS_ROOT must point to the existing initialized Speculators checkout.')
if os.environ.get('RESUME'):
    cfg = json.loads((Path(os.environ['RESUME']) / 'four_anchor_config.json').read_text())['draft_config']
    if (cfg.get('anchor_layout'), cfg.get('format_version'), cfg['num_proposals']) != ('target_four', 3, 15):
        raise SystemExit('Resume requires format-3 target_four training; old and embedding-only checkpoints are incompatible.')
print('Layout: [4 actual target-state anchors, bonus included] + 15 masks = 19 queries in both rounds')
print('Inference processes the pending bonus with the target before drafting; the extra forward is reported.')
print('Training NPUs:', parts[1], 'Evaluator NPU:', parts[2][0])
PY
if [[ "${CHECK_ONLY:-0}" == 1 ]]; then
    exec bash "$ROOT/four_anchor/scripts/train_ascend.sh"
fi
args=(--directory "$EPOCH_BENCHMARK_DIR" --root "$VLLM_SOURCE_ROOT" --code-root "$ROOT"
      --run "$OUTPUT_DIR" --model "$MODEL" --npu "$EVAL_NPU"
      --limit "${EVAL_LIMIT:-128}" --max-new-tokens "${EVAL_MAX_NEW_TOKENS:-2048}")
prepare=()
if [[ -n "${BENCHMARK_PLAN:-}" ]]; then prepare+=(--prompt-plan "$BENCHMARK_PLAN"); fi
# Tokenize/cache once before training. No target model or NPU is allocated here.
if [[ -n "${RESUME:-}" && -s "$EPOCH_BENCHMARK_DIR/prepared/plan.json" ]]; then
    python - "$EPOCH_BENCHMARK_DIR" "${EVAL_LIMIT:-128}" "${EVAL_MAX_NEW_TOKENS:-2048}" "$MODEL" <<'PY'
import json, sys
from pathlib import Path
folder = Path(sys.argv[1])
plan = json.loads((folder / 'prepared/plan.json').read_text())
if (plan['limit_per_benchmark'], plan['max_new_tokens'], Path(plan['target']).resolve()) != (
        int(sys.argv[2]), int(sys.argv[3]), Path(sys.argv[4]).resolve()):
    raise SystemExit('Resume benchmark settings differ from the frozen prompt plan.')
(folder / 'training_finished.json').unlink(missing_ok=True)
PY
else
    TORCH_DEVICE_BACKEND_AUTOLOAD=0 USE_TORCH=0 USE_TF=0 \
        python -m four_anchor.epoch_benchmarks prepare "${args[@]}" "${prepare[@]}"
fi
worker_pid=''
cleanup() {
    if [[ -n "$worker_pid" ]]; then
        kill -TERM -- "-$worker_pid" 2>/dev/null || true
        wait "$worker_pid" 2>/dev/null || true
    fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
setsid env TORCH_DEVICE_BACKEND_AUTOLOAD=0 USE_TORCH=0 USE_TF=0 \
    python -m four_anchor.epoch_benchmarks worker "${args[@]}" \
    > "$EPOCH_BENCHMARK_DIR/worker.log" 2>&1 &
worker_pid=$!
train_status=0
bash "$ROOT/four_anchor/scripts/train_ascend.sh" || train_status=$?
python -m four_anchor.epoch_benchmarks finish --directory "$EPOCH_BENCHMARK_DIR" --exit-code "$train_status"
echo "Training exited with status $train_status. Waiting for queued epoch benchmarks to finish."
eval_status=0
wait "$worker_pid" || eval_status=$?
worker_pid=''
printf 'Epoch results: %s/epoch_results.md\n' "$EPOCH_BENCHMARK_DIR"
if [[ "$train_status" != 0 ]]; then exit "$train_status"; fi
exit "$eval_status"
