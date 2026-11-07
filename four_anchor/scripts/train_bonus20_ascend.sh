#!/usr/bin/env bash
# Train four past target anchors + bonus embedding + fifteen masks; round two is draft-only.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
: "${CONDA_PREFIX:?Activate attention_prefix_ali first}"
# The installer records the cloned dependency checkout, never an old server path.
if [[ -z "${VLLM_SOURCE_ROOT:-}" && -f "$ROOT/bonus20_local_paths.json" ]]; then
    VLLM_SOURCE_ROOT="$(python - "$ROOT/bonus20_local_paths.json" <<'PY_PATH'
import json, sys
print(json.load(open(sys.argv[1]))['dependency_root'])
PY_PATH
)"
fi
export VLLM_SOURCE_ROOT="${VLLM_SOURCE_ROOT:-$ROOT}"
export SPECULATORS_ROOT="${SPECULATORS_ROOT:-$VLLM_SOURCE_ROOT/speculators}"
: "${MODEL:?Set MODEL to the local Qwen3-4B model directory on THIS server}"
: "${DATASET:?Set DATASET to the local preprocessed save_to_disk training dataset}"
export MODEL DATASET
export ANCHOR_LAYOUT=bonus_addon BLOCK_SIZE=15
export EVAL_EACH_EPOCH="${EVAL_EACH_EPOCH:-1}" EVAL_CONCURRENCY="${EVAL_CONCURRENCY:-32}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TARGET_NPUS="${TARGET_NPUS:-0,1}" TRAIN_NPUS="${TRAIN_NPUS:-2,3,4,5,6}"
export EVAL_NPU="${EVAL_NPU:-7}"
export SEQ_LENGTH="${SEQ_LENGTH:-9216}" MAX_ANCHORS="${MAX_ANCHORS:-1536}"
export EPOCHS="${EPOCHS:-5}" TRAIN_DATA_RATIO="${TRAIN_DATA_RATIO:-1.0}"
if [[ -n "${RESUME:-}" && -z "${OUTPUT_DIR:-}" ]]; then
    export OUTPUT_DIR="$(cd -- "$RESUME/../.." && pwd)"
fi
export OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/output/bonus20_train_$(date +%Y%m%d_%H%M%S)_$$}"
if [[ "$EVAL_EACH_EPOCH" == 1 ]]; then
    export EPOCH_BENCHMARK_DIR="$OUTPUT_DIR/evaluation"
else
    unset EPOCH_BENCHMARK_DIR
fi
export PYTHONPATH="$ROOT:$SPECULATORS_ROOT/src:$SPECULATORS_ROOT/hs_connectors/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
cd "$ROOT"
python - <<'PY'
import os
import json
from pathlib import Path
import subprocess
parts = [os.environ['TARGET_NPUS'].split(','), os.environ['TRAIN_NPUS'].split(','), [os.environ['EVAL_NPU']]]
if os.environ['EVAL_EACH_EPOCH'] not in {'0', '1'}:
    raise SystemExit('EVAL_EACH_EPOCH must be 0 or 1')
if not 1 <= int(os.environ['EVAL_CONCURRENCY']) <= 32:
    raise SystemExit('EVAL_CONCURRENCY must be 1..32')
devices = sum(parts if os.environ['EVAL_EACH_EPOCH'] == '1' else parts[:2], [])
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
    if (cfg.get('anchor_layout'), cfg.get('format_version'), cfg['num_proposals']) != ('bonus_addon', 4, 15):
        raise SystemExit('Resume requires format-4 bonus_addon training. Start a fresh run for this new objective.')
print('Round 1: 4 past target-state anchors + known bonus embedding + 15 masks = 20 queries')
print('Round 2: 4 first-pass drafter-state anchors + 15 masks = 19 queries; no bonus target forward')
print('Training NPUs:', parts[1], 'Evaluator NPU:', parts[2][0])
PY
if [[ "${CHECK_ONLY:-0}" == 1 || "$EVAL_EACH_EPOCH" == 0 ]]; then
    exec bash "$ROOT/four_anchor/scripts/train_ascend.sh"
fi
args=(--directory "$EPOCH_BENCHMARK_DIR" --root "$VLLM_SOURCE_ROOT" --code-root "$ROOT"
      --run "$OUTPUT_DIR" --model "$MODEL" --npu "$EVAL_NPU"
      --limit "${EVAL_LIMIT:-128}" --max-new-tokens "${EVAL_MAX_NEW_TOKENS:-2048}"
      --inference-layout bonus_addon_batch --concurrency "$EVAL_CONCURRENCY")
if [[ -n "${BENCHMARK_CACHE_DIR:-}" ]]; then args+=(--cache-dir "$BENCHMARK_CACHE_DIR"); fi
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
