#!/usr/bin/env bash
# Isolated inference only. The training tree, queue and checkpoint are read-only.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
FORK="${FORK:-/home/n84449292/a84384644/attention_head_forked}"
TRAIN_ROOT="${TRAIN_ROOT:-$FORK/four-anchor-15-train-benchmarks}"
RUN="${RUN:-$TRAIN_ROOT/output/target19_20261104_024656_3124841}"
export DRAFT="${DRAFT:-$RUN/checkpoints/checkpoint-022309}"
export MODEL="${MODEL:-/home/n84449292/m84379596/Huggingface/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c}"
export VLLM_SOURCE_ROOT="${VLLM_SOURCE_ROOT:-$FORK/four-anchor-dflash-pipeline}"
export SPECULATORS_ROOT="$VLLM_SOURCE_ROOT/speculators"
export CONCURRENCY="${CONCURRENCY:-32}"
export EVAL_NPU="${EVAL_NPU:-7}"
export ANCHOR_POLICY=bonus_addon NUM_SPECULATIVE_TOKENS=15 DRAFT_GRAPH=0 PROFILE_STEPS=0
export TRACE_STEPS=0 MODE=two_pass
WITH_BASELINE="${WITH_BASELINE:-0}"
[[ "$WITH_BASELINE" == 0 || "$WITH_BASELINE" == 1 ]] || { echo "WITH_BASELINE must be 0 or 1" >&2; exit 2; }
export RUN
PLAN="${BENCHMARK_PLAN:-$RUN/evaluation/prepared/plan.json}"
OUT="${OUT:-$ROOT/output/bonus20_c${CONCURRENCY}_$(date +%Y%m%d_%H%M%S)_$$}"
action="${1:-smoke}"
case "$action" in check|prepare|smoke|evaluate|benchmarks) ;; *) echo "Use check, prepare, smoke, evaluate or benchmarks" >&2; exit 2;; esac
: "${CONDA_PREFIX:?Activate attention_prefix_ali first}"
test -s "$DRAFT/four_anchor_config.json"
test -s "$DRAFT/model.safetensors"
test -s "$PLAN"
# Check the patch hashes and prohibit evaluation output in the training tree.
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python - "$ROOT" "$TRAIN_ROOT" "$OUT" "$PLAN" "$DRAFT" <<'PY'
import hashlib, json, sys
from pathlib import Path
root, training, output, plan, draft = map(lambda p: Path(p).resolve(), sys.argv[1:])
if root == training or training in root.parents:
    raise SystemExit('Run this from the separate inference checkout')
if not output.is_relative_to(root / 'output'):
    raise SystemExit('OUT must be under the separate inference checkout/output')
if output.exists():
    raise SystemExit('OUT already exists; use a fresh output directory')
manifest = json.loads((root / 'four_anchor/BONUS_BATCH_RUNTIME.json').read_text())
for name, expected in manifest['sha256'].items():
    if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
        raise SystemExit(f'Inference runtime changed: {name}')
cfg = json.loads((draft / 'four_anchor_config.json').read_text())['draft_config']
if (cfg.get('anchor_layout'), cfg.get('format_version'), cfg.get('num_anchors'), cfg.get('num_proposals')) != ('target_four', 3, 4, 15):
    raise SystemExit('Requires the existing format-3 target_four / 4 anchors / 15 proposals checkpoint')
print('Inference runtime verified; checkpoint and frozen prompt plan are read-only', flush=True)
PY
if [[ "$action" == check ]]; then
    exec bash "$ROOT/four_anchor/scripts/vllm_batch_ascend.sh" check
fi
if [[ "$action" == smoke || "$action" == evaluate ]]; then
    # Same frozen GSM8K prompts; speculative only unless WITH_BASELINE=1.
    export DATA
    DATA="$(TORCH_DEVICE_BACKEND_AUTOLOAD=0 python - "$PLAN" "$MODEL" <<'PY'
import hashlib, json, sys
from pathlib import Path
plan = json.loads(Path(sys.argv[1]).read_text())
if Path(plan['target']).resolve() != Path(sys.argv[2]).resolve():
    raise SystemExit('Frozen plan uses a different target')
entry = plan['benchmarks']['gsm8k']
path = Path(entry['data'])
if hashlib.sha256(path.read_bytes()).hexdigest() != entry['prompt_file_sha256']:
    raise SystemExit('Frozen GSM8K prompts changed')
print(path)
PY
)"
    if [[ "$action" == smoke ]]; then
        export LIMIT="${LIMIT:-32}" MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
    else
        export LIMIT="${LIMIT:-128}" MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
    fi
    export WARMUP="${WARMUP:-2}" MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
    echo "Results: $OUT"
    if [[ "$WITH_BASELINE" == 1 ]]; then
        MODE=ar BASELINE= OUTPUT_DIR="$OUT/ar" bash "$ROOT/four_anchor/scripts/vllm_batch_ascend.sh" evaluate
        export BASELINE="$OUT/ar/per_prompt.jsonl"
    fi
    MODE=two_pass BASELINE="${BASELINE:-}" OUTPUT_DIR="$OUT/two_pass" \
        bash "$ROOT/four_anchor/scripts/vllm_batch_ascend.sh" evaluate
else
    args=(--root "$VLLM_SOURCE_ROOT" --code-root "$ROOT" --inference-layout bonus_addon_batch --concurrency "$CONCURRENCY"
          --run "$RUN" --draft "$DRAFT" --model "$MODEL" --prompt-plan "$PLAN"
          --output "$OUT" --npu "$EVAL_NPU" --limit 128 --max-new-tokens 2048
          --datasets "${BENCHMARKS:-all}")
    if [[ "$WITH_BASELINE" == 1 ]]; then args+=(--with-baseline); fi
    if [[ "$action" == prepare ]]; then args+=(--prepare-only); fi
    exec python "$ROOT/four_anchor/scripts/run_batched_benchmarks.py" "${args[@]}"
fi
