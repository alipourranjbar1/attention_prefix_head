#!/usr/bin/env bash
# Complete local pipeline. Inputs are external paths, not hard-coded cluster paths.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
: "${TARGET:?Set TARGET to a Qwen3-4B checkpoint or HF model ID}"
: "${EVAL_PROMPTS:?Set EVAL_PROMPTS to HELD-OUT evaluation prompts}"
run_dir="${RUN_DIR:-four_anchor/runs/qwen3_4b}"
device="${DEVICE:-cuda:0}"
dtype="${DTYPE:-bfloat16}"
mkdir -p "$run_dir/data"
if [[ -n "${PREPARED_DATA:-}" ]]; then
  training_data="$PREPARED_DATA"
else
  : "${TRAIN_PROMPTS:?Set TRAIN_PROMPTS or provide PREPARED_DATA}"
  training_data="$run_dir/data/train.jsonl"
  python -m four_anchor.cli prepare --target "$TARGET" --data "$TRAIN_PROMPTS" \
    --output "$training_data" --device "$device" --dtype "$dtype" \
    --max-new-tokens "${TEACHER_TOKENS:-256}"
fi
init_args=()
if [[ -n "${DFLASH_INIT:-}" ]]; then
  init_args+=(--initialize-dflash "$DFLASH_INIT")
fi
bash four_anchor/scripts/train.sh --target "$TARGET" --data "$training_data" \
  --output "$run_dir/train" --device "$device" --dtype "$dtype" "${init_args[@]}" "$@"
checkpoint="$(cat "$run_dir/train/latest_checkpoint.txt")"
python -m four_anchor.cli evaluate --checkpoint "$checkpoint" --target "$TARGET" \
  --data "$EVAL_PROMPTS" --output "$run_dir/evaluation" --device "$device" --dtype "$dtype" \
  --limit "${EVAL_LIMIT:-100}" --max-new-tokens "${EVAL_TOKENS:-256}"
python -m four_anchor.cli generate --checkpoint "$checkpoint" --target "$TARGET" \
  --device "$device" --dtype "$dtype" --prompt 'Explain speculative decoding in simple terms.' \
  --output "$run_dir/example.json" --max-new-tokens 128
