#!/usr/bin/env bash
# Fresh full training using confirmed settings from an isolated benchmark.
set -euo pipefail
: "${CONDA_PREFIX:?Activate attention_prefix_ali first}"
: "${SETTINGS:?Set SETTINGS to the speed_trials directory's selected.env}"
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${RESUME:-}" ]]; then
    echo 'This entrypoint starts fresh. Existing training/checkpoints are preserved.' >&2
    exit 1
fi
test -s "$SETTINGS"
source "$SETTINGS"
export OUTPUT_DIR="$ROOT/output/dspark_measured_$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p "$OUTPUT_DIR"
LOG="$OUTPUT_DIR/launcher.log"
nohup bash "$ROOT/four_anchor/scripts/train_dspark_ascend.sh" >"$LOG" 2>&1 < /dev/null &
printf 'Launcher PID: %s\nLog: %s\nCheckpoints: %s/checkpoints\n' "$!" "$LOG" "$OUTPUT_DIR"
tail -F "$LOG"
