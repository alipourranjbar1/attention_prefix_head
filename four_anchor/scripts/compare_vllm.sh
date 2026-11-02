#!/usr/bin/env bash
# Sequential plain-Qwen and four-anchor runs on the same physical NPU.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN="${RUN:-$ROOT/output/four_anchor_dflash_20261031_025300_2111006}"
COMPARISON_DIR="${COMPARISON_DIR:-$RUN/evaluation/known_bonus_comparison_$(date +%Y%m%d_%H%M%S)_$$}"
export VLLM_SOURCE_ROOT="${VLLM_SOURCE_ROOT:-$ROOT}"
export EVAL_NPU="${EVAL_NPU:-7}" LIMIT="${LIMIT:-128}" MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
export NUM_SPECULATIVE_TOKENS="${NUM_SPECULATIVE_TOKENS:-15}"
export PROFILE_STEPS=0 TRACE_STEPS=0
mkdir -p "$COMPARISON_DIR"
printf 'Comparison directory: %s\n' "$COMPARISON_DIR"
MODE=ar DRAFT_GRAPH=0 BASELINE='' OUTPUT_DIR="$COMPARISON_DIR/ar" \
    bash "$ROOT/four_anchor/scripts/vllm_ascend.sh" evaluate
failed=0
if [[ "${COMPARE_ONE_PASS:-0}" == 1 ]]; then
    MODE=one_pass BASELINE="$COMPARISON_DIR/ar/per_prompt.jsonl" OUTPUT_DIR="$COMPARISON_DIR/one_pass" \
        bash "$ROOT/four_anchor/scripts/vllm_ascend.sh" evaluate || failed=1
fi
MODE=two_pass BASELINE="$COMPARISON_DIR/ar/per_prompt.jsonl" OUTPUT_DIR="$COMPARISON_DIR/two_pass" \
    bash "$ROOT/four_anchor/scripts/vllm_ascend.sh" evaluate || failed=1
python - "$COMPARISON_DIR" <<'PY'
import json
from pathlib import Path
import sys
root = Path(sys.argv[1])
rows = []
for mode in ('ar', 'one_pass', 'two_pass'):
    path = root / mode / 'summary.json'
    if path.exists():
        result = json.loads(path.read_text())
        rows.append({'mode': mode,
                     'tokens_per_second': result['tokens_per_second_including_prefill'],
                     'acceptance_with_bonus': result['acceptance_length_including_bonus'],
                     'speedup_vs_ar': result.get('measured_speedup_vs_ar', 1),
                     'matching_prompts': result.get('matching_prompts'),
                     'all_match_greedy_target': result.get('all_match_greedy_target')})
(root / 'comparison.json').write_text(json.dumps(rows, indent=2) + '\n')
print(json.dumps(rows, indent=2))
print('Saved:', root / 'comparison.json')
PY
exit "$failed"
