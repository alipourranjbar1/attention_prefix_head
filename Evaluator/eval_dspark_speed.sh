#!/usr/bin/env bash
# Evaluate the existing off/cut server and compare the two completed timing runs.
set -euo pipefail
ROOT="${ROOT:-/home/n84449292/a84384644/attention_head_forked/attention_prefix_head}"
: "${CONDA_PREFIX:?Activate attention_prefix_ali first}"
PYTHON_BIN="$CONDA_PREFIX/bin/python"
PORT="${PORT:-8217}"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export NO_PROXY=localhost,127.0.0.1,::1 no_proxy=localhost,127.0.0.1,::1
if [[ -r "$CONDA_PREFIX/lib/libstdc++.so.6" ]]; then
    export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"
fi
if ! curl --noproxy '*' --max-time 5 -fsS "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
    echo 'Server is not ready. Check the server terminal before rerunning this evaluator.'
    exit 1
fi
SERVER_MANIFEST="$(readlink -f "$ROOT/output/evaluator_servers/port_$PORT.json")"
DETAILS="$("$PYTHON_BIN" - "$SERVER_MANIFEST" <<'PY'
import json
import sys
record = json.load(open(sys.argv[1]))
confidence = record.get("dspark_confidence") or {}
if record.get("kind") != "dspark" or record.get("proposals") != 7 or record.get("max_num_seqs") != 64:
    raise SystemExit("Expected a DSpark server with seven proposals and capacity 64")
if confidence.get("mode") not in {"off", "cut"} or confidence.get("trace_directory"):
    raise SystemExit("Use run_dspark_speed_server.sh off or cut; accuracy tracing must be disabled")
values = [confidence["mode"], record["target"]]
if any("\n" in value for value in values):
    raise SystemExit("Newlines in model paths are unsupported")
print("\n".join(values))
PY
)"
mapfile -t FIELDS <<< "$DETAILS"
MODE="${FIELDS[0]}"
TARGET="${FIELDS[1]}"
BASE="$ROOT/output/dspark4b_manual/c64"
OUTPUT="$BASE/speed_${MODE}_$(date +%Y%m%d_%H%M%S_%N)_$$"
mkdir -p "$BASE"
"$PYTHON_BIN" "$ROOT/Evaluator/evaluator.py" \
    --base-url "http://127.0.0.1:$PORT" --server-manifest "$SERVER_MANIFEST" \
    --target "$TARGET" --method "dspark_speed_$MODE" \
    --manifest "$ROOT/Evaluator/retrace_eval_prompts_all_v2.json" \
    --datasets gsm8k --num-prompts 128 --max-new-tokens 2048 \
    --temperatures 0 --concurrencies 64 --seed 42 --no-thinking \
    --output "$OUTPUT"
"$PYTHON_BIN" - "$OUTPUT/run_info.json" <<'PY'
import json
import sys
report = json.load(open(sys.argv[1]))
if not report.get("complete"):
    raise SystemExit("Evaluation is incomplete; no speed bookmark saved")
for cell in report["cells"]:
    summary = cell["summary"]
    print(f'{cell["benchmark"]}: aggregate throughput = {summary["wall_tokens_per_second"]:.3f} tokens/s', flush=True)
PY
printf '%s\n' "$OUTPUT/run_info.json" > "$BASE/latest_speed_${MODE}.txt"
if [[ -s "$BASE/latest_speed_off.txt" && -s "$BASE/latest_speed_cut.txt" ]]; then
    "$PYTHON_BIN" "$ROOT/Evaluator/dspark_confidence_report.py" compare \
        --baseline "$(< "$BASE/latest_speed_off.txt")" \
        --cut "$(< "$BASE/latest_speed_cut.txt")" \
        --output "$OUTPUT/speed_comparison.json"
    cat "$OUTPUT/speed_comparison.csv"
else
    echo 'Timing saved. Restart the server in the other mode and run this evaluator again.'
fi
printf 'Results: %s\n' "$OUTPUT"
