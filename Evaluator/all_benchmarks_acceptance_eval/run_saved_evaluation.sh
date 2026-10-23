#!/usr/bin/env bash
set -eo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="${EVAL_REPO:-/home/n84449292/a84384644/attention_imp_code/vLLM_NPU_spec_main}"
HEAD="${EVAL_HEAD:-$REPO/output/acceptance_multi_20261020_033003_1552425/checkpoints/final.pt}"
ENV_ROOT="${ACCEPTANCE_ENV_ROOT:-/home/n84449292/a84384644/attention_modif}"
PYTHON_BIN="$ENV_ROOT/bin/python"
CANN_ROOT="${ACCEPTANCE_CANN_ROOT:-/home/n84449292/m84379596/CANN/9.1.0}"
TARGET_NPUS="${EVAL_TARGET_NPUS:-0,1,2,3,4,5}"
EVAL_NPU="${EVAL_NPU:-6}"
PORT="${EVAL_PORT:-8115}"
VERIFY_PORT="${EVAL_PORT:-8115}"
OUT="${BENCHMARK_OUTPUT:-$REPO/output/acceptance_saved_eval_$(date +%Y%m%d_%H%M%S)_$$}"
[[ "$TARGET_NPUS" =~ ^[0-9]+(,[0-9]+)*$ && "$EVAL_NPU" =~ ^[0-9]+$ ]] || { echo 'Invalid NPU list'; exit 1; }
IFS=, read -ra DEVICES <<< "$TARGET_NPUS"
for id in "${DEVICES[@]}"; do
    [[ "$id" != "$EVAL_NPU" ]] || { echo 'Target and evaluator NPUs must be separate'; exit 1; }
done
[[ ! -e "$OUT" && ! -e "$OUT.runtime" ]] || { echo "Choose a fresh BENCHMARK_OUTPUT: $OUT"; exit 1; }
[[ -f "$HEAD" ]] || { echo "Missing checkpoint: $HEAD"; exit 1; }
source "$CANN_ROOT/cann/set_env.sh"
source "$CANN_ROOT/nnal/atb/set_env.sh"
export PATH="$ENV_ROOT/bin:$PATH"
export LD_LIBRARY_PATH="$ENV_ROOT/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$REPO/speculators/src:$REPO/speculators/hs_connectors/src:$REPO/vllm:$REPO/vllm-ascend:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
export NO_PROXY=localhost,127.0.0.1,::1 no_proxy=localhost,127.0.0.1,::1
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY ftp_proxy FTP_PROXY
unset ASCEND_LAUNCH_BLOCKING RANK LOCAL_RANK WORLD_SIZE TORCH_COMPILE_DISABLE TORCHDYNAMO_DISABLE
unset VLLM_ASCEND_PREFIX_ACCEPTANCE_TRACE_DIR ACCEPTANCE_CONFIG VLLM_ASCEND_PREFIX_ACCEPTANCE_CONFIG
command -v curl >/dev/null
command -v setsid >/dev/null
# Read paths and sequence length from the actual trained checkpoint.
META="$("$PYTHON_BIN" - "$HEAD" "$PORT" <<'PY'
import json,socket,sys,torch
from pathlib import Path
c=torch.load(sys.argv[1],map_location='cpu',weights_only=True)
s=c['settings']
for key in ('target','draft'):
    if not Path(s[key]).is_dir(): raise SystemExit('Missing '+key+': '+s[key])
with socket.socket() as sock: sock.bind(('127.0.0.1',int(sys.argv[2])))
print(s['target'])
print(s['draft'])
print(s['seq_len'])
print(' '.join(map(str,json.loads((Path(s['draft'])/'config.json').read_text())['aux_hidden_state_layer_ids'])))
PY
)"
mapfile -t CONFIG <<< "$META"
[[ "${#CONFIG[@]}" == 4 ]] || { echo 'Unexpected checkpoint metadata output'; exit 1; }
TARGET="${CONFIG[0]}"
SEQ_LEN="${CONFIG[2]}"
read -ra LAYERS <<< "${CONFIG[3]}"
SERVER_PID='' EVAL_PID=''
cleanup() {
    local pid
    trap '' INT TERM
    for pid in "$EVAL_PID" "$SERVER_PID"; do
        [[ -z "$pid" ]] || kill -TERM -- "-$pid" 2>/dev/null || true
    done
    sleep 3
    for pid in "$EVAL_PID" "$SERVER_PID"; do
        [[ -z "$pid" ]] || kill -KILL -- "-$pid" 2>/dev/null || true
    done
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
SAVED_RUN="${EVAL_REUSE_RUN:-/home/n84449292/a84384644/attention_imp_code/vLLM_NPU_spec_main/output/all_benchmarks_acceptance_20261022_033506_3223512}"
read -ra BENCHMARK_LIST <<< "${BENCHMARKS:-gsm8k math500 humaneval mbpp aime25 mtbench alpaca}"
for file in evaluate_acceptance.py report_directional_errors.py summarize_benchmarks.py; do
    [[ -f "$SCRIPT_DIR/$file" ]] || { echo "Put this launcher inside your existing all_benchmarks_acceptance_eval directory. Missing: $file"; exit 1; }
done
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# Validate saved data before starting the verification server. Never regenerate.
"$PYTHON_BIN" - "$SAVED_RUN" "$HEAD" "$OUT" "${EVAL_PROMPTS_PER_DATASET:-128}" "${BENCHMARK_LIST[@]}" <<'PYDATA'
import json,sys
from pathlib import Path
import torch
from datasets import load_from_disk
saved_run,head,out=map(Path,sys.argv[1:4])
cap=int(sys.argv[4])
if cap<1: raise SystemExit('EVAL_PROMPTS_PER_DATASET must be positive')
checkpoint=torch.load(head,map_location='cpu',weights_only=True)
settings=checkpoint['settings']
manifest=json.loads((saved_run/'manifest.json').read_text())
selected={}
for name in sys.argv[5:]:
    path=saved_run/name/'prepared'/'dataset'
    preparation=json.loads((path.parent/'preparation.json').read_text())
    summary=json.loads((path.parent/'summary.json').read_text())
    if Path(preparation['target']).resolve()!=Path(settings['target']).resolve():
        raise SystemExit('Saved generation target differs from checkpoint target: '+name)
    data=load_from_disk(str(path))
    if not {'input_ids','loss_mask','seq_len'}.issubset(data.column_names):
        raise SystemExit('Invalid saved columns: '+name)
    if len(data)!=summary['tasks'] or not len(data):
        raise SystemExit('Saved dataset incomplete: '+name)
    for row in data:
        n=int(row['seq_len'])
        if n!=len(row['input_ids']) or n!=len(row['loss_mask']) or not 0<n<=settings['seq_len']:
            raise SystemExit('Invalid saved token/mask lengths: '+name)
    entry=dict(manifest['benchmarks'][name])
    entry.update(selected_rows=min(cap,len(data)),reused_dataset=str(path.resolve()))
    selected[name]=entry
    print(f'Reusing {name}: {min(cap,len(data))}/{len(data)} saved rows; {path}',flush=True)
manifest.update(benchmarks=selected,reused_from=str(saved_run.resolve()),mode='evaluation_only_no_regeneration')
out.mkdir(parents=True,exist_ok=False)
(out/'manifest.json').write_text(json.dumps(manifest,indent=2))
PYDATA
wait_ready() {
    local port="$1" logfile="$2"
    for ((attempt=0;attempt<360;attempt++)); do
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then tail -n 100 "$logfile"; return 1; fi
        if curl --noproxy '*' --connect-timeout 2 --max-time 3 -sf "http://127.0.0.1:$port/v1/models" >/dev/null; then return 0; fi
        sleep 5
    done
    tail -n 100 "$logfile"
    echo 'Server startup timed out'
    return 1
}
wait_job() {
    local logfile="$1" status
    while kill -0 "$EVAL_PID" 2>/dev/null; do
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then tail -n 100 "$logfile"; echo 'Target server exited'; return 1; fi
        sleep 2
    done
    if wait "$EVAL_PID"; then EVAL_PID=''; else status=$?; EVAL_PID=''; return "$status"; fi
}
stop_server() {
    kill -TERM -- "-$SERVER_PID" 2>/dev/null || true
    for ((attempt=0;attempt<30;attempt++)); do
        if ! kill -0 -- "-$SERVER_PID" 2>/dev/null; then break; fi
        sleep 1
    done
    kill -KILL -- "-$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
    SERVER_PID=''
    sleep 2
}
cd "$REPO/speculators"
echo "Evaluation ONLY: target NPUs $TARGET_NPUS; evaluator physical NPU $EVAL_NPU"
ASCEND_RT_VISIBLE_DEVICES="$TARGET_NPUS" setsid "$PYTHON_BIN" scripts/launch_vllm.py "$TARGET" \
    --target-layer-ids "${LAYERS[@]}" -- \
    --data-parallel-size "${#DEVICES[@]}" --port "$VERIFY_PORT" \
    --max-model-len "$((SEQ_LEN+256))" --gpu-memory-utilization 0.85 \
    > "$OUT/verification_server.log" 2>&1 &
SERVER_PID=$!
wait_ready "$VERIFY_PORT" "$OUT/verification_server.log"
for benchmark in "${BENCHMARK_LIST[@]}"; do
    mkdir -p "$OUT/$benchmark"
    echo "Evaluating acceptance head: $benchmark"
    TORCH_COMPILE_DISABLE=1 TORCHDYNAMO_DISABLE=1 ASCEND_RT_VISIBLE_DEVICES="$EVAL_NPU" \
        setsid "$PYTHON_BIN" "$SCRIPT_DIR/evaluate_acceptance.py" \
        --checkpoint "$HEAD" --dataset "$SAVED_RUN/$benchmark/prepared/dataset" \
        --output "$OUT/$benchmark/evaluation" --endpoint "http://127.0.0.1:$VERIFY_PORT/v1" \
        --max-samples "${EVAL_PROMPTS_PER_DATASET:-128}" > >(tee "$OUT/$benchmark/evaluation.log") 2>&1 &
    EVAL_PID=$!
    wait_job "$OUT/verification_server.log"
    "$PYTHON_BIN" "$SCRIPT_DIR/report_directional_errors.py" "$OUT/$benchmark/evaluation"
done
stop_server
"$PYTHON_BIN" "$SCRIPT_DIR/summarize_benchmarks.py" "$OUT"
