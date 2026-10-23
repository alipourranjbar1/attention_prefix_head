#!/usr/bin/env bash
set -eo pipefail
bash -n -- "${BASH_SOURCE[0]}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="$SCRIPT_DIR/$(basename -- "${BASH_SOURCE[0]}")"

# ============================================================
# Paths — new server
# ============================================================
REPO="${EVAL_REPO:-/home/n84449292/a84384644/attention_head_forked/attention_prefix_head}"
ENV_ROOT="${ACCEPTANCE_ENV_ROOT:-/home/n84449292/.conda/envs/attention_prefix_ali}"
PYTHON_BIN="$ENV_ROOT/bin/python"

CANN_ROOT="${ACCEPTANCE_CANN_ROOT:-/home/n84449292/a84384644/CANN/9.1.0}"

# Original trained acceptance-head checkpoint.
HEAD_SOURCE="${EVAL_HEAD:-/home/n84449292/a84384644/linear_head_selector/checkpoints/final.pt}"

# A model directory or Hugging Face cache directory.
TARGET_PATH="${EVAL_TARGET_MODEL:-/home/n84449292/m84379596/Huggingface/models--Qwen--Qwen3-4B}"
DRAFT_PATH="${EVAL_DRAFT_MODEL:-/home/n84449292/a84384644/prefix_drafter_check/checkpoints/4}"

# Local benchmark cache. All datasets must already be available.
CACHE_ROOT="${EVAL_CACHE_ROOT:-/home/n84449292/.cache/huggingface/datasets}"

# ============================================================
# Devices and evaluation settings
# ============================================================
TARGET_NPUS="${EVAL_TARGET_NPUS:-0}"
EVAL_NPU="${EVAL_NPU:-2}"

PORT="${GENERATION_PORT:-8116}"
VERIFY_PORT="${EVAL_PORT:-8115}"

PROMPTS="${EVAL_PROMPTS_PER_DATASET:-128}"
MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-1024}"
CONCURRENCY="${EVAL_GENERATION_CONCURRENCY:-4}"

OUT="${BENCHMARK_OUTPUT:-$REPO/output/all_benchmarks_acceptance_$(date +%Y%m%d_%H%M%S)_$$}"

read -ra BENCHMARK_LIST <<< \
    "${BENCHMARKS:-gsm8k math500 humaneval mbpp lcb aime25 mtbench alpaca}"

# ============================================================
# Validate files and settings
# ============================================================
[[ "$TARGET_NPUS" =~ ^[0-9]+(,[0-9]+)*$ &&
   "$EVAL_NPU" =~ ^[0-9]+$ ]] || {
    echo "ERROR: Invalid NPU list"
    exit 1
}

IFS=, read -ra DEVICES <<< "$TARGET_NPUS"

declare -A SEEN_DEVICES=()
for id in "${DEVICES[@]}"; do
    [[ "$id" != "$EVAL_NPU" ]] || {
        echo "ERROR: Target and evaluator NPUs must be separate"
        exit 1
    }
    [[ -z "${SEEN_DEVICES[$id]:-}" ]] || {
        echo "ERROR: Duplicate target NPU: $id"
        exit 1
    }
    SEEN_DEVICES[$id]=1
done

[[ "${#BENCHMARK_LIST[@]}" -gt 0 ]] || {
    echo "ERROR: BENCHMARKS must not be empty"
    exit 1
}

[[ ! -e "$OUT" && ! -e "$OUT.runtime" ]] || {
    echo "ERROR: Choose a fresh BENCHMARK_OUTPUT: $OUT"
    exit 1
}

for file in \
    "$PYTHON_BIN" \
    "$HEAD_SOURCE" \
    "$CANN_ROOT/ascend-toolkit/set_env.sh" \
    "$CANN_ROOT/nnal/atb/set_env.sh" \
    "$REPO/speculators/scripts/launch_vllm.py" \
    "$SCRIPT_DIR/benchmark_sources.py" \
    "$SCRIPT_DIR/prepare_benchmark.py" \
    "$SCRIPT_DIR/evaluate_acceptance.py" \
    "$SCRIPT_DIR/report_directional_errors.py" \
    "$SCRIPT_DIR/summarize_benchmarks.py"
do
    [[ -f "$file" ]] || {
        echo "ERROR: Missing required file: $file"
        exit 1
    }
done

[[ -d "$CACHE_ROOT" ]] || {
    echo "ERROR: Missing benchmark cache: $CACHE_ROOT"
    echo "Set EVAL_CACHE_ROOT to the correct local dataset cache."
    exit 1
}

command -v curl >/dev/null
command -v setsid >/dev/null

# ============================================================
# CANN and Python environment
# ============================================================
source "$CANN_ROOT/ascend-toolkit/set_env.sh"
source "$CANN_ROOT/nnal/atb/set_env.sh"

export PATH="$ENV_ROOT/bin:$PATH"
export LD_LIBRARY_PATH="$ENV_ROOT/lib:${LD_LIBRARY_PATH:-}"

# Fix the previously observed CXXABI runtime mismatch.
CXX_RUNTIME="$ENV_ROOT/lib/libstdc++.so.6"
[[ -f "$CXX_RUNTIME" ]] || {
    echo "ERROR: Missing C++ runtime: $CXX_RUNTIME"
    exit 1
}
export LD_PRELOAD="$CXX_RUNTIME${LD_PRELOAD:+:$LD_PRELOAD}"

export PYTHONPATH="$REPO/speculators/src:$REPO/speculators/hs_connectors/src:$REPO/vllm:$REPO/vllm-ascend:${PYTHONPATH:-}"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1

export NO_PROXY=localhost,127.0.0.1,::1
export no_proxy="$NO_PROXY"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
unset all_proxy ALL_PROXY ftp_proxy FTP_PROXY
unset ASCEND_LAUNCH_BLOCKING RANK LOCAL_RANK WORLD_SIZE
unset TORCH_COMPILE_DISABLE TORCHDYNAMO_DISABLE
unset VLLM_ASCEND_PREFIX_ACCEPTANCE_TRACE_DIR
unset ACCEPTANCE_CONFIG VLLM_ASCEND_PREFIX_ACCEPTANCE_CONFIG

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Avoid the outer vllm directory shadowing the installed package.
cd "$REPO/speculators"

# ============================================================
# Resolve model paths and create a relocated checkpoint copy.
# Original checkpoint and weights remain unchanged.
# ============================================================
"$PYTHON_BIN" - \
    "$HEAD_SOURCE" \
    "$TARGET_PATH" \
    "$DRAFT_PATH" \
    "$PORT" \
    "$VERIFY_PORT" \
    "$OUT" \
    "$SCRIPT_DIR" <<'PY'
import ast
import json
import socket
import sys
from pathlib import Path

import torch

head_source = Path(sys.argv[1]).expanduser().resolve(strict=True)
target_input = Path(sys.argv[2]).expanduser()
draft_input = Path(sys.argv[3]).expanduser()
ports = [int(sys.argv[4]), int(sys.argv[5])]
out = Path(sys.argv[6]).expanduser().resolve()
script_dir = Path(sys.argv[7])

for name in (
    "benchmark_sources.py",
    "prepare_benchmark.py",
    "evaluate_acceptance.py",
    "report_directional_errors.py",
    "summarize_benchmarks.py",
):
    path = script_dir / name
    ast.parse(path.read_bytes(), filename=str(path))

if ports[0] == ports[1]:
    raise SystemExit("Generation and verification ports must differ.")

sockets = []
try:
    for port in ports:
        if not 1 <= port <= 65535:
            raise SystemExit(f"Invalid port: {port}")
        sock = socket.socket()
        sockets.append(sock)
        sock.bind(("127.0.0.1", port))
finally:
    for sock in sockets:
        sock.close()


def resolve_model(path):
    if (path / "config.json").is_file():
        return path.resolve()

    ref = path / "refs" / "main"
    if ref.is_file():
        snapshot = path / "snapshots" / ref.read_text().strip()
        if (snapshot / "config.json").is_file():
            return snapshot.resolve()

    raise SystemExit(
        f"Cannot resolve model directory: {path}\n"
        "Provide the exact snapshot directory containing config.json."
    )


target = resolve_model(target_input)
draft = resolve_model(draft_input)

cfg = json.loads((draft / "config.json").read_text())
if cfg.get("speculators_model_type") != "dflash_prefix":
    raise SystemExit("Expected a DFlash-Prefix draft checkpoint.")

layers = cfg.get("aux_hidden_state_layer_ids")
if (
    not isinstance(layers, list)
    or not layers
    or any(type(x) is not int or x < 0 for x in layers)
):
    raise SystemExit("Invalid or missing aux_hidden_state_layer_ids.")

checkpoint = torch.load(
    head_source, map_location="cpu", weights_only=True
)
settings = checkpoint.get("settings")
if not isinstance(settings, dict):
    raise SystemExit("Acceptance checkpoint has no settings dictionary.")

seq_len = int(settings["seq_len"])
if seq_len <= 0:
    raise SystemExit("Invalid checkpoint seq_len.")

original_paths = {
    "target": str(settings.get("target")),
    "draft": str(settings.get("draft")),
}

settings["target"] = str(target)
settings["draft"] = str(draft)

out.mkdir(parents=True, exist_ok=False)
runtime_head = out / "head_relocated.pt"
torch.save(checkpoint, runtime_head)

metadata = {
    "original_head": str(head_source),
    "runtime_head": str(runtime_head),
    "original_paths": original_paths,
    "target": str(target),
    "draft": str(draft),
    "seq_len": seq_len,
    "target_layer_ids": layers,
}
(out / "runtime_config.json").write_text(
    json.dumps(metadata, indent=2) + "\n"
)
(out / "runtime_config.lines").write_text(
    "\n".join([
        str(target),
        str(draft),
        str(seq_len),
        " ".join(map(str, layers)),
    ]) + "\n"
)

print("Preflight passed.")
print("Target:", target)
print("Draft:", draft)
print("Sequence length:", seq_len)
print("Runtime head:", runtime_head)
PY

mapfile -t CONFIG < "$OUT/runtime_config.lines"
[[ "${#CONFIG[@]}" == 4 ]] || {
    echo "ERROR: Unexpected checkpoint metadata output"
    exit 1
}

TARGET="${CONFIG[0]}"
DRAFT="${CONFIG[1]}"
SEQ_LEN="${CONFIG[2]}"
read -ra LAYERS <<< "${CONFIG[3]}"
HEAD="$OUT/head_relocated.pt"

cp "$SCRIPT_PATH" "$OUT/launcher_source.sh"

# ============================================================
# Process management
# ============================================================
SERVER_PID=''
EVAL_PID=''

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

wait_ready() {
    local port="$1" logfile="$2"
    local attempt
    for ((attempt=0;attempt<360;attempt++)); do
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            tail -n 100 "$logfile"
            return 1
        fi
        if curl --noproxy '*' \
            --connect-timeout 2 --max-time 3 \
            -sf "http://127.0.0.1:$port/v1/models" >/dev/null
        then
            return 0
        fi
        sleep 5
    done
    tail -n 100 "$logfile"
    echo "ERROR: Server startup timed out"
    return 1
}

wait_job() {
    local logfile="$1" status
    while kill -0 "$EVAL_PID" 2>/dev/null; do
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            tail -n 100 "$logfile"
            echo "ERROR: Target server exited"
            return 1
        fi
        sleep 2
    done
    if wait "$EVAL_PID"; then
        EVAL_PID=''
    else
        status=$?
        EVAL_PID=''
        return "$status"
    fi
}

stop_server() {
    local attempt
    kill -TERM -- "-$SERVER_PID" 2>/dev/null || true
    for ((attempt=0;attempt<30;attempt++)); do
        if ! kill -0 -- "-$SERVER_PID" 2>/dev/null; then
            break
        fi
        sleep 1
    done
    kill -KILL -- "-$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
    SERVER_PID=''
    sleep 2
}

# ============================================================
# Validate local benchmark datasets
# ============================================================
extra=()
[[ "${EVAL_ALL_ROWS:-0}" != 1 ]] || extra+=(--all-rows)
[[ -z "${EVAL_LCB_PATH:-}" ]] || extra+=(--lcb-path "$EVAL_LCB_PATH")

echo "Checking local datasets before starting servers..."
"$PYTHON_BIN" "$SCRIPT_DIR/benchmark_sources.py" \
    --cache-root "$CACHE_ROOT" \
    --benchmarks "${BENCHMARK_LIST[@]}" \
    --limit "$PROMPTS" \
    --output "$OUT/manifest.json" \
    "${extra[@]}"

# ============================================================
# Stage 1: Generate target-model responses
# ============================================================
echo "Stage 1: response generation on target NPU(s) $TARGET_NPUS"
echo "Output: $OUT"

ASCEND_RT_VISIBLE_DEVICES="$TARGET_NPUS" \
setsid "$PYTHON_BIN" -m vllm.entrypoints.openai.api_server \
    --model "$TARGET" \
    --host 127.0.0.1 \
    --port "$PORT" \
    --data-parallel-size "${#DEVICES[@]}" \
    --max-model-len "$((SEQ_LEN+256))" \
    --gpu-memory-utilization 0.85 \
    --max-num-seqs 8 \
    --enforce-eager \
    --generation-config vllm \
    > "$OUT/generation_server.log" 2>&1 &
SERVER_PID=$!

wait_ready "$PORT" "$OUT/generation_server.log"

for benchmark in "${BENCHMARK_LIST[@]}"; do
    mkdir -p "$OUT/$benchmark"
    echo "Generating: $benchmark"

    ASCEND_RT_VISIBLE_DEVICES="" \
    setsid "$PYTHON_BIN" "$SCRIPT_DIR/prepare_benchmark.py" \
        --checkpoint "$HEAD" \
        --endpoint "http://127.0.0.1:$PORT/v1" \
        --manifest "$OUT/manifest.json" \
        --benchmark "$benchmark" \
        --output "$OUT/$benchmark/prepared" \
        --max-new-tokens "$MAX_NEW_TOKENS" \
        --concurrency "$CONCURRENCY" \
        > >(tee "$OUT/$benchmark/preparation.log") 2>&1 &
    EVAL_PID=$!

    wait_job "$OUT/generation_server.log"
done

stop_server

# ============================================================
# Stage 2: Evaluate acceptance-head predictions
# ============================================================
echo "Stage 2: verification on target NPU(s) $TARGET_NPUS"

ASCEND_RT_VISIBLE_DEVICES="$TARGET_NPUS" \
setsid "$PYTHON_BIN" scripts/launch_vllm.py "$TARGET" \
    --provenance-dir "$OUT/verification_provenance" \
    --target-layer-ids "${LAYERS[@]}" -- \
    --data-parallel-size "${#DEVICES[@]}" \
    --port "$VERIFY_PORT" \
    --max-model-len "$((SEQ_LEN+256))" \
    --gpu-memory-utilization 0.85 \
    > "$OUT/verification_server.log" 2>&1 &
SERVER_PID=$!

wait_ready "$VERIFY_PORT" "$OUT/verification_server.log"

for benchmark in "${BENCHMARK_LIST[@]}"; do
    echo "Evaluating acceptance head: $benchmark"

    TORCH_COMPILE_DISABLE=1 \
    TORCHDYNAMO_DISABLE=1 \
    ASCEND_RT_VISIBLE_DEVICES="$EVAL_NPU" \
    setsid "$PYTHON_BIN" "$SCRIPT_DIR/evaluate_acceptance.py" \
        --checkpoint "$HEAD" \
        --dataset "$OUT/$benchmark/prepared/dataset" \
        --output "$OUT/$benchmark/evaluation" \
        --endpoint "http://127.0.0.1:$VERIFY_PORT/v1" \
        --max-samples 0 \
        > >(tee "$OUT/$benchmark/evaluation.log") 2>&1 &
    EVAL_PID=$!

    wait_job "$OUT/verification_server.log"

    "$PYTHON_BIN" "$SCRIPT_DIR/report_directional_errors.py" \
        "$OUT/$benchmark/evaluation"
done

stop_server

"$PYTHON_BIN" "$SCRIPT_DIR/summarize_benchmarks.py" "$OUT"
echo "Finished: $OUT"