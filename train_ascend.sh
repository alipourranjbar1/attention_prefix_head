#!/usr/bin/env bash
# NAVER DLoop training with isolated Ascend platform/data adapters.
set -eo pipefail
PACKAGE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
: "${CONDA_PREFIX:?Activate the existing Ascend Conda environment first}"
export BASE="${BASE:-/home/n84449292/a84384644}"
export CANN_ROOT="${CANN_ROOT:-$BASE/CANN/9.1.0}"
export MODEL="${MODEL:-$BASE/models/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c}"
export DATASET="${DATASET:-$BASE/datasets/open_perfectblend.qwen3-4b-rollout.qwen3.seq3072}"
export NPUS="${NPUS:-0,1,2,3,4,5,6,7}"
export MASTER_PORT="${MASTER_PORT:-29716}"
export EPOCHS="${EPOCHS:-1}" MAX_LENGTH="${MAX_LENGTH:-4096}"
export NUM_ANCHORS="${NUM_ANCHORS:-256}" LR="${LR:-0.0006}"
export NUM_WORKERS="${NUM_WORKERS:-8}" RESUME="${RESUME:-0}"
export INIT_MODE="${INIT_MODE:-pretrained}"
export HEAD_LOSS_CHUNK_SIZE="${HEAD_LOSS_CHUNK_SIZE:-0}"
export PACKED_TOKEN_BUDGET="${PACKED_TOKEN_BUDGET:-0}"
export PACKING_BUCKET_WIDTH="${PACKING_BUCKET_WIDTH:-128}"
export PACKING_MAX_RECORDS="${PACKING_MAX_RECORDS:-128}"
export PACKING_CACHE_DIR="${PACKING_CACHE_DIR:-$PACKAGE_ROOT/cache/packing}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-5000}" LOG_INTERVAL="${LOG_INTERVAL:-50}"
if [[ "$RESUME" == 1 && -z "${OUTPUT_DIR:-}" ]]; then
    echo 'RESUME=1 requires OUTPUT_DIR from the original DLoop run.' >&2; exit 2
fi
export OUTPUT_DIR="${OUTPUT_DIR:-$PACKAGE_ROOT/output/dloop_native_$(date +%Y%m%d_%H%M%S)_$$}"
# CANN must be initialized before any import of Torch/Transformers.
if [[ -f "$CANN_ROOT/cann/set_env.sh" ]]; then
    source "$CANN_ROOT/cann/set_env.sh"
elif [[ -f "$CANN_ROOT/ascend-toolkit/set_env.sh" ]]; then
    source "$CANN_ROOT/ascend-toolkit/set_env.sh"
else
    echo "CANN environment script missing under $CANN_ROOT" >&2; exit 2
fi
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
if [[ -f "$CANN_ROOT/nnal/atb/set_env.sh" ]]; then
    DLOOP_ABI="$(TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -c 'import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))')"
    source "$CANN_ROOT/nnal/atb/set_env.sh" --cxx_abi="$DLOOP_ABI"
    export LD_LIBRARY_PATH="${ATB_HOME_PATH:?ATB setup failed}/lib:$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
fi
set -u
# Deliberately do not inherit PYTHONPATH from any other research project.
export PYTHONPATH="$PACKAGE_ROOT:$PACKAGE_ROOT/upstream"
export SPECFORGE_DEVICE=npu
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TORCH_COMPILE_DISABLE=1 TORCHDYNAMO_DISABLE=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export SOC_VERSION="${SOC_VERSION:-ascend910_9372}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-1800}"
unset ASCEND_LAUNCH_BLOCKING RANK LOCAL_RANK WORLD_SIZE MASTER_ADDR
cd "$PACKAGE_ROOT"
DRAFT_SOURCE="$(python -m ascend_support.preflight --resolve-draft)"
initialization=()
if [[ "$INIT_MODE" == scratch ]]; then
    initialization+=(--scratch --draft-config-path "$DRAFT_SOURCE")
    printf 'Initialization: SCRATCH drafter; frozen pretrained target embeddings/head retained.\n'
else
    export DRAFT_INIT="$DRAFT_SOURCE"
    initialization+=(--draft-init-path "$DRAFT_INIT")
    printf 'Initialization: released pretrained DFlash drafter.\n'
fi
IFS=',' read -r -a devices <<< "$NPUS"
packing_args=()
if [[ "$PACKED_TOKEN_BUDGET" != 0 ]]; then
    if [[ ! -d "$DATASET" ]]; then
        echo 'Packed training requires the local save_to_disk token dataset.' >&2; exit 2
    fi
    # Run one CPU indexing/planning job, before allocating any target models.
    # Plan-only mode does not create an OUTPUT_DIR that would block a later run.
    mkdir -p "$PACKING_CACHE_DIR"
    PACKING_PLAN="$(mktemp "$PACKING_CACHE_DIR/plan.XXXXXX.json")"
    python -m ascend_support.packed_data --dataset "$DATASET" \
        --cache-dir "$PACKING_CACHE_DIR" --output "$PACKING_PLAN" \
        --max-length "$MAX_LENGTH" --token-budget "$PACKED_TOKEN_BUDGET" \
        --num-anchors "$NUM_ANCHORS" --world-size "${#devices[@]}" --epochs "$EPOCHS" \
        --bucket-width "$PACKING_BUCKET_WIDTH" --max-records "$PACKING_MAX_RECORDS"
    PACKING_INDEX="$(python - "$PACKING_PLAN" <<'PY_INDEX'
import json, sys
print(json.load(open(sys.argv[1]))['index_file'])
PY_INDEX
    )"
    packing_args+=(--packed-token-budget "$PACKED_TOKEN_BUDGET" --packing-index "$PACKING_INDEX"
        --packing-bucket-width "$PACKING_BUCKET_WIDTH" --packing-max-records "$PACKING_MAX_RECORDS")
fi
if [[ "${CHECK_ONLY:-0}" == 1 ]]; then
    python -m ascend_support.preflight --check-only
    echo 'Preflight complete. No training job started.'
    exit 0
fi
python -m ascend_support.preflight
if [[ "$PACKED_TOKEN_BUDGET" != 0 ]]; then
    cp -- "$PACKING_PLAN" "$OUTPUT_DIR/packing_plan.json"
fi
extra=()
if [[ -d "$DATASET" ]]; then extra+=(--saved-tokenized-data); else extra+=(--on-the-fly-loader); fi
if [[ "$RESUME" == 1 ]]; then extra+=(--resume); fi
if [[ -n "${MAX_STEPS:-}" ]]; then extra+=(--max-steps "$MAX_STEPS"); fi
if [[ "${ACTIVATION_CHECKPOINTING:-0}" == 1 ]]; then extra+=(--activation-checkpointing); fi
if [[ "$HEAD_LOSS_CHUNK_SIZE" != 0 ]]; then extra+=(--head-loss-chunk-size "$HEAD_LOSS_CHUNK_SIZE"); fi
args=(
    --target-model-path "$MODEL" --target-model-backend hf
    "${initialization[@]}" --train-data-path "$DATASET"
    --chat-template qwen --max-length "$MAX_LENGTH" --output-dir "$OUTPUT_DIR"
    --num-epochs "$EPOCHS" --batch-size 1 --learning-rate "$LR"
    --warmup-ratio 0.04 --max-grad-norm 1.0 --seed 42
    --attention-backend sdpa --num-anchors "$NUM_ANCHORS"
    --loss-type dflash --loss-decay-gamma 7.0 --dloop-max-stage 3
    --position-offset-max 28672 --log-interval "$LOG_INTERVAL" --save-interval "$SAVE_INTERVAL"
    --report-to tensorboard --dataloader-num-workers "$NUM_WORKERS"
    "${extra[@]}" "${packing_args[@]}"
)
python - "$OUTPUT_DIR" "${args[@]}" <<'PY'
import json, os, shlex, shutil, sys
from pathlib import Path
root = Path.cwd()
out = Path(sys.argv[1])
command = sys.argv[2:]
record = out / 'train_args.json'
if os.environ['RESUME'] == '1':
    def schedule_args(values):
        values = list(values)
        values = [v for v in values if v != '--resume']
        if '--max-steps' in values:
            pos = values.index('--max-steps'); del values[pos:pos+2]
        return values
    if schedule_args(json.loads(record.read_text())) != schedule_args(command):
        raise SystemExit('Resume arguments differ from the saved training recipe.')
else:
    record.write_text(json.dumps(command, indent=2) + '\n')
    (out / 'train_command.txt').write_text(shlex.join(command) + '\n')
    for name in ('UPSTREAM_COMMIT', 'UPSTREAM_SHA256.json', 'ASCEND_PORT.patch', 'README.md'):
        shutil.copy2(root / name, out / name)
    snapshot = out / 'source_snapshot'
    shutil.copytree(root / 'upstream', snapshot / 'upstream',
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copytree(root / 'ascend_support', snapshot / 'ascend_support',
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copytree(root / 'configs', snapshot / 'configs')
    for name in ('UPSTREAM_COMMIT', 'UPSTREAM_SHA256.json'):
        shutil.copy2(root / name, snapshot / name)
print('Training:', out, flush=True)
print('Each selected NPU holds a frozen HF target and a drafter training rank.', flush=True)
PY
ASCEND_RT_VISIBLE_DEVICES="$NPUS" python -m torch.distributed.run \
    --nnodes=1 --nproc_per_node="${#devices[@]}" \
    --master_addr=127.0.0.1 --master_port="$MASTER_PORT" \
    -m ascend_support.entry "${args[@]}" 2>&1 | tee -a "$OUTPUT_DIR/train.log"
printf '\nRun: %s\nCheckpoints: %s/epoch_*_step_*\n' "$OUTPUT_DIR" "$OUTPUT_DIR"
