#!/usr/bin/env bash
# Source, rather than execute, this file. No package installation or upgrades.
PAIR_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export BASE="${BASE:-/home/n84449292/a84384644}"
export RUN_DIR="${RUN_DIR:-$BASE/parallel_training_runs/dedup_base_pair}"
export CANN_ROOT="${CANN_ROOT:-$BASE/CANN/9.1.0}"
if [[ -z "${CONDA_PREFIX:-}" ]]; then
  source /home/n84449292/miniforge3/etc/profile.d/conda.sh
  conda activate four_anchor_ali
fi
if [[ -f "$CANN_ROOT/cann/set_env.sh" ]]; then
  source "$CANN_ROOT/cann/set_env.sh"
elif [[ -f "$CANN_ROOT/ascend-toolkit/set_env.sh" ]]; then
  source "$CANN_ROOT/ascend-toolkit/set_env.sh"
else
  echo "Missing CANN environment under $CANN_ROOT" >&2
  return 2
fi
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
if [[ -f "$CANN_ROOT/nnal/atb/set_env.sh" ]]; then
  PAIR_ABI="$(TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -c 'import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))')"
  source "$CANN_ROOT/nnal/atb/set_env.sh" --cxx_abi="$PAIR_ABI"
  export LD_LIBRARY_PATH="$ATB_HOME_PATH/lib:$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
fi
export PYTHONPATH="$PAIR_ROOT:$PAIR_ROOT/vendor/mine:$PAIR_ROOT/vendor/speculators/src:$PAIR_ROOT/vendor/dloop:$PAIR_ROOT/vendor/dloop/upstream:${PYTHONPATH:-}"
export SPECULATORS_ROOT="$PAIR_ROOT/vendor/speculators"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"
export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-1800}"
export SPECFORGE_DEVICE=npu TORCH_COMPILE_DISABLE=1 TORCHDYNAMO_DISABLE=1
export SOC_VERSION="${SOC_VERSION:-ascend910_9372}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
unset ASCEND_LAUNCH_BLOCKING
