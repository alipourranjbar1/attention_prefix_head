#!/usr/bin/env bash
set -eo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export BASE="${BASE:-/home/n84449292/a84384644}"
export PREPARED_PAIR="${PREPARED_PAIR:-$BASE/parallel_training_runs/dedup_base_pair}"
export RUN_DIR="${DLOOP_RUN_DIR:-$BASE/parallel_training_runs/dloop_1target_7train_a256_extra1}"
source "$ROOT/environment.sh"
set -u
cd "$ROOT"
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -m pair.dloop_8card prepare \
    --source "$PREPARED_PAIR" --run-dir "$RUN_DIR" --teacher-port "${TEACHER_PORT:-8093}"
