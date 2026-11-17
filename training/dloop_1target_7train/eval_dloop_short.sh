#!/usr/bin/env bash
set -eo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export BASE="${BASE:-/home/n84449292/a84384644}"
export RUN_DIR="${DLOOP_RUN_DIR:-$BASE/parallel_training_runs/dloop_1target_7train_a256_extra1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
source "$ROOT/environment.sh"
set -u
cd "$ROOT"
exec python -m pair.quick_eval --run-dir "$RUN_DIR" "$@"
