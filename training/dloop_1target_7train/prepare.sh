#!/usr/bin/env bash
set -eo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/environment.sh"
python -m pair.prepare --run-dir "$RUN_DIR" --eval-fraction "${EVAL_FRACTION:-0.25}" --epochs "${EPOCHS:-1}" --anchors-per-sequence "${ANCHORS_PER_SEQUENCE:-512}"
