#!/usr/bin/env bash
set -eo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/environment.sh"
python -m pair.run dloop --run-dir "$RUN_DIR" "$@"
