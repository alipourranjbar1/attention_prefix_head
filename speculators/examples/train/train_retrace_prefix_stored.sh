#!/usr/bin/env bash
# Activate the existing Ascend environment before running this launcher.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${RETRACE_ROOT:-$(cd -- "$SCRIPT_DIR/../../.." && pwd)}"
PYTHON_BIN="${RETRACE_PYTHON:-python}"
if [[ -n "${RETRACE_CANN_ROOT:-}" ]]; then
    set +u
    source "$RETRACE_CANN_ROOT/ascend-toolkit/set_env.sh"
    source "$RETRACE_CANN_ROOT/nnal/atb/set_env.sh"
    set -u
fi
export PYTHONPATH="$ROOT/speculators/src:$ROOT/speculators/hs_connectors/src:$ROOT/vllm:$ROOT/vllm-ascend:${PYTHONPATH:-}"
export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-3600}"
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"
exec "$PYTHON_BIN" -m speculators.models.retrace.stored_launch --root "$ROOT" "$@"
