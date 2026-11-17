#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
config="${CONFIG:-four_anchor/configs/qwen3_4b.json}"
workers="${NPROC_PER_NODE:-1}"
if [[ "$workers" == "1" ]]; then
  exec python -m four_anchor.cli train --config "$config" "$@"
else
  exec python -m torch.distributed.run --nnodes=1 --nproc_per_node="$workers" \
    --master_addr="${MASTER_ADDR:-127.0.0.1}" --master_port="${MASTER_PORT:-29500}" \
    -m four_anchor.cli train --config "$config" "$@"
fi
