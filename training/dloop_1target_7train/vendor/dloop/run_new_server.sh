#!/usr/bin/env bash
set -eo pipefail
source /home/n84449292/miniforge3/etc/profile.d/conda.sh
conda activate four_anchor_ali
PACKAGE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$PACKAGE_ROOT/train_ascend.sh"
