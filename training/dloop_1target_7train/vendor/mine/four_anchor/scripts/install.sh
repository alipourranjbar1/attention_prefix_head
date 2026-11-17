#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
python -c 'import torch; print("Using installed torch:", torch.__version__)'
if [[ ! -f speculators/src/speculators/models/dflash/model_definitions.py ]]; then
  git submodule update --init speculators
fi
# Do not let an upstream all-platform dependency list replace an existing
# CUDA/Ascend torch stack or install unrelated audio/vision kernels.
python -m pip install --no-deps -e ./speculators/hs_connectors
python -m pip install --no-deps -e ./speculators
python -m pip install -e '.[test]'
python -c 'from speculators.models.dflash.model_definitions import Qwen3DFlashDecoderLayer; print("Regular DFlash ready")'
