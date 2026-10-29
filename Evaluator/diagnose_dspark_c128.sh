#!/usr/bin/env bash
# CPU-only inspection of already-completed runs. Does not contact a server.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ORIGINAL_SUITE="${ORIGINAL_SUITE:-$ROOT/output/dspark_full_benchmark_c128}"
RERUN_SUITE="${RERUN_SUITE:-$ROOT/output/dspark_c128_cut_training_20261029_032419_2332808}"
DIAGNOSTIC_DIR="${DIAGNOSTIC_DIR:-$ROOT/output/dspark_diagnostic_$(date +%Y%m%d_%H%M%S)_$$}"
exec "${PYTHON_BIN:-python}" "$ROOT/Evaluator/diagnose_dspark_repeats.py" \
  --original-suite "$ORIGINAL_SUITE" \
  --rerun-suite "$RERUN_SUITE" \
  --output "$DIAGNOSTIC_DIR"
