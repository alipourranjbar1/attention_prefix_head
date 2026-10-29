
#!/usr/bin/env bash
set -eo pipefail

ROOT=/home/n84449292/a84384644/attention_head_forked/attention_prefix_head
OLD_RUN="$ROOT/output/qwen3_4b_ar_16_20261025_104825_2573162"
: "${CONDA_PREFIX:?Activate attention_prefix_ali before running this script}"

for env_file in \
  /home/n84449292/a84384644/CANN/9.1.0/ascend-toolkit/set_env.sh \
  /home/n84449292/a84384644/CANN/9.1.0/nnal/atb/set_env.sh; do
    if [[ -f "$env_file" ]]; then source "$env_file"; fi
done
set -u

CXX_LIB="$CONDA_PREFIX/lib/libstdc++.so.6"
[[ -r "$CXX_LIB" ]] || { echo "Missing Conda C++ library: $CXX_LIB"; exit 1; }
export LD_PRELOAD="$CXX_LIB${LD_PRELOAD:+:$LD_PRELOAD}"
export PYTHONPATH="$ROOT/speculators/src:$ROOT/speculators/hs_connectors/src:$ROOT/vllm:$ROOT/vllm-ascend:${PYTHONPATH:-}"
export PYTHON_BIN="$CONDA_PREFIX/bin/python"

cd "$ROOT"
[[ "$(git branch --show-current)" == dflash-autoregressive ]] || {
    echo "Switch to dflash-autoregressive first."; exit 1;
}
"$PYTHON_BIN" -c 'import sqlite3, torch, torch_npu; print("Imports OK")'

export MODE=ar CHECK_ONLY=0
export VLLM_NPUS=1 VLLM_DP=1
export TRAIN_NPUS=2,3,4,5 NUM_TRAIN_NPUS=4
export VLLM_PORT=8096 VLLM_RPC_PORT=29696 MAX_MODEL_LEN=6400
export OUTPUT_DIR="$ROOT/output/qwen3_4b_ar_repeat_$(date +%Y%m%d_%H%M%S)_$$"
export BASELINE_CONFIG
BASELINE_CONFIG="$(mktemp /tmp/ar_baseline.XXXXXX)"
trap 'rm -f "$BASELINE_CONFIG"' EXIT

# The existing launcher requires a vanilla-DFlash config as its input.
# It converts this temporary config back to AR before training.
"$PYTHON_BIN" - "$OLD_RUN/checkpoints/run.yaml" "$BASELINE_CONFIG" <<'PY'
import sys
from pathlib import Path
from speculators.train.config import TrainConfig

f = TrainConfig.resolve(["--config", sys.argv[1]]).flatten()
if f["speculator_type"] != "dflash_ar":
    raise SystemExit("Expected the original AR training configuration")
f.update(
    speculator_type="dflash",
    sample_from_anchor=False,
    from_pretrained="",
    prefix_backbone_init=None,
    no_resume_from_checkpoint=True,
    save_best=False,
)
Path(sys.argv[2]).write_text(TrainConfig.from_flat(f).dump_yaml())
PY

# Prevent unrelated path overrides from an earlier terminal command.
unset MODEL DATASET
printf 'New run: %s\nTraining log: %s/logs/train.log\n' "$OUTPUT_DIR" "$OUTPUT_DIR"
bash "$ROOT/Evaluator/train_dflash_ar.sh"