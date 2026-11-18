#!/usr/bin/env bash
set -euo pipefail
: "${ASCEND_RT_VISIBLE_DEVICES:?Set this to one allocated physical NPU ID before running}"
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
base="${BASE:-$HOME/a84384644}"
hf_store="${HF_STORE:-$HOME/m84379596/Huggingface}"
target="${TARGET:-$hf_store/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c}"
dataset="${DATASET:-$hf_store/datasets/open_perfectblend.qwen3-4b-rollout.qwen3.seq3072}"
out="${OUT:-$base/layer_audits/$(date +%Y%m%d_%H%M%S)_$$}"
[[ -f "$target/config.json" && -e "$dataset" ]] || {
    echo "Check TARGET=$target and DATASET=$dataset" >&2
    exit 1
}
if [[ -z "${DRAFT:-}" ]]; then
    for candidate in "$base/models/Qwen3-4B-DFlash-b16" "$base/Qwen3-4B-DFlash-b16" \
        "$hf_store"/models--z-lab--Qwen3-4B-DFlash-b16/snapshots/*; do
        if [[ -f "$candidate/dflash.py" && -f "$candidate/config.json" ]] && compgen -G "$candidate/*.safetensors" >/dev/null; then
            DRAFT=$candidate
            break
        fi
    done
fi
if [[ -z "${DRAFT:-}" ]]; then
    echo 'Resolving the official DFlash checkpoint (downloads weights once on a cache miss).' >&2
    DRAFT=$(python - <<'PY'
from huggingface_hub import snapshot_download
print(snapshot_download('z-lab/Qwen3-4B-DFlash-b16', revision='b74e3a329c4d963783143b1e970d95b002be72bd',
                        allow_patterns=['*.py','*.json','*.safetensors']))
PY
)
fi
printf 'Python %s\nTarget %s\nDraft %s\nOutput %s\n' "$(command -v python)" "$target" "$DRAFT" "$out"
exec python "$script_dir/audit.py" --target "$target" --draft "$DRAFT" \
    --dataset "$dataset" --out "$out" --device npu:0 "$@"
