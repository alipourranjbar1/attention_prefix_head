"""Local inputs, dependency provenance, and resource checks; never stop a job."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys

from .verify_sources import ROOT, verify


def has_weights(path):
    files = list(Path(path).glob("*.safetensors"))
    return bool(files) and all(p.stat().st_size > 0 for p in files)


def resolve_draft():
    mode = os.environ.get("INIT_MODE", "pretrained")
    if mode not in ("pretrained", "scratch"):
        raise ValueError("INIT_MODE must be pretrained or scratch")
    if mode == "scratch":
        if os.environ.get("DRAFT_INIT"):
            raise ValueError("INIT_MODE=scratch must not have DRAFT_INIT set; unset DRAFT_INIT")
        config_dir = ROOT / "configs/qwen3_4b_dflash_scratch"
        if not (config_dir / "config.json").is_file():
            raise ValueError(f"Bundled scratch architecture is missing: {config_dir}")
        return config_dir
    explicit = os.environ.get("DRAFT_INIT")
    if explicit:
        p = Path(explicit).expanduser().resolve()
        if not (p / "config.json").is_file() or not has_weights(p):
            raise ValueError(f"DRAFT_INIT has no complete local config/weights: {p}")
        return p
    base = Path(os.environ.get("BASE", "/home/n84449292/a84384644"))
    candidates = [base / "models/Qwen3-4B-DFlash-b16"]
    caches = [base / "models", Path.home() / ".cache/huggingface/hub"]
    if os.environ.get("HF_HOME"):
        caches.append(Path(os.environ["HF_HOME"]) / "hub")
    if os.environ.get("HF_HUB_CACHE"):
        caches.append(Path(os.environ["HF_HUB_CACHE"]))
    for cache in caches:
        candidates.extend(sorted((cache / "models--z-lab--Qwen3-4B-DFlash-b16/snapshots").glob("*")))
    for p in candidates:
        if (p / "config.json").is_file() and has_weights(p):
            return p.resolve()
    raise ValueError(
        "The official recipe needs the released regular DFlash checkpoint "
        "z-lab/Qwen3-4B-DFlash-b16. Set DRAFT_INIT to its local directory. "
        "No scratch or custom-model fallback is used. To download it on a connected host, run:\n"
        'python -c \'from huggingface_hub import snapshot_download; '
        'snapshot_download("z-lab/Qwen3-4B-DFlash-b16", '
        'local_dir="/home/n84449292/a84384644/models/Qwen3-4B-DFlash-b16", '
        'allow_patterns=["*.json", "*.safetensors"])\''
    )


def validate_configs(target_path, draft_path):
    target = json.loads((target_path / "config.json").read_text())
    cfg = json.loads((draft_path / "config.json").read_text())
    if (draft_path / "four_anchor_config.json").exists():
        raise ValueError("DRAFT_INIT is a custom four-anchor checkpoint, not regular DFlash")
    if target.get("model_type") != "qwen3" or cfg.get("model_type") != "qwen3":
        raise ValueError("This launcher is for the Qwen3 DFlash host")
    dflash = cfg.get("dflash_config", {})
    if cfg.get("architectures") != ["DFlashDraftModel"] or cfg.get("block_size") != 16:
        raise ValueError("Expected regular DFlashDraftModel with block_size=16")
    if dflash.get("projector_type") or dflash.get("shift_label") or dflash.get("pure_draft_prefix_len", 0):
        raise ValueError("Expected the regular DFlash architecture")
    for key in ("hidden_size", "vocab_size"):
        if target[key] != cfg[key]:
            raise ValueError(f"Target/drafter {key} mismatch")
    taps = dflash.get("target_layer_ids")
    if not taps or not all(0 <= i < target["num_hidden_layers"] for i in taps):
        raise ValueError("Invalid target layer taps in draft checkpoint")
    if not 0 <= dflash.get("mask_token_id", -1) < cfg["vocab_size"]:
        raise ValueError("Missing/invalid released draft mask_token_id")
    return cfg


def require_free(ids, table):
    if not ids or any(not x.isdigit() for x in ids) or len(set(ids)) != len(ids):
        raise ValueError("NPUS must contain distinct comma-separated numeric IDs")
    for device in ids:
        if not re.search(r"No running processes found in NPU\s+" + re.escape(device) + r"\b", table):
            raise ValueError(f"NPU {device} occupied or unclear. No process was stopped.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolve-draft", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.resolve_draft:
        print(resolve_draft())
        return
    verify()
    model = Path(os.environ["MODEL"]).resolve()
    draft = resolve_draft()
    mode = os.environ.get("INIT_MODE", "pretrained")
    dataset = Path(os.environ["DATASET"]).resolve()
    cfg = validate_configs(model, draft)
    if not dataset.exists():
        raise FileNotFoundError(dataset)
    ids = os.environ["NPUS"].split(",")
    if not args.check_only:
        table = subprocess.check_output(["npu-smi", "info"], text=True)
        require_free(ids, table)
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", int(os.environ["MASTER_PORT"])))
    import torch
    import torch_npu
    from specforge.core.dflash import OnlineDFlashModel
    from specforge.modeling.draft.dflash import DFlashDraftModel
    from specforge.optimizer import BF16Optimizer
    from specforge.data.on_the_fly import OnTheFlyDataset
    from accelerate.utils import set_seed
    from torch.utils.tensorboard import SummaryWriter
    if dataset.is_dir():
        from .tokenized_data import SavedTokenDataset
        data = SavedTokenDataset(dataset, int(os.environ["MAX_LENGTH"]), 64)
        # Validate real data through the exact upstream short-row replacement.
        sample = data[0]
        dataset_info = {"format": "hf_save_to_disk", "rows": len(data),
            "fingerprint": data.dataset._fingerprint,
            "first_loaded_tokens": sample["input_ids"].shape[-1]}
    else:
        with dataset.open() as f:
            row = json.loads(f.readline())
        if "conversations" not in row:
            raise ValueError("Raw JSONL needs upstream conversations records")
        dataset_info = {"format": "raw_jsonl", "size_bytes": dataset.stat().st_size}
    versions = {name: importlib.metadata.version(name) for name in
                ("torch", "torch_npu", "transformers", "datasets", "accelerate")}
    report = dict(upstream_commit=(ROOT / "UPSTREAM_COMMIT").read_text().strip(),
        model=str(model), draft_init=str(draft), dataset=str(dataset), dataset_info=dataset_info,
        initialization_mode=mode,
        initialization_note=("Random drafter; frozen pretrained target embeddings/head retained"
            if mode == "scratch" else "Released pretrained regular DFlash drafter"),
        head_loss_chunk_size=int(os.environ.get("HEAD_LOSS_CHUNK_SIZE", "0")),
        packed_token_budget=int(os.environ.get("PACKED_TOKEN_BUDGET", "0")),
        max_record_length=int(os.environ["MAX_LENGTH"]),
        num_anchors=int(os.environ["NUM_ANCHORS"]),
        anchor_budget_scope=("packed_rank_batch" if int(os.environ.get("PACKED_TOKEN_BUDGET", "0")) else "individual_record"),
        activation_checkpointing=os.environ.get("ACTIVATION_CHECKPOINTING", "0") == "1",
        target_backend="hf", train_npus=ids, world_size=len(ids),
        loss="native DFlash CE on rollout token IDs", block_size=16,
        num_draft_layers=cfg["num_hidden_layers"], target_layer_ids=cfg["dflash_config"]["target_layer_ids"],
        loop_max_stage=3, versions=versions,
        draft_config_sha256=hashlib.sha256((draft / "config.json").read_bytes()).hexdigest(),
        warm_start_files=([{"name":p.name,"bytes":p.stat().st_size}
            for p in sorted(draft.glob("*.safetensors"))] if mode == "pretrained" else []))
    print(json.dumps(report, indent=2), flush=True)
    if not args.check_only:
        out = Path(os.environ["OUTPUT_DIR"])
        resume = os.environ.get("RESUME", "0") == "1"
        if out.exists() and not resume:
            raise ValueError(f"Fresh OUTPUT_DIR already exists: {out}")
        if resume and not out.is_dir():
            raise ValueError("RESUME=1 requires the original OUTPUT_DIR")
        out.mkdir(parents=True, exist_ok=resume)
        if resume:
            old = json.loads((out / "launch_provenance.json").read_text())
            if old.get("initialization_mode", "pretrained") != mode:
                raise ValueError("Resume mismatch: initialization_mode")
            if old.get("packed_token_budget", 0) != report["packed_token_budget"]:
                raise ValueError("Cannot resume an unpacked optimizer schedule into a packed run")
            for key in ("model", "draft_init", "dataset", "dataset_info", "world_size", "draft_config_sha256"):
                if old[key] != report[key]:
                    raise ValueError(f"Resume mismatch: {key}")
        else:
            (out / "launch_provenance.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
