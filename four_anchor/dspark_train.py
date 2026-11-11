"""Ascend/CUDA entrypoint for four-anchor, two-round DSpark with a local or vLLM teacher."""
from dataclasses import dataclass, asdict
from pathlib import Path
import argparse
import json
import math
import os

import torch

from .train import TrainConfig


@dataclass
class DSparkTrainConfig(TrainConfig):
    draft_type: str = "dspark"
    markov_rank: int = 256
    confidence_alpha: float = 1.0
    boundary_confidence: str = "confidence_head"
    training_mode: str = "packed"
    anchor_layout: str = "bonus_addon"
    num_proposals: int = 15
    epochs: int | None = 5
    gradient_accumulation: int = 1
    learning_rate: float = 0.0006
    max_sequence_length: int = 3072
    max_anchors: int = 64
    blocks_per_chunk: int = 4
    head_chunk_size: int = 16
    train_data_ratio: float = 1.0
    num_workers: int = 0
    checkpoint_each_epoch: bool = True
    packed_execution: str = "cached"
    tensorboard: bool = True
    loss_graph: str = "off"
    head_loss_backend: str = "checkpoint"
    benchmark_only: bool = False
    proposal_base_batched: bool = False

    def __post_init__(self):
        super().__post_init__()
        if (self.draft_type, self.training_mode, self.anchor_layout) != ("dspark", "packed", "bonus_addon"):
            raise ValueError("DSpark requires packed bonus_addon training")
        if self.markov_rank < 1 or not math.isfinite(self.confidence_alpha) or self.confidence_alpha <= 0:
            raise ValueError("Markov rank and confidence weight must be positive")
        if self.boundary_confidence not in {"draft_probability", "confidence_head"}:
            raise ValueError("boundary_confidence must be draft_probability or confidence_head")
        if self.initialize_dflash:
            raise ValueError("Start a fresh DSpark run, or RESUME its own FourAnchorDSpark checkpoint")
        if self.num_workers and not self.vllm_endpoint:
            raise ValueError("Local frozen teacher requires num_workers=0")
        if self.loss_graph not in {"off", "npu"}:
            raise ValueError("LOSS_GRAPH must be off or npu")
        if self.head_loss_backend not in {"checkpoint", "cached_vjp"}:
            raise ValueError("HEAD_LOSS_BACKEND must be checkpoint or cached_vjp")
        if self.head_loss_backend == "cached_vjp" and self.loss_graph != "off":
            raise ValueError("cached_vjp requires LOSS_GRAPH=off")
        if self.benchmark_only and (self.resume or not self.max_steps or self.max_steps > 100):
            raise ValueError("Benchmark mode requires a fresh run and MAX_STEPS in 1..100")
        if self.loss_graph == "npu" and self.device.split(":")[0] != "npu":
            raise ValueError("LOSS_GRAPH=npu requires an Ascend device")
        if self.epoch_benchmark_dir:
            raise ValueError("Existing DFlash evaluators cannot load FourAnchorDSpark; automatic evaluation is disabled")
        if self.device.split(":")[0] not in {"cuda", "cpu", "npu"}:
            raise ValueError("Use npu, cuda, or cpu")


def build_config(env):
    required = ("MODEL", "DATASET", "OUTPUT_DIR")
    missing = [key for key in required if not env.get(key)]
    if missing:
        raise ValueError("Set paths on this server: " + ", ".join(missing))
    values = dict(target=str(Path(env["MODEL"]).expanduser().resolve()),
                  data=str(Path(env["DATASET"]).expanduser().resolve()),
                  output=str(Path(env["OUTPUT_DIR"]).expanduser().resolve()),
                  mask_token_id=int(env.get("MASK_TOKEN_ID", "151669")),
                  target_layer_ids=[int(x) for x in env.get("TARGET_LAYER_IDS", "1,9,17,25,33").split(",")],
                  loss_fn={"ce": .1, "tv": .9})
    integers = {"EPOCHS": "epochs", "SEQ_LENGTH": "max_sequence_length", "MAX_ANCHORS": "max_anchors",
                "BLOCKS_PER_CHUNK": "blocks_per_chunk", "HEAD_CHUNK_SIZE": "head_chunk_size",
                "GRAD_ACCUM": "gradient_accumulation", "DRAFT_LAYERS": "num_draft_layers",
                "MARKOV_RANK": "markov_rank", "WARMUP_STEPS": "warmup_steps", "SEED": "seed",
                "MAX_STEPS": "max_steps", "PROFILE_STEPS": "profile_steps",
                "NUM_WORKERS": "num_workers", "REQUEST_TIMEOUT": "request_timeout",
                "MAX_RETRIES": "max_retries"}
    decimals = {"LR": "learning_rate", "CONFIDENCE_ALPHA": "confidence_alpha",
                "TRAIN_DATA_RATIO": "train_data_ratio", "BOUNDARY_EXPLORATION": "boundary_exploration",
                "BOUNDARY_BETA": "boundary_beta"}
    strings = {"DISTRIBUTED_STRATEGY": "distributed_strategy", "BOUNDARY_CONFIDENCE": "boundary_confidence",
               "RESUME": "resume", "DATA_CACHE_DIR": "data_cache_dir",
               "VLLM_ENDPOINT": "vllm_endpoint", "HIDDEN_STATES_DIR": "hidden_states_dir",
               "TEACHER_MODEL_NAME": "teacher_model_name", "LOSS_GRAPH": "loss_graph", "DEVICE": "device",
               "HEAD_LOSS_BACKEND": "head_loss_backend"}
    for mapping, cast in ((integers, int), (decimals, float), (strings, str)):
        for key, field in mapping.items():
            if env.get(key):
                values[field] = cast(env[key])
    for key, field in (("GRADIENT_CHECKPOINTING", "gradient_checkpointing"),
                       ("PROPOSAL_BASE_BATCHED", "proposal_base_batched")):
        if key in env:
            if env[key] not in {"0", "1"}:
                raise ValueError(f"{key} must be 0 or 1")
            values[field] = env[key] == "1"
    if env.get("BENCHMARK_ONLY", "0") not in {"0", "1"}:
        raise ValueError("BENCHMARK_ONLY must be 0 or 1")
    values["benchmark_only"] = env.get("BENCHMARK_ONLY", "0") == "1"
    return DSparkTrainConfig(**values)


def validate_inputs(cfg, metadata_only=False):
    from .data import load_records
    from transformers import AutoConfig
    if not Path(cfg.target).is_dir() or not (Path(cfg.target) / "config.json").is_file():
        raise ValueError("MODEL must be the local Qwen3-4B directory containing config.json and weights")
    if not Path(cfg.data).is_dir():
        raise ValueError("DATASET must be a local Hugging Face save_to_disk directory")
    tc = AutoConfig.from_pretrained(cfg.target, local_files_only=True, trust_remote_code=False)
    if tc.model_type != "qwen3" or any(i < 0 or i >= tc.num_hidden_layers for i in cfg.target_layer_ids):
        raise ValueError("Target must be Qwen3 with valid target-layer indices")
    if not 0 <= cfg.mask_token_id < tc.vocab_size:
        raise ValueError("MASK_TOKEN_ID is outside the target vocabulary")
    if cfg.max_sequence_length + 30 > tc.max_position_embeddings:
        raise ValueError("SEQ_LENGTH exceeds the supported target context")
    data = load_records(cfg.data, cfg.split)
    if not len(data) or not {"input_ids", "loss_mask"}.issubset(data.column_names):
        raise ValueError("Dataset must contain nonempty input_ids and loss_mask columns")
    row = data[0]
    if len(row["input_ids"]) != len(row["loss_mask"]):
        raise ValueError("Dataset token IDs and loss masks must have matching lengths")
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if cfg.distributed_strategy == "fsdp" and world < 2:
        raise ValueError("FSDP needs at least two selected training GPUs")
    if cfg.device.startswith("npu") and not metadata_only:
        import torch_npu
        if not torch.npu.is_available():
            raise RuntimeError("Ascend NPU is unavailable")
        torch.npu.set_device(int(os.environ.get("LOCAL_RANK", "0")))
        if cfg.loss_graph == "npu" and not hasattr(torch.npu, "NPUGraph"):
            raise RuntimeError("This torch_npu build does not expose NPUGraph")
    if cfg.device.startswith("cuda") and not metadata_only:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable: check the NVIDIA driver and CUDA PyTorch installation")
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        if cfg.dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("The selected GPU does not support BF16")
    return {"dataset_rows": len(data), "world_size": world, "model": cfg.target,
            "architecture": "FourAnchorDSpark", "markov_rank": cfg.markov_rank,
            "round_1_queries": 20, "round_2_queries": 19, "speculative_tokens": 15,
            "boundary_confidence": cfg.boundary_confidence}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="Saved DSpark JSON recipe; otherwise use environment variables")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--plan", action="store_true", help="Count packed batches without loading model weights")
    args = parser.parse_args(argv)
    cfg = DSparkTrainConfig(**json.loads(Path(args.config).read_text())) if args.config else build_config(os.environ)
    details = validate_inputs(cfg, metadata_only=args.plan or args.check_only)
    if int(os.environ.get("RANK", "0")) == 0:
        print(json.dumps(details, indent=2), flush=True)
    if args.check_only:
        return
    if args.plan:
        from .packed_data import prepare_records, schedule
        data, lengths, summary = prepare_records(cfg)
        counts = schedule(cfg, data, lengths, details["world_size"])
        print(json.dumps({**summary, "steps_per_epoch": counts, "total_steps": sum(counts)}, indent=2))
        return
    from .packed_train import train_packed
    saved = train_packed(cfg)
    print("Benchmark completed; no training checkpoint saved" if cfg.benchmark_only else f"Saved checkpoint: {saved}", flush=True)


if __name__ == "__main__":
    main()
