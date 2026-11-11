"""Build the Ascend DSpark recipe from the existing bonus20 recipe."""
from dataclasses import asdict
from pathlib import Path
import json
import os

from .ascend_config import build_config as build_dflash_config
from .dspark_train import DSparkTrainConfig


def build_config(env):
    if env.get("EVAL_EACH_EPOCH", "0") != "0" or env.get("EPOCH_BENCHMARK_DIR"):
        raise ValueError("DSpark needs its own serving adapter; leave EVAL_EACH_EPOCH=0")
    base = build_dflash_config({**env, "ANCHOR_LAYOUT": "bonus_addon", "BLOCK_SIZE": "15"})
    values = asdict(base)
    values["boundary_exploration"] = float(env.get("BOUNDARY_EXPLORATION", base.boundary_exploration))
    values["boundary_beta"] = float(env.get("BOUNDARY_BETA", base.boundary_beta))
    gc = env.get("GRADIENT_CHECKPOINTING", "1")
    if gc not in {"0", "1"}:
        raise ValueError("GRADIENT_CHECKPOINTING must be 0 or 1")
    values["gradient_checkpointing"] = gc == "1"
    if env.get("BENCHMARK_ONLY", "0") not in {"0", "1"}:
        raise ValueError("BENCHMARK_ONLY must be 0 or 1")
    if env.get("PROPOSAL_BASE_BATCHED", "0") not in {"0", "1"}:
        raise ValueError("PROPOSAL_BASE_BATCHED must be 0 or 1")
    return DSparkTrainConfig(**values, markov_rank=int(env.get("MARKOV_RANK", "256")),
        confidence_alpha=float(env.get("CONFIDENCE_ALPHA", "1.0")),
        boundary_confidence=env.get("BOUNDARY_CONFIDENCE", "confidence_head"),
        loss_graph=env.get("LOSS_GRAPH", "off"),
        head_loss_backend=env.get("HEAD_LOSS_BACKEND", "checkpoint"),
        benchmark_only=env.get("BENCHMARK_ONLY", "0") == "1",
        proposal_base_batched=env.get("PROPOSAL_BASE_BATCHED", "0") == "1")


def main():
    cfg = build_config(os.environ)
    folder = Path(os.environ["OUTPUT_DIR"]).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "train_config.json"
    # Avoid overwriting another recipe/log directory by accident.
    if path.exists() and not cfg.resume:
        raise ValueError("OUTPUT_DIR already contains a recipe; choose a new OUTPUT_DIR or set RESUME")
    if cfg.resume:
        saved = json.loads((Path(cfg.resume) / "four_anchor_config.json").read_text())
        if saved.get("architecture") != "FourAnchorDSpark":
            raise ValueError("RESUME must be a FourAnchorDSpark training checkpoint")
        for name in ("trainer.pt", "model.safetensors"):
            if not (Path(cfg.resume) / name).is_file():
                raise ValueError(f"Incomplete resume checkpoint: {name}")
        if path.exists():
            from datetime import datetime
            backup = folder / ("train_config_before_resume_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".json")
            backup.write_bytes(path.read_bytes())
    path.write_text(json.dumps(asdict(cfg), indent=2) + "\n")
    settings = {"architecture": "FourAnchorDSpark", "round_one_queries": 20, "round_two_queries": 19,
        "target_npus": os.environ["TARGET_NPUS"], "train_npus": os.environ["TRAIN_NPUS"],
        "conditioning_anchors": 4, "num_proposals": 15, "sequence_length_per_rank": cfg.max_sequence_length,
        "max_blocks_per_rank": cfg.max_anchors, "loss_fn": cfg.loss_fn, "confidence_alpha": cfg.confidence_alpha,
        "boundary_confidence": cfg.boundary_confidence,
        "boundary_exploration": cfg.boundary_exploration,
        "boundary_mismatch_cap": True,
        "loss_graph": cfg.loss_graph,
        "graph_scope": "disabled" if cfg.loss_graph == "off" else "distribution loss and first derivatives only",
        "target_graph_requested": os.environ.get("TARGET_EXECUTION", "eager") == "graph",
        "target_graph_runtime_verified": False,
        "backbone_graph": False, "distributed_strategy": cfg.distributed_strategy,
        "gradient_checkpointing": cfg.gradient_checkpointing,
        "head_loss_backend": cfg.head_loss_backend,
        "proposal_base_batched": cfg.proposal_base_batched,
        "epochs": cfg.epochs, "learning_rate": cfg.learning_rate, "train_data_ratio": cfg.train_data_ratio}
    (folder / "launch_settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    print(json.dumps(settings, indent=2), flush=True)


if __name__ == "__main__":
    main()
