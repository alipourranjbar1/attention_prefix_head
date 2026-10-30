"""Build a validated train configuration from the Ascend shell launcher's env."""
from dataclasses import asdict
from pathlib import Path
import json
import os

from .train import TrainConfig


def build_config(env):
    target = Path(env["MODEL"]).expanduser().resolve()
    data = Path(env["DATASET"]).expanduser().resolve()
    model = json.loads((target / "config.json").read_text())
    if model.get("model_type") != "qwen3" or model.get("vocab_size") != 151936 or model.get("num_hidden_layers", 0) <= 33:
        raise ValueError("Expected a Qwen3-4B checkpoint exposing decoder layers 1,9,17,25,33 and vocab 151936")
    if not (data / "state.json").exists():
        raise ValueError("DATASET must be a preprocessed HF save_to_disk Dataset")
    features = json.loads((data / "dataset_info.json").read_text())["features"]
    if not {"input_ids", "loss_mask"} <= set(features):
        raise ValueError("Dataset needs input_ids and loss_mask")
    target_devices, train_devices = env["TARGET_NPUS"].split(","), env["TRAIN_NPUS"].split(",")
    devices = target_devices + train_devices
    if not all(d.isdigit() for d in devices) or len(set(devices)) != len(devices):
        raise ValueError("TARGET_NPUS and TRAIN_NPUS must be nonempty, disjoint, distinct numeric IDs")
    output = Path(env["OUTPUT_DIR"]).expanduser().resolve()
    config = TrainConfig(
        target=str(target), data=str(data), output=str(output / "checkpoints"),
        mask_token_id=151669, target_layer_ids=[1, 9, 17, 25, 33],
        device="npu", dtype="bfloat16", attention="sdpa", num_draft_layers=5,
        num_anchors=4, num_proposals=int(env.get("BLOCK_SIZE", "16")),
        max_sequence_length=int(env.get("SEQ_LENGTH", "3072")),
        epochs=int(env.get("EPOCHS", "5")), learning_rate=float(env.get("LR", "0.0006")),
        seed=int(env.get("SEED", "42")), gradient_accumulation=int(env.get("GRADIENT_ACCUMULATION", "1")),
        warmup_steps=int(env.get("WARMUP_STEPS", "50")), log_every=int(env.get("LOG_FREQ", "10")),
        checkpoint_each_epoch=True, tensorboard=True,
        max_steps=int(env["MAX_STEPS"]) if env.get("MAX_STEPS") else None,
        vllm_endpoint=f"http://127.0.0.1:{int(env.get('PORT', '8080'))}/v1",
        hidden_states_dir=str(output / "hidden_states"), request_timeout=180, max_retries=8,
        initialize_dflash=env.get("DFLASH_INIT") or None, resume=env.get("RESUME") or None,
    )
    return config


if __name__ == "__main__":
    cfg = build_config(os.environ)
    out = Path(os.environ["OUTPUT_DIR"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "train_config.json").write_text(json.dumps(asdict(cfg), indent=2) + "\n")
    settings = {"target_npus": os.environ["TARGET_NPUS"], "train_npus": os.environ["TRAIN_NPUS"],
                "conditioning_anchors": 4, "new_tokens_per_pass": cfg.num_proposals,
                "query_positions": 4 + cfg.num_proposals, "training_blocks_per_sequence": 1,
                "legacy_max_anchors": os.environ.get("MAX_ANCHORS", "512"),
                "legacy_max_anchors_applied": False, "hf_decoder_layer_ids": cfg.target_layer_ids,
                "vllm_capture_ids": [i + 1 for i in cfg.target_layer_ids], "last_layer_causal": False,
                "model": cfg.target, "dataset": cfg.data, "epochs": cfg.epochs,
                "learning_rate": cfg.learning_rate, "sequence_length": cfg.max_sequence_length}
    (out / "launch_settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    print(json.dumps(settings, indent=2), flush=True)
