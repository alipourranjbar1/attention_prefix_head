from dataclasses import replace
import importlib
import json

import pytest
import torch
from datasets import Dataset
from safetensors.torch import load_file

from four_anchor.ascend_config import build_config
from four_anchor.train import TrainConfig


def test_epoch_coverage_partial_accumulation_and_exact_resume(components, tmp_path, monkeypatch):
    target, _, _ = components
    module = importlib.import_module("four_anchor.train")
    data = Dataset.from_list([{"input_ids": [i + 1, 7, 8, 9, 10, 11, 12],
                               "loss_mask": [0, 0, 1, 1, 1, 1, 1]} for i in range(5)])
    monkeypatch.setattr(module, "load_target", lambda *a, **kw: (target, None))
    monkeypatch.setattr(module, "load_records", lambda *a, **kw: data)
    seen = []
    real_example = module.training_example
    def traced(*args, **kwargs):
        seen.append(kwargs["row_index"])
        return real_example(*args, **kwargs)
    monkeypatch.setattr(module, "training_example", traced)
    cfg = TrainConfig("tiny", "unused", str(tmp_path / "full"), 63, [0, 1], device="cpu", dtype="float32",
                      num_draft_layers=2, epochs=2, gradient_accumulation=2, max_sequence_length=32,
                      checkpoint_each_epoch=True, tensorboard=True, warmup_steps=0)
    module.train(cfg)
    assert len(seen) == 10
    assert sorted(seen[:5]) == sorted(seen[5:]) == list(range(5))
    assert (tmp_path / "full/checkpoint-000003/model.safetensors").exists()
    assert (tmp_path / "full/checkpoint-000006/model.safetensors").exists()
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    events = EventAccumulator(str(tmp_path / "full/logs/four_anchor_dflash")).Reload()
    assert [event.step for event in events.Scalars("train/loss")] == list(range(1, 7))
    early = replace(cfg, output=str(tmp_path / "early"), max_steps=2)
    module.train(early)
    resumed = replace(cfg, output=str(tmp_path / "resumed"), resume=str(tmp_path / "early/checkpoint-000002"))
    module.train(resumed)
    full = load_file(str(tmp_path / "full/checkpoint-000006/model.safetensors"))
    actual = load_file(str(tmp_path / "resumed/checkpoint-000006/model.safetensors"))
    assert all(torch.equal(full[k], actual[k]) for k in full)


def test_ascend_recipe_matches_user_settings(tmp_path):
    model, data = tmp_path / "model", tmp_path / "data"
    model.mkdir()
    data.mkdir()
    (model / "config.json").write_text(json.dumps({"model_type": "qwen3", "vocab_size": 151936, "num_hidden_layers": 36}))
    (data / "state.json").write_text("{}")
    (data / "dataset_info.json").write_text(json.dumps({"features": {"input_ids": {}, "loss_mask": {}}}))
    env = {"MODEL": str(model), "DATASET": str(data), "OUTPUT_DIR": str(tmp_path / "out"),
           "TARGET_NPUS": "0,1", "TRAIN_NPUS": "2,3,4,5,6,7", "MAX_ANCHORS": "512"}
    cfg = build_config(env)
    assert (cfg.epochs, cfg.learning_rate, cfg.max_sequence_length) == (5, 0.0006, 3072)
    assert (cfg.num_anchors, cfg.num_proposals, cfg.num_draft_layers) == (4, 16, 5)
    assert cfg.target_layer_ids == [1, 9, 17, 25, 33]
    assert cfg.vllm_endpoint == "http://127.0.0.1:8080/v1"
    assert cfg.tensorboard and cfg.checkpoint_each_epoch
    with pytest.raises(ValueError, match="disjoint"):
        build_config({**env, "TRAIN_NPUS": "1,2"})
