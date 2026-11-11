from dataclasses import asdict
import json
from pathlib import Path
import pytest

from four_anchor.dspark_ascend_config import build_config
from four_anchor.dspark_preflight import devices, require_free


def test_matches_new_server_recipe_and_separates_graph_scope(tmp_path):
    model, data = tmp_path / "model", tmp_path / "data"
    model.mkdir(); data.mkdir()
    (model / "config.json").write_text(json.dumps({"model_type": "qwen3", "vocab_size": 151936, "num_hidden_layers": 36}))
    (data / "state.json").write_text("{}")
    (data / "dataset_info.json").write_text(json.dumps({"features": {"input_ids": {}, "loss_mask": {}}}))
    env = {"MODEL": str(model), "DATASET": str(data), "OUTPUT_DIR": str(tmp_path / "out"),
           "TARGET_NPUS": "0,1", "TRAIN_NPUS": "2,3,4,5,6", "TRAIN_DATA_RATIO": "1.0"}
    cfg = build_config(env)
    assert (cfg.anchor_layout, cfg.num_anchors, cfg.num_proposals) == ("bonus_addon", 4, 15)
    assert (cfg.max_sequence_length, cfg.max_anchors, cfg.epochs) == (9216, 1536, 5)
    assert cfg.learning_rate == .0006 and cfg.train_data_ratio == 1.
    assert cfg.device == "npu" and cfg.distributed_strategy == "fsdp"
    assert cfg.loss_graph == "off" and cfg.loss_fn == {"ce": .1, "tv": .9}
    assert cfg.boundary_confidence == "confidence_head"
    assert cfg.confidence_alpha == 1. and cfg.markov_rank == 256
    assert cfg.epoch_benchmark_dir is None
    assert build_config({**env, "LOSS_GRAPH": "npu"}).loss_graph == "npu"
    assert build_config({**env, "BOUNDARY_CONFIDENCE": "draft_probability"}).boundary_confidence == "draft_probability"
    assert build_config({**env, "BOUNDARY_EXPLORATION": "0"}).boundary_exploration == 0.
    with pytest.raises(ValueError, match="exploration"):
        build_config({**env, "BOUNDARY_EXPLORATION": "2"})
    with pytest.raises(ValueError, match="DSpark needs"):
        build_config({**env, "EVAL_EACH_EPOCH": "1"})


@pytest.mark.parametrize("target,train", [("0,1", "1,2"), ("0,0", "2,3"), ("", "2,3"), ("0,1", "8")])
def test_invalid_allocations_rejected(target, train):
    with pytest.raises(ValueError):
        devices(target, train)


def test_busy_npu7_is_allowed_but_busy_trainer_is_not():
    text = "\n".join(f"| No running processes found in NPU {i} |" for i in range(7))
    text += "\n| 7 0 | 12345 | VLLMEngineCore | 32000 |"
    require_free(text, devices("0,1", "2,3,4,5,6"))
    with pytest.raises(ValueError, match="NPU 7"):
        require_free(text, devices("0,1", "2,3,4,5,6,7"))
    with pytest.raises(ValueError, match="NPU 2"):
        require_free(text.replace("No running processes found in NPU 2", "unknown"), ["2"])


def test_launcher_does_not_select_eval_card_or_install_packages():
    script = Path(__file__).parents[1].joinpath("scripts/train_dspark_ascend.sh").read_text()
    assert 'TRAIN_NPUS:-2,3,4,5,6}' in script
    assert 'LOSS_GRAPH:-off' in script and '-m four_anchor.dspark_train' in script
    assert 'BOUNDARY_CONFIDENCE:-confidence_head' in script
    for forbidden in ("pip install", "git switch", "git checkout", "git reset", "pkill", "killall"):
        assert forbidden not in script
    assert script.index('four_anchor.dspark_preflight') < script.index('setsid env')
    assert 'EVAL_EACH_EPOCH:-0' in script
