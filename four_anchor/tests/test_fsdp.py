import json

import pytest
import torch
import torch.distributed as dist

from four_anchor.fsdp_check import run_check
from four_anchor.train import TrainConfig
from four_anchor.packed_train import train_packed


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize("layout", ["past", "bonus_addon"])
def test_fsdp_hooks_full_checkpoint_optimizer_conversion_and_resume_single_rank(tmp_path, dtype, layout):
    # No networking is permitted here. These are real FSDP module/state APIs
    # with a single-rank in-process collective backend, so NO_SHARD only.
    # The launcher executes the same check with real HCCL and FULL_SHARD.
    import torch.testing._internal.distributed.fake_pg  # noqa: F401
    torch.set_num_threads(1)
    dist.init_process_group("fake", rank=0, world_size=1)
    try:
        run_check(torch.device("cpu"), tmp_path, dtype, anchor_layout=layout)
    finally:
        dist.destroy_process_group()
    result = json.loads((tmp_path / "result.json").read_text())
    assert result["status"] == "passed"
    assert result["world_size"] == 1
    assert result["checkpoint_continuation"] == "matched"


def test_fsdp_requires_multiple_trainers_before_loading_data(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "1")
    cfg = TrainConfig("tiny", "unused", "unused", 63, [0, 1], training_mode="packed",
                      epochs=1, distributed_strategy="fsdp")
    with pytest.raises(ValueError, match="at least two"):
        train_packed(cfg)


def test_fsdp_not_silently_enabled_for_legacy_online_trainer():
    with pytest.raises(ValueError, match="packed trainer"):
        TrainConfig("tiny", "unused", "unused", 63, [0, 1], distributed_strategy="fsdp")
