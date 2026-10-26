"""Ascend-aware entry point; training settings are supplied by run.yaml."""
import os
import torch

if os.environ.get('ASCEND_RT_VISIBLE_DEVICES') is not None:
    import torch_npu  # noqa: F401
    torch.npu.set_device(int(os.environ.get('LOCAL_RANK', '0')))

from speculators.train.cli import main
from speculators.train.config import TrainConfig

if __name__ == '__main__':
    main(TrainConfig.resolve())
