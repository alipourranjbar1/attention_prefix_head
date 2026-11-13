"""Rank-local optimizer/RNG persistence for the upstream FSDP trainer.

The upstream checkpoint path stores only rank 0's optimizer shards; these I/O
helpers save each rank independently. They do not change the optimizer.
"""
import json
import os
from pathlib import Path
import random
import re

import numpy as np
import torch
import torch.distributed as dist


def capture_rng():
    state = {"python": random.getstate(), "numpy": np.random.get_state(),
             "torch": torch.get_rng_state()}
    if hasattr(torch, "npu") and torch.npu.is_available():
        state["npu"] = torch.npu.get_rng_state()
    elif torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state()
    return state


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "npu" in state:
        torch.npu.set_rng_state(state["npu"])
    if "cuda" in state:
        torch.cuda.set_rng_state(state["cuda"])


def save_rank_state(save_dir, args, epoch, step, optimizer):
    path = Path(save_dir)
    rank = dist.get_rank()
    state = dict(epoch=epoch, global_step=step, args=args,
                 world_size=dist.get_world_size(), rank=rank, rng=capture_rng(),
                 **optimizer.state_dict())
    temporary = path / f"training_state_rank{rank}.pt.tmp"
    torch.save(state, temporary)
    os.replace(temporary, path / f"training_state_rank{rank}.pt")


def mark_complete(save_dir):
    dist.barrier()
    if dist.get_rank() == 0:
        path = Path(save_dir)
        for rank in range(dist.get_world_size()):
            if not (path / f"training_state_rank{rank}.pt").is_file():
                raise RuntimeError(f"Missing saved optimizer state for rank {rank}")
        (path / "COMPLETE.json").write_text(json.dumps({"world_size": dist.get_world_size()}) + "\n")
        temporary = path.parent / "latest_checkpoint.txt.tmp"
        temporary.write_text(str(path.resolve()) + "\n")
        os.replace(temporary, path.parent / "latest_checkpoint.txt")
    dist.barrier()


def get_last_complete_checkpoint(folder):
    pattern = re.compile(r"epoch_(\d+)_step_(\d+)$")
    checkpoints = []
    for path in Path(folder).glob("epoch_*_step_*"):
        match = pattern.fullmatch(path.name)
        if not match or not (path / "COMPLETE.json").is_file():
            continue
        world = json.loads((path / "COMPLETE.json").read_text())["world_size"]
        if not all((path / f"training_state_rank{r}.pt").is_file() for r in range(world)):
            continue
        checkpoints.append((tuple(map(int, match.groups())), path))
    if not checkpoints:
        raise ValueError("No complete checkpoint found; resume cannot start fresh training")
    (epoch, step), path = max(checkpoints)
    world = json.loads((path / "COMPLETE.json").read_text())["world_size"]
    if dist.is_initialized() and world != dist.get_world_size():
        raise ValueError(f"Resume needs the original {world} training ranks")
    return str(path), (epoch, step)
