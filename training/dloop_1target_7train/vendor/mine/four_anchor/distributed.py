"""DDP/FSDP ownership and full portable checkpoint conversion.

FSDP wraps the whole PackedTask once. Do not auto-wrap individual decoder layers:
the number of sampled chunks and redraft calls can differ across ranks. A layer
wrapper would issue data-dependent collectives and can deadlock. Root wrapping
keeps one synchronized forward/backward boundary per packed microbatch.
"""
from contextlib import nullcontext

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP, ShardingStrategy, MixedPrecision,
    StateDictType, FullStateDictConfig, FullOptimStateDictConfig,
)


def wrap_task(task, strategy, device, dtype, world):
    if strategy == "fsdp":
        if not dist.is_initialized():
            raise ValueError("FSDP requires an initialized process group")
        # Keep the borrowed frozen target modules replicated and out of the
        # flattened FP32 trainable shard (their native weights may be BF16).
        frozen = [p for p in task.parameters() if not p.requires_grad]
        return FSDP(task, sharding_strategy=ShardingStrategy.FULL_SHARD,
                    device_id=device, use_orig_params=True, ignored_states=frozen,
                    sync_module_states=device.type != "cpu", forward_prefetch=False,
                    backward_prefetch=None, limit_all_gathers=True,
                    mixed_precision=MixedPrecision(param_dtype=getattr(torch, dtype),
                        reduce_dtype=torch.float32, keep_low_precision_grads=False,
                        cast_forward_inputs=False, cast_root_forward_inputs=False))
    return DistributedDataParallel(task, device_ids=[device.index] if device.type != "cpu" else None) if world > 1 else task


def full_state_context(wrapped):
    if not isinstance(wrapped, FSDP):
        return nullcontext()
    # CPU FSDP tests need no CPU offload (the tensors already reside there).
    offload = next(wrapped.parameters()).device.type != "cpu"
    return FSDP.state_dict_type(wrapped, StateDictType.FULL_STATE_DICT,
                                FullStateDictConfig(offload_to_cpu=offload, rank0_only=True),
                                FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True))


def checkpoint_states(wrapped, draft, optimizer):
    """ALL ranks enter; only rank zero receives full FSDP state dictionaries."""
    if not isinstance(wrapped, FSDP):
        return None, optimizer.state_dict()
    with full_state_context(wrapped):
        state = wrapped.state_dict()
        optim = FSDP.optim_state_dict(wrapped, optimizer)
    draft_state = {k.removeprefix("drafter."): v for k, v in state.items()
                   if k.startswith("drafter.") and not k.startswith(("drafter.embed_tokens.", "drafter.lm_head."))}
    return draft_state, optim


def restore_optimizer(wrapped, optimizer, state):
    if isinstance(wrapped, FSDP):
        # The full dictionary is CPU-loaded on rank zero only. PyTorch scatters
        # the appropriate optimizer tensors to the other training ranks.
        with full_state_context(wrapped):
            state = FSDP.optim_state_dict_to_load(wrapped, optimizer, state)
    optimizer.load_state_dict(state)


def clip_grad_norm(wrapped, draft, max_norm):
    if isinstance(wrapped, FSDP):
        # This computes one global norm across parameter shards on all ranks.
        norm = wrapped.clip_grad_norm_(max_norm)
        if not torch.isfinite(norm):
            raise FloatingPointError("Nonfinite global FSDP gradient norm")
        return norm
    return torch.nn.utils.clip_grad_norm_(draft.parameters(), max_norm, error_if_nonfinite=True)
