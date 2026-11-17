"""Small real-FSDP preflight, run on trainer NPUs before the target server starts.

Exercises different block counts per rank, backward, clipping, optimizer shards,
full checkpoint export, and an equivalent continuation after restore.
"""
from pathlib import Path
import argparse
import json
import os
import random
from datetime import timedelta

import torch
import torch.distributed as dist
from transformers import Qwen3Config, Qwen3ForCausalLM

from .distributed import wrap_task, checkpoint_states, restore_optimizer, clip_grad_norm
from .model import DraftConfig, FourAnchorDFlash
from .packed_objective import PackedTask
from .runtime import resolve_device
from .train import TrainConfig


def run_check(device, output, dtype="bfloat16", packed_execution="cached", anchor_layout="past"):
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.manual_seed(193)
    # Keep a conventional accelerator attention head size (128), while making
    # depth/vocabulary small enough for a brief hardware compatibility check.
    config = Qwen3Config(vocab_size=64, hidden_size=512, intermediate_size=1024, num_hidden_layers=2,
                        num_attention_heads=4, num_key_value_heads=2, head_dim=128,
                        max_position_embeddings=128, eos_token_id=None, pad_token_id=0)
    config._attn_implementation = "sdpa"
    target = Qwen3ForCausalLM(config).to(device=device, dtype=getattr(torch, dtype)).eval().requires_grad_(False)
    proposals = 15 if anchor_layout != "past" else 4
    draft = FourAnchorDFlash(DraftConfig("fsdp-preflight", [0, 1], 63, num_draft_layers=2, num_proposals=proposals,
                                       anchor_layout=anchor_layout,
                                       format_version={"past": 1, "bonus_in_four": 2, "target_four": 3, "bonus_addon": 4}[anchor_layout]), target)
    for p in draft.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    draft.gradient_checkpointing = True
    cfg = TrainConfig("fsdp-preflight", "synthetic", str(output), 63, [0, 1], device=str(device), dtype=dtype,
                      training_mode="packed", distributed_strategy="fsdp", epochs=1, num_workers=0,
                      num_proposals=proposals, max_anchors=16, blocks_per_chunk=2, head_chunk_size=7,
                      anchor_layout=anchor_layout,
                      max_sequence_length=32, loss_fn={"ce": .1, "tv": .9}, packed_execution=packed_execution)
    task = PackedTask(draft, target.model.norm, cfg, random.Random(321 + rank)).train()
    full_elements = sum(p.numel() for p in draft.parameters() if p.requires_grad)
    wrapped = wrap_task(task, "fsdp", device, dtype, world)
    local_elements = sum(p.numel() for p in draft.parameters() if p.requires_grad)
    if world > 1 and local_elements >= full_elements:
        raise AssertionError("Expected sharded trainable parameters")
    optimizer = torch.optim.AdamW([p for p in draft.parameters() if p.requires_grad], lr=.0006)
    n = 7 + rank
    batch = {"ids": torch.randint(1, 62, (n,), device=device),
             "weights": torch.ones(n, device=device),
             "features": torch.randn(n, config.hidden_size * 2, device=device, dtype=getattr(torch, dtype)),
             "final": torch.randn(n, config.hidden_size, device=device, dtype=getattr(torch, dtype)),
             "positions": torch.arange(n, device=device),
             "documents": torch.zeros(n, device=device, dtype=torch.long),
             "offsets": torch.zeros(1, device=device, dtype=torch.long),
             "lengths": torch.tensor([n], device=device)}

    def step(model, drafter, opt):
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=dtype == "bfloat16"):
            numerator, denominator, metrics = model(batch)
        numerator.backward()
        dist.all_reduce(denominator)
        for p in drafter.parameters():
            if p.grad is not None:
                p.grad.mul_(world / denominator)
        norm = clip_grad_norm(model, drafter, 1.)
        opt.step()
        return float(norm)

    norm = step(wrapped, draft, optimizer)
    rng_state = task.rng.getstate()
    saved_model, saved_optimizer = checkpoint_states(wrapped, draft, optimizer)
    output = Path(output)
    if rank == 0:
        draft.save(output, state_dict=saved_model)
        torch.save(saved_optimizer, output / "optimizer.pt")
    dist.barrier()
    del saved_model, saved_optimizer
    step(wrapped, draft, optimizer)
    expected, _ = checkpoint_states(wrapped, draft, optimizer)
    del optimizer, wrapped, task, draft

    restored = FourAnchorDFlash.load(output, target, trainable_dtype=torch.float32)
    restored.gradient_checkpointing = True
    task2 = PackedTask(restored, target.model.norm, cfg, random.Random()).train()
    task2.rng.setstate(rng_state)
    wrapped2 = wrap_task(task2, "fsdp", device, dtype, world)
    optimizer2 = torch.optim.AdamW([p for p in restored.parameters() if p.requires_grad], lr=.0006)
    state = torch.load(output / "optimizer.pt", map_location="cpu", weights_only=True) if rank == 0 else {}
    restore_optimizer(wrapped2, optimizer2, state)
    step(wrapped2, restored, optimizer2)
    actual, _ = checkpoint_states(wrapped2, restored, optimizer2)
    if rank == 0:
        assert actual.keys() == expected.keys()
        for key in actual:
            torch.testing.assert_close(actual[key], expected[key], atol=1e-6, rtol=1e-5, msg=key)
        result = {"status": "passed", "world_size": world, "device": str(device), "dtype": dtype,
                  "sharding": "FULL_SHARD" if world > 1 else "NO_SHARD (single-rank unit test)",
                  "rank0_trainable_elements": local_elements, "full_trainable_elements": full_elements,
                  "first_global_grad_norm": norm, "checkpoint_continuation": "matched", "packed_execution": packed_execution}
        (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
    dist.barrier()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="npu")
    parser.add_argument("--output", required=True)
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="bfloat16")
    parser.add_argument("--packed-execution", choices=["reference", "cached"], default="cached")
    parser.add_argument("--anchor-layout", choices=["past", "bonus_in_four", "target_four", "bonus_addon"], default="past")
    args = parser.parse_args()
    device = resolve_device(f"{args.device}:{os.environ.get('LOCAL_RANK', '0')}" if args.device != "cpu" else "cpu")
    dist.init_process_group(backend={"npu": "hccl", "cuda": "nccl"}.get(device.type, "gloo"),
                            timeout=timedelta(seconds=180))
    try:
        run_check(device, args.output, args.dtype, args.packed_execution, args.anchor_layout)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
