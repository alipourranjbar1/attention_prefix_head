"""Packed two-pass trainer, with exact step plans, per-epoch bars, and resume."""
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
import json
import math
import os
import random
import time

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .model import DraftConfig, FourAnchorDFlash
from .packed_data import TeacherRecords, collate_documents, epoch_batches, prepare_records, schedule
from .packed_objective import PackedTask
from .provenance import record, checkpoint_hashes
from .runtime import TargetRunner, load_target, resolve_device, synchronize
from .distributed import wrap_task, checkpoint_states, restore_optimizer, clip_grad_norm


def metric_row(metrics):
    values = metrics.tolist()
    num, den = values[:2]
    ce1, tv1, kl1, w1 = values[2:6]
    ce2, tv2, kl2, w2 = values[6:10]
    kept, predicted, blocks = values[10:13]
    return {"loss": num / max(den, 1e-8),
            "first_ce": ce1 / max(w1, 1e-8), "first_tv": tv1 / max(w1, 1e-8),
            "first_kl": kl1 / max(w1, 1e-8),
            "second_ce": ce2 / max(w2, 1e-8), "second_tv": tv2 / max(w2, 1e-8),
            "second_kl": kl2 / max(w2, 1e-8),
            "boundary_mean": kept / max(blocks, 1), "predicted_boundary_mean": predicted / max(blocks, 1),
            "training_blocks": blocks}


def memory_metrics(device):
    if device.type not in {"cuda", "npu"}:
        return {}
    backend = getattr(torch, device.type)
    return {"rank0_" + label + "_gib": getattr(backend, fn)(device) / 2**30
            for label, fn in (("memory_allocated", "memory_allocated"),
                              ("memory_reserved", "memory_reserved"),
                              ("peak_memory_allocated", "max_memory_allocated"))}


def train_packed(cfg):
    from .train import TrainConfig, _rng_state, _restore_rng
    world, rank, local_rank = (int(os.environ.get(k, d)) for k, d in
                               [("WORLD_SIZE", "1"), ("RANK", "0"), ("LOCAL_RANK", "0")])
    if cfg.distributed_strategy == "fsdp" and world < 2:
        raise ValueError("FSDP requires torchrun with at least two training ranks")
    data, lengths, data_summary = prepare_records(cfg)
    # Deterministic plans use the same packing algorithm on every rank. Cached
    # rank-zero counts establish the actual scheduler length before model setup.
    counts = schedule(cfg, data, lengths, world, verbose=rank == 0)
    total_steps = sum(counts)
    stop_step = min(cfg.max_steps or total_steps, total_steps)
    device_name = cfg.device
    if world > 1 and cfg.device.split(":")[0] in {"cuda", "npu"}:
        device_name = f"{cfg.device.split(':')[0]}:{local_rank}"
    device = resolve_device(device_name)
    if world > 1:
        dist.init_process_group(backend={"npu": "hccl", "cuda": "nccl"}.get(device.type, "gloo"))
    torch.manual_seed(cfg.seed)
    rng = random.Random(cfg.seed + rank)
    if cfg.vllm_endpoint:
        from .remote_teacher import FrozenTargetInterface
        target = FrozenTargetInterface(cfg.target, device, getattr(torch, cfg.dtype), include_final=True)
        target_norm = target.final_norm
        runner = None
    else:
        if cfg.num_workers:
            raise ValueError("Local target training requires num_workers=0; worker prefetch needs the remote teacher")
        target, _ = load_target(cfg.target, str(device), cfg.dtype, cfg.target_revision, cfg.attention)
        final = target.config.num_hidden_layers - 1
        capture_ids = cfg.target_layer_ids + ([final] if final not in cfg.target_layer_ids else [])
        runner = TargetRunner(target, capture_ids)
        target_norm = target.model.norm
    if cfg.max_sequence_length + 2 * cfg.num_proposals > target.config.max_position_embeddings:
        raise ValueError("Target context window must allow two proposal blocks beyond the sequence length")
    draft_cfg = DraftConfig(cfg.target, cfg.target_layer_ids, cfg.mask_token_id, cfg.num_draft_layers,
                            cfg.num_anchors, cfg.num_proposals, cfg.attention, cfg.target_revision,
                            format_version={"past": 1, "bonus_in_four": 2, "target_four": 3}[cfg.anchor_layout],
                            anchor_layout=cfg.anchor_layout)
    draft = FourAnchorDFlash.load(cfg.resume, target, trainable_dtype=torch.float32) if cfg.resume else FourAnchorDFlash(draft_cfg, target)
    if asdict(draft.cfg) != asdict(draft_cfg):
        raise ValueError("Resume model configuration differs")
    if cfg.initialize_dflash:
        draft.initialize_regular_dflash(cfg.initialize_dflash)
    draft.gradient_checkpointing = cfg.gradient_checkpointing
    for parameter in draft.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.float()
    task = PackedTask(draft, target_norm, cfg, rng).train()
    wrapped = wrap_task(task, cfg.distributed_strategy, device, cfg.dtype, world)
    optimizer = torch.optim.AdamW([p for p in draft.parameters() if p.requires_grad],
                                  lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    step, first_epoch, next_batch = 0, 0, 0
    if cfg.resume:
        # Avoid loading six copies of the full optimizer onto NPU memory.
        state = torch.load(Path(cfg.resume) / "trainer.pt", map_location="cpu", weights_only=True) if rank == 0 else None
        full_optimizer = state.pop("optimizer") if rank == 0 else {}
        metadata = [state]
        if world > 1:
            dist.broadcast_object_list(metadata, src=0, device=device)
        state = metadata[0]
        previous = asdict(TrainConfig(**state["train_config"]))
        operational = {"output", "resume", "initialize_dflash", "device", "checkpoint_every", "log_every",
                       "max_steps", "tensorboard", "vllm_endpoint", "hidden_states_dir", "data_cache_dir",
                       "progress", "num_workers", "profile_steps", "epoch_benchmark_dir"}
        changes = [k for k, v in previous.items() if k not in operational and asdict(cfg)[k] != v]
        if changes or previous["training_mode"] != "packed":
            raise ValueError(f"Resume recipe differs: {changes}; use a fresh run for the packed objective")
        if state["world_size"] != world or state["data_fingerprint"] != data._fingerprint:
            raise ValueError("Exact resume requires the same dataset and world size")
        if state["steps_per_epoch"] != counts:
            raise ValueError("Resume packing schedule differs")
        if cfg.distributed_strategy == "ddp" and rank != 0:
            full_optimizer = torch.load(Path(cfg.resume) / "trainer.pt", map_location="cpu", weights_only=True)["optimizer"]
        restore_optimizer(wrapped, optimizer, full_optimizer)
        del full_optimizer
        step, first_epoch, next_batch = state["step"], state["next_epoch"], state["next_batch"]
        _restore_rng(torch.load(Path(cfg.resume) / f"rng_rank_{rank}.pt", weights_only=True), rng, device)
    output = Path(cfg.output)
    settings = {**asdict(cfg), "world_size": world, "dataset_fingerprint": data._fingerprint,
                "data_summary": data_summary, "steps_per_epoch": counts, "total_steps": total_steps,
                "optimizer": "AdamW", "target_final_hidden": "pre-final-norm decoder output",
                "training_boundary": "confidence/exploration capped at dataset prefix mismatch"}
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        record(output, "train", settings)
        (output / "train_config.json").write_text(json.dumps(asdict(cfg), indent=2) + "\n")
        print(json.dumps({"event": "packed_training_start", **settings}), flush=True)
    if world > 1:
        dist.barrier()
    writer = None
    if rank == 0 and cfg.tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(str(output / "logs/four_anchor_dflash"), purge_step=step if cfg.resume else None)
    records = TeacherRecords(data, cfg, target.config, runner)
    initial_step, started = step, time.monotonic()
    saved = str(Path(cfg.resume)) if cfg.resume else None

    def save(next_epoch, batch_offset):
        nonlocal saved
        path = output / f"checkpoint-{step:06d}"
        # FSDP gathers are collective: every rank must participate before only
        # rank zero writes the portable model and full optimizer checkpoint.
        model_state, optimizer_state = checkpoint_states(wrapped, draft, optimizer)
        if rank == 0:
            draft.save(path, state_dict=model_state)
            torch.save({"optimizer": optimizer_state, "step": step, "next_epoch": next_epoch,
                        "next_batch": batch_offset, "world_size": world, "data_fingerprint": data._fingerprint,
                        "train_config": asdict(cfg), "steps_per_epoch": counts}, path / "trainer.pt")
            record(path, "train", settings)
            checkpoint_hashes(path)
        if world > 1:
            dist.barrier()
        torch.save(_rng_state(rng, device), path / f"rng_rank_{rank}.pt")
        if world > 1:
            dist.barrier()
        if rank == 0:
            (output / "latest_checkpoint.txt").write_text(str(path.resolve()) + "\n")
            if cfg.epoch_benchmark_dir and batch_offset == 0 and next_epoch > 0:
                from .epoch_benchmarks import publish_epoch
                publish_epoch(cfg.epoch_benchmark_dir, path, next_epoch, step)
        saved = str(path)

    try:
        for epoch in range(first_epoch, cfg.epochs):
            if step >= stop_step:
                break
            batches = epoch_batches(cfg, lengths, data._fingerprint, world, rank, epoch)
            offset = next_batch if epoch == first_epoch else 0
            if (len(batches) + cfg.gradient_accumulation - 1) // cfg.gradient_accumulation != counts[epoch]:
                raise ValueError("Distributed packing plans disagree on epoch length")
            remaining_updates = min(counts[epoch] - offset // cfg.gradient_accumulation, stop_step - step)
            selected_batches = batches[offset:offset + remaining_updates * cfg.gradient_accumulation]
            options = {}
            if cfg.num_workers:
                options.update(multiprocessing_context="spawn", prefetch_factor=2)
            loader = DataLoader(records, batch_sampler=selected_batches, collate_fn=collate_documents,
                                num_workers=cfg.num_workers, generator=torch.Generator().manual_seed(cfg.seed + epoch),
                                **options)
            iterator = iter(loader)
            progress = tqdm(total=counts[epoch], initial=offset // cfg.gradient_accumulation,
                            desc=f"Epoch {epoch + 1}/{cfg.epochs}", unit="step", dynamic_ncols=True,
                            mininterval=1, disable=rank != 0 or not cfg.progress)
            try:
                while offset < len(batches) and step < stop_step:
                    profile = step - initial_step < cfg.profile_steps
                    phase_times = [0., 0., 0., 0.]
                    redraft_blocks = 0
                    accumulation = min(cfg.gradient_accumulation, len(batches) - offset)
                    optimizer.zero_grad(set_to_none=True)
                    if step < cfg.warmup_steps:
                        multiplier = (step + 1) / max(1, cfg.warmup_steps)
                    else:
                        multiplier = 0.5 * (1 + math.cos(math.pi * (step - cfg.warmup_steps) /
                                                          max(1, total_steps - cfg.warmup_steps)))
                    for group in optimizer.param_groups:
                        group["lr"] = cfg.learning_rate * multiplier
                    metrics = torch.zeros(13, device=device)
                    for micro in range(accumulation):
                        if profile:
                            synchronize(device)
                        tick = time.perf_counter()
                        batch = {k: v.to(device=device) for k, v in next(iterator).items()}
                        if profile:
                            synchronize(device)
                        phase_times[0] += time.perf_counter() - tick
                        if batch["ids"].numel() > cfg.max_sequence_length:
                            raise ValueError("Packed batch exceeds the per-rank token budget")
                        sync = wrapped.no_sync() if world > 1 and micro + 1 < accumulation else nullcontext()
                        with sync:
                            tick = time.perf_counter()
                            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=cfg.dtype == "bfloat16"):
                                numerator, denominator, values = wrapped(batch)
                            if profile:
                                synchronize(device)
                            phase_times[1] += time.perf_counter() - tick
                            if not torch.isfinite(numerator) or denominator <= 0:
                                raise FloatingPointError("Nonfinite or unsupervised packed loss")
                            tick = time.perf_counter()
                            numerator.backward()
                            if profile:
                                synchronize(device)
                            phase_times[2] += time.perf_counter() - tick
                        metrics += values
                        redraft_blocks += task.last_redraft_blocks
                    metrics = torch.cat([metrics, metrics.new_tensor([redraft_blocks])])
                    tick = time.perf_counter()
                    if world > 1:
                        dist.all_reduce(metrics)
                    # DDP/FSDP averaged gradient sums by world size. Normalize by
                    # the actual global supervised weight across this whole update.
                    scale = world / metrics[1]
                    for parameter in draft.parameters():
                        if parameter.grad is not None:
                            parameter.grad.mul_(scale)
                    grad_norm = clip_grad_norm(wrapped, draft, 1.0)
                    optimizer.step()
                    if profile:
                        synchronize(device)
                    phase_times[3] = time.perf_counter() - tick
                    step += 1
                    offset += accumulation
                    elapsed = time.monotonic() - started
                    row = {"step": step, "total_steps": total_steps, "epoch": epoch + 1,
                           "step_in_epoch": math.ceil(offset / cfg.gradient_accumulation),
                           "steps_in_epoch": counts[epoch], **metric_row(metrics),
                           "redraft_blocks_computed": metrics[13].item(),
                           "learning_rate": optimizer.param_groups[0]["lr"], "grad_norm": float(grad_norm),
                           "estimated_remaining_seconds": elapsed / (step - initial_step) * (total_steps - step)}
                    if (cfg.packed_execution == "cached" or cfg.anchor_layout == "bonus_in_four") and not (cfg.loss_fn or {}).get("kl_div", 0):
                        row.pop("first_kl")
                        row.pop("second_kl")
                    if profile:
                        times = torch.tensor(phase_times, device=device)
                        if world > 1:
                            dist.all_reduce(times, op=dist.ReduceOp.MAX)
                        row.update(dict(zip(("max_rank_input_seconds", "max_rank_forward_seconds",
                                             "max_rank_backward_seconds", "max_rank_optimizer_seconds"), times.tolist())))
                    if rank == 0 and (step % cfg.log_every == 0 or profile):
                        row.update(memory_metrics(device))
                        progress.write(json.dumps(row))
                        with (output / "metrics.jsonl").open("a") as stream:
                            stream.write(json.dumps(row) + "\n")
                        if writer:
                            for key, value in row.items():
                                writer.add_scalar("train/" + key, value, step)
                    epoch_end = offset == len(batches)
                    periodic = epoch_end if cfg.checkpoint_each_epoch else step % cfg.checkpoint_every == 0
                    if periodic or step == stop_step:
                        save(epoch + int(epoch_end), 0 if epoch_end else offset)
                    progress.set_postfix(loss=f"{row['loss']:.4f}", global_step=f"{step}/{total_steps}",
                                         total_eta_h=f"{row['estimated_remaining_seconds'] / 3600:.2f}", refresh=False)
                    progress.update(1)
            finally:
                progress.close()
                # Release prefetch workers before opening the next epoch loader.
                if cfg.num_workers:
                    iterator._shutdown_workers()
    finally:
        records.close()
        if runner:
            runner.close()
        if writer:
            writer.close()
        if world > 1:
            dist.destroy_process_group()
    return saved
