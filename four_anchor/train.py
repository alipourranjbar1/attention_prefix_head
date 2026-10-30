"""Online two-view training with shared draft weights and branch-correct teachers.

This is a new rollout objective for the four-anchor extension, not an exact
reproduction of D-Loop's complementary ground-truth-prefix objective.
"""
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
import json
import math
import os
import random

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist

from .data import load_records, training_example
from .model import DraftConfig, FourAnchorDFlash, make_anchors, select_boundary
from .runtime import TargetRunner, load_target, resolve_device
from .provenance import record, checkpoint_hashes


@dataclass
class TrainConfig:
    target: str
    data: str
    output: str
    mask_token_id: int
    target_layer_ids: list[int]
    target_revision: str | None = None
    device: str = "cuda"
    dtype: str = "bfloat16"
    attention: str = "sdpa"
    num_draft_layers: int = 5
    num_anchors: int = 4
    num_proposals: int = 16
    max_sequence_length: int = 2048
    steps: int = 1000
    epochs: int | None = None
    max_steps: int | None = None
    gradient_accumulation: int = 8
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_steps: int = 50
    loss_decay: float = 7.0
    redraft_weight: float = 1.0
    boundary_exploration: float = 0.5
    boundary_beta: float = 0.7
    seed: int = 42
    checkpoint_every: int = 100
    log_every: int = 1
    gradient_checkpointing: bool = True
    initialize_dflash: str | None = None
    resume: str | None = None
    split: str = "train"
    checkpoint_each_epoch: bool = False
    tensorboard: bool = False
    vllm_endpoint: str | None = None
    hidden_states_dir: str | None = None
    teacher_model_name: str = "four-anchor-teacher"
    request_timeout: int = 180
    max_retries: int = 8

    def __post_init__(self):
        if self.steps < 1 or self.gradient_accumulation < 1 or self.max_sequence_length < 2:
            raise ValueError("steps/accumulation must be positive; max_sequence_length >= 2")
        if self.loss_decay <= 0 or self.redraft_weight <= 0:
            raise ValueError("Loss decay and redraft weight must be positive")
        if not 0 <= self.boundary_exploration <= 1 or not 0 < self.boundary_beta < 1:
            raise ValueError("Invalid boundary exploration or beta")
        if self.dtype not in {"float32", "bfloat16"}:
            raise ValueError("Use float32 or bfloat16; fp16 loss scaling is not implemented")
        if self.resume and self.initialize_dflash:
            raise ValueError("Choose resume OR regular DFlash initialization")
        if self.checkpoint_every < 1 or self.log_every < 1 or self.learning_rate <= 0:
            raise ValueError("Invalid logging/checkpoint interval or learning rate")
        if self.epochs is not None and self.epochs < 1:
            raise ValueError("epochs must be positive")
        if self.max_steps is not None and self.max_steps < 1:
            raise ValueError("max_steps must be positive")
        if self.vllm_endpoint and not self.hidden_states_dir:
            raise ValueError("Remote teacher requires hidden_states_dir")


class TrainingTask(nn.Module):
    def __init__(self, drafter, runner, cfg, rng):
        super().__init__()
        self.drafter, self.runner, self.cfg, self.rng = drafter, runner, cfg, rng

    def forward(self, prefix, labels, valid):
        cfg, draft = self.cfg, self.drafter
        state = self.runner.prefill(prefix)
        context_features = state.features
        n, m = prefix.shape[1], cfg.num_proposals
        anchors = make_anchors(prefix, context_features, cfg.num_anchors, draft.hidden_size)
        first = draft(context_features, anchors, n)
        keep = select_boundary(first.logits)
        if self.rng.random() < cfg.boundary_exploration:
            keep = self.rng.choices(range(m), weights=[cfg.boundary_beta ** i for i in range(m)])[0]
        anchors2 = make_anchors(prefix, context_features, cfg.num_anchors, draft.hidden_size, first, keep)
        # A wrong retained prefix changes the correct continuation. Re-query the
        # frozen teacher on that actual branch instead of using oracle gold features.
        branch = self.runner.append(state, first.tokens[:, :keep].detach())
        eos = self.runner.target.generation_config.eos_token_id
        eos = [] if eos is None else eos if isinstance(eos, list) else [eos]
        second_labels, second_valid = self.runner.rollout(branch, m, eos)
        # Only original verified target features reach pass two. No branch teacher
        # states escape rollout into this input (including during training).
        second = draft(context_features, anchors2, n + keep)
        w = torch.exp(-torch.arange(m, device=prefix.device).float() / cfg.loss_decay)[None]
        w1, w2 = valid * w, second_valid * w * cfg.redraft_weight
        ce1 = F.cross_entropy(first.logits.float().transpose(1, 2), labels, reduction="none")
        ce2 = F.cross_entropy(second.logits.float().transpose(1, 2), second_labels, reduction="none")
        loss = ((ce1 * w1).sum() + (ce2 * w2).sum()) / (w1.sum() + w2.sum()).clamp_min(1e-8)
        return loss, {"first_ce": float((ce1.detach() * w1).sum() / w1.sum().clamp_min(1e-8)),
                      "second_ce": float((ce2.detach() * w2).sum() / w2.sum().clamp_min(1e-8)),
                      "boundary": keep}


def _rng_state(rng, device):
    state = {"python": rng.getstate(), "torch": torch.get_rng_state()}
    if device.type in {"cuda", "npu"}:
        state[device.type] = getattr(torch, device.type).get_rng_state(device).cpu()
    return state


def _restore_rng(state, rng, device):
    rng.setstate(state["python"])
    torch.set_rng_state(state["torch"].cpu())
    if device.type in state:
        getattr(torch, device.type).set_rng_state(state[device.type].cpu(), device)


def train(cfg: TrainConfig):
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device_name = cfg.device
    if world > 1 and cfg.device.split(":")[0] in {"cuda", "npu"}:
        device_name = f"{cfg.device.split(':')[0]}:{local_rank}"
    device = resolve_device(device_name)
    if world > 1:
        dist.init_process_group(backend={"cuda": "nccl", "npu": "hccl"}.get(device.type, "gloo"))
    torch.manual_seed(cfg.seed)
    rng = random.Random(cfg.seed + rank)
    if cfg.vllm_endpoint:
        from .remote_teacher import FrozenTargetInterface, RemoteTeacher
        target = FrozenTargetInterface(cfg.target, device, getattr(torch, cfg.dtype))
    else:
        target, _ = load_target(cfg.target, str(device), cfg.dtype, cfg.target_revision, cfg.attention)
    if cfg.max_sequence_length + 2 * cfg.num_proposals > target.config.max_position_embeddings:
        raise ValueError("Reserve two proposal blocks beyond max_sequence_length inside the target context window")
    draft_cfg = DraftConfig(cfg.target, cfg.target_layer_ids, cfg.mask_token_id, cfg.num_draft_layers,
                            cfg.num_anchors, cfg.num_proposals, cfg.attention, cfg.target_revision)
    if cfg.resume:
        draft = FourAnchorDFlash.load(cfg.resume, target)
        if asdict(draft.cfg) != asdict(draft_cfg):
            raise ValueError("Resume model configuration differs from checkpoint")
    else:
        draft = FourAnchorDFlash(draft_cfg, target)
        if cfg.initialize_dflash:
            draft.initialize_regular_dflash(cfg.initialize_dflash)
    draft.gradient_checkpointing = cfg.gradient_checkpointing
    # Keep optimizer/master weights in float32; autocast handles the bf16 forward.
    for parameter in draft.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.float()
    draft.train()
    if cfg.vllm_endpoint:
        runner = RemoteTeacher(target, cfg.target_layer_ids, cfg.vllm_endpoint, cfg.hidden_states_dir,
                               cfg.teacher_model_name, cfg.request_timeout, cfg.max_retries)
        runner.check_server()
    else:
        runner = TargetRunner(target, cfg.target_layer_ids)
    task = TrainingTask(draft, runner, cfg, rng)
    wrapped = DistributedDataParallel(task, device_ids=[device.index] if device.type != "cpu" else None) if world > 1 else task
    optimizer = torch.optim.AdamW([p for p in draft.parameters() if p.requires_grad],
                                  lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    data = load_records(cfg.data, cfg.split)
    if not len(data):
        raise ValueError("Empty dataset")
    data_fingerprint = getattr(data, "_fingerprint", None)
    sampler = None
    steps_per_epoch = None
    total_steps = cfg.steps
    if cfg.epochs is not None:
        from torch.utils.data.distributed import DistributedSampler
        sampler = DistributedSampler(data, num_replicas=world, rank=rank, shuffle=True,
                                     seed=cfg.seed, drop_last=False)
        steps_per_epoch = math.ceil(len(sampler) / cfg.gradient_accumulation)
        total_steps = cfg.epochs * steps_per_epoch
    stop_step = min(total_steps, cfg.max_steps) if cfg.max_steps is not None else total_steps
    start_step = 0
    if cfg.resume:
        state = torch.load(Path(cfg.resume) / "trainer.pt", map_location=device, weights_only=True)
        if state["world_size"] != world or state["data_fingerprint"] != data_fingerprint:
            raise ValueError("Exact resume requires the same world size and dataset fingerprint")
        operational = {"output", "resume", "initialize_dflash", "device", "checkpoint_every", "log_every",
                       "max_steps", "tensorboard", "vllm_endpoint", "hidden_states_dir"}
        current = asdict(cfg)
        previous_config = asdict(TrainConfig(**state["train_config"]))
        changes = [k for k, value in previous_config.items() if k not in operational and current[k] != value]
        if changes:
            raise ValueError(f"Resume training configuration differs: {changes}")
        if state.get("total_steps", previous_config["steps"]) != total_steps:
            raise ValueError("Resume schedule length differs")
        optimizer.load_state_dict(state["optimizer"])
        start_step = state["step"]
        rng_state = torch.load(Path(cfg.resume) / f"rng_rank_{rank}.pt", map_location="cpu", weights_only=True)
        _restore_rng(rng_state, rng, device)
    output = Path(cfg.output)
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        settings = {**asdict(cfg), "world_size": world, "data_fingerprint": data_fingerprint,
                    "target_commit": getattr(target.config, "_commit_hash", None),
                    "dataset_rows": len(data), "total_steps": total_steps,
                    "steps_per_epoch": steps_per_epoch, "training_blocks_per_sequence": 1}
        record(output, "train", settings)
        (output / "train_config.json").write_text(json.dumps(asdict(cfg), indent=2) + "\n")
    if world > 1:
        dist.barrier()
    writer = None
    if rank == 0 and cfg.tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(str(output / "logs" / "four_anchor_dflash"), purge_step=start_step if cfg.resume else None)

    def save(step):
        path = output / f"checkpoint-{step:06d}"
        if rank == 0:
            draft.save(path)
            torch.save({"optimizer": optimizer.state_dict(), "step": step, "world_size": world,
                        "data_fingerprint": data_fingerprint, "train_config": asdict(cfg),
                        "total_steps": total_steps}, path / "trainer.pt")
            record(path, "train", settings)
            checkpoint_hashes(path)
        if world > 1:
            dist.barrier()
        torch.save(_rng_state(rng, device), path / f"rng_rank_{rank}.pt")
        if world > 1:
            dist.barrier()
        if rank == 0:
            (output / "latest_checkpoint.txt").write_text(str(path.resolve()) + "\n")

    try:
        current_epoch, epoch_indices = -1, None
        for step in range(start_step, stop_step):
            row_indices = [None] * cfg.gradient_accumulation
            if sampler is not None:
                epoch, update_in_epoch = divmod(step, steps_per_epoch)
                if epoch != current_epoch:
                    sampler.set_epoch(epoch)
                    epoch_indices = list(sampler)
                    current_epoch = epoch
                begin = update_in_epoch * cfg.gradient_accumulation
                row_indices = epoch_indices[begin:begin + cfg.gradient_accumulation]
            accumulation = len(row_indices)
            optimizer.zero_grad(set_to_none=True)
            aggregates = torch.zeros(4, device=device)
            if step < cfg.warmup_steps:
                multiplier = (step + 1) / max(1, cfg.warmup_steps)
            else:
                progress = (step - cfg.warmup_steps) / max(1, total_steps - cfg.warmup_steps)
                multiplier = 0.5 * (1 + math.cos(math.pi * progress))
            for group in optimizer.param_groups:
                group["lr"] = cfg.learning_rate * multiplier
            for micro, row_index in enumerate(row_indices):
                example = training_example(data, rng, cfg.num_proposals, cfg.max_sequence_length,
                                           target.config.vocab_size, device, row_index=row_index)
                sync = wrapped.no_sync() if world > 1 and micro + 1 < accumulation else nullcontext()
                with sync:
                    with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                        enabled=cfg.dtype == "bfloat16"):
                        loss, metrics = wrapped(*example)
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Nonfinite training loss")
                    (loss / accumulation).backward()
                aggregates += torch.tensor([loss.detach().item(), metrics["first_ce"], metrics["second_ce"],
                                            metrics["boundary"]], device=device) / accumulation
            grad_norm = torch.nn.utils.clip_grad_norm_(draft.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            if world > 1:
                dist.all_reduce(aggregates)
                aggregates /= world
            if rank == 0 and (step + 1) % cfg.log_every == 0:
                row = {"step": step + 1, "loss": aggregates[0].item(), "first_ce": aggregates[1].item(),
                       "second_ce": aggregates[2].item(), "boundary_mean": aggregates[3].item(),
                       "learning_rate": optimizer.param_groups[0]["lr"], "grad_norm": float(grad_norm)}
                if steps_per_epoch is not None:
                    row["epoch"] = (step + 1) / steps_per_epoch
                print(json.dumps(row), flush=True)
                with (output / "metrics.jsonl").open("a") as stream:
                    stream.write(json.dumps(row) + "\n")
                if writer is not None:
                    for key, value in row.items():
                        if key != "step":
                            writer.add_scalar("train/" + key, value, step + 1)
            epoch_end = steps_per_epoch is not None and (step + 1) % steps_per_epoch == 0
            periodic = epoch_end if cfg.checkpoint_each_epoch else (step + 1) % cfg.checkpoint_every == 0
            if periodic or step + 1 == stop_step:
                save(step + 1)
    finally:
        runner.close()
        if writer is not None:
            writer.close()
        if world > 1:
            dist.destroy_process_group()
    return str(output / f"checkpoint-{stop_step:06d}")
