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
    runner = TargetRunner(target, cfg.target_layer_ids)
    task = TrainingTask(draft, runner, cfg, rng)
    wrapped = DistributedDataParallel(task, device_ids=[device.index] if device.type != "cpu" else None) if world > 1 else task
    optimizer = torch.optim.AdamW([p for p in draft.parameters() if p.requires_grad],
                                  lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    data = load_records(cfg.data, cfg.split)
    if not len(data):
        raise ValueError("Empty dataset")
    data_fingerprint = getattr(data, "_fingerprint", None)
    start_step = 0
    if cfg.resume:
        state = torch.load(Path(cfg.resume) / "trainer.pt", map_location=device, weights_only=True)
        if state["world_size"] != world or state["data_fingerprint"] != data_fingerprint:
            raise ValueError("Exact resume requires the same world size and dataset fingerprint")
        operational = {"output", "resume", "initialize_dflash", "device", "checkpoint_every", "log_every"}
        current = asdict(cfg)
        changes = [k for k, value in state["train_config"].items() if k not in operational and current[k] != value]
        if changes:
            raise ValueError(f"Resume training configuration differs: {changes}")
        optimizer.load_state_dict(state["optimizer"])
        start_step = state["step"]
        rng_state = torch.load(Path(cfg.resume) / f"rng_rank_{rank}.pt", map_location="cpu", weights_only=True)
        _restore_rng(rng_state, rng, device)
    output = Path(cfg.output)
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        settings = {**asdict(cfg), "world_size": world, "data_fingerprint": data_fingerprint,
                    "target_commit": getattr(target.config, "_commit_hash", None)}
        record(output, "train", settings)
        (output / "train_config.json").write_text(json.dumps(asdict(cfg), indent=2) + "\n")
    if world > 1:
        dist.barrier()

    def save(step):
        path = output / f"checkpoint-{step:06d}"
        if rank == 0:
            draft.save(path)
            torch.save({"optimizer": optimizer.state_dict(), "step": step, "world_size": world,
                        "data_fingerprint": data_fingerprint, "train_config": asdict(cfg)}, path / "trainer.pt")
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
        for step in range(start_step, cfg.steps):
            optimizer.zero_grad(set_to_none=True)
            aggregates = torch.zeros(4, device=device)
            if step < cfg.warmup_steps:
                multiplier = (step + 1) / max(1, cfg.warmup_steps)
            else:
                progress = (step - cfg.warmup_steps) / max(1, cfg.steps - cfg.warmup_steps)
                multiplier = 0.5 * (1 + math.cos(math.pi * progress))
            for group in optimizer.param_groups:
                group["lr"] = cfg.learning_rate * multiplier
            for micro in range(cfg.gradient_accumulation):
                example = training_example(data, rng, cfg.num_proposals, cfg.max_sequence_length,
                                           target.config.vocab_size, device)
                sync = wrapped.no_sync() if world > 1 and micro + 1 < cfg.gradient_accumulation else nullcontext()
                with sync:
                    with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                        enabled=cfg.dtype == "bfloat16"):
                        loss, metrics = wrapped(*example)
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Nonfinite training loss")
                    (loss / cfg.gradient_accumulation).backward()
                aggregates += torch.tensor([loss.detach().item(), metrics["first_ce"], metrics["second_ce"],
                                            metrics["boundary"]], device=device) / cfg.gradient_accumulation
            grad_norm = torch.nn.utils.clip_grad_norm_(draft.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            if world > 1:
                dist.all_reduce(aggregates)
                aggregates /= world
            if rank == 0 and (step + 1) % cfg.log_every == 0:
                row = {"step": step + 1, "loss": aggregates[0].item(), "first_ce": aggregates[1].item(),
                       "second_ce": aggregates[2].item(), "boundary_mean": aggregates[3].item(),
                       "learning_rate": optimizer.param_groups[0]["lr"], "grad_norm": float(grad_norm)}
                print(json.dumps(row), flush=True)
                with (output / "metrics.jsonl").open("a") as stream:
                    stream.write(json.dumps(row) + "\n")
            if (step + 1) % cfg.checkpoint_every == 0 or step + 1 == cfg.steps:
                save(step + 1)
    finally:
        runner.close()
        if world > 1:
            dist.destroy_process_group()
    return str(output / f"checkpoint-{cfg.steps:06d}")
