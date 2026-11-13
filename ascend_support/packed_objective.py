"""Batch-wide anchor allocation for the unchanged native DLoop forward.

Only sampling/loop-depth RNG plumbing changes. Documents remain in independent
batch rows; original attention, main + final-stage CE, and carrier gradients
are used. Quotas are shared fairly among documents, capped by each document's
eligible positions and ceil(budget / batch_size). Sampling within each
document is uniform without replacement as in upstream.
"""
from types import MethodType

import numpy as np
import torch


def allocate_anchors(capacities, budget, seed):
    capacities = np.asarray(capacities, dtype=np.int64)
    if np.any(capacities < 0) or budget < 1:
        raise ValueError("Invalid anchor capacity/budget")
    if not len(capacities):
        raise ValueError("Empty packed batch")
    # Bound padded work too: a single response-rich record must not create
    # hundreds of dummy blocks for every other document in the batch.
    capacities = np.minimum(capacities, (budget + len(capacities) - 1)//len(capacities))
    quotas = np.zeros_like(capacities)
    remaining = min(int(budget), int(capacities.sum()))
    rng = np.random.default_rng(seed)
    while remaining:
        eligible = np.flatnonzero(quotas < capacities)
        if remaining < len(eligible):
            quotas[rng.permutation(eligible)[:remaining]] += 1
            break
        share = remaining // len(eligible)
        added = np.minimum(capacities[eligible] - quotas[eligible], share)
        quotas[eligible] += added
        remaining -= int(added.sum())
    return quotas


def _sample_packed_anchors(self, seq_len, loss_mask, device):
    lengths = self._packed_lengths.to(device=device)
    positions = torch.arange(seq_len, device=device).unsqueeze(0)
    # Use each document's true end, not the right-padded batch width.
    valid = (loss_mask > 0.5) & (positions <= (lengths - 2*self.block_size).unsqueeze(1))
    counts = valid.sum(dim=1).detach().cpu().numpy()
    # Upstream leaves at least one eligible start unsampled for a singleton.
    capacities = np.maximum(counts - 1, 0)
    quotas = allocate_anchors(capacities, self.num_anchors,
                              self._packed_seed + 1000003*self._packed_step)
    n = int(quotas.max())
    if n <= 0:
        raise ValueError("Packed batch has no eligible DLoop anchors")
    random_values = torch.rand(loss_mask.shape, device=device)
    random_values = torch.where(valid, random_values, torch.full_like(random_values, 2.0))
    chosen = random_values.argsort(dim=1)[:, :n]
    keep = torch.arange(n, device=device).unsqueeze(0) < torch.as_tensor(quotas, device=device).unsqueeze(1)
    # Sort only real selections; otherwise padded dummy entries could steal
    # valid labels when quotas differ across documents.
    anchors = torch.where(keep, chosen, torch.full_like(chosen, seq_len)).sort(dim=1).values
    anchors = torch.where(keep, anchors, torch.zeros_like(anchors))
    self._packed_sampled_anchors = int(quotas.sum())
    self._packed_allocated_anchor_slots = int(len(quotas)*n)
    return anchors, keep


def _packed_loop_depth(self):
    # Every FSDP rank must execute the same number of stages even when local
    # batch shapes/RNG consumption differ. This is an independent uniform RNG
    # stream keyed by optimizer step; no extra target call or communication.
    generator = torch.Generator().manual_seed(self._packed_seed + 1000033*self._packed_step)
    return int(torch.randint(1, self.loop_max_stage + 1, (), generator=generator))


def enable_packed_sampling(model, seed=42):
    from specforge.core.dflash import OnlineDFlashModel
    if type(model) is not OnlineDFlashModel or model.loop_max_stage < 1 or model.loss_type != "dflash":
        raise ValueError("Packing supports native regular DFlash DLoop only")
    model._packed_seed = int(seed)
    model._sample_anchor_positions = MethodType(_sample_packed_anchors, model)
    model._sample_loop_depth = MethodType(_packed_loop_depth, model)
    return model


def set_packed_step(model, lengths, global_step):
    model._packed_lengths = lengths
    model._packed_step = int(global_step)
