"""Same two-view objective, shared context work and per-block redraft reuse."""
from dataclasses import fields
import torch

from .cached_loss import teacher_cache, projected_loss
from .model import Anchors


def select_anchors(anchors, indices):
    return Anchors(**{f.name: getattr(anchors, f.name)[indices] for f in fields(Anchors)})


def cached_forward(task, batch):
    from .packed_objective import packed_anchors, rejection_boundary
    cfg, draft = task.cfg, task.drafter
    eligible = ((batch["weights"] > 0) & (batch["positions"] > 0)).nonzero().flatten().tolist()
    if not eligible:
        raise ValueError("Packed batch has no trainable starts")
    selected = task.rng.sample(eligible, min(cfg.max_anchors, len(eligible)))
    device = batch["ids"].device
    indices = torch.tensor(selected, device=device)
    all_starts, all_docs = batch["positions"][indices], batch["documents"][indices]
    memory = draft.packed_memory(batch["features"][None], batch["positions"])
    targets = teacher_cache(batch["final"], task.target_norm, draft.lm_head, cfg.head_chunk_size)
    all_anchors = packed_anchors(batch, all_starts, all_starts, all_docs, draft.hidden_size, cfg.num_anchors)
    mix = cfg.loss_fn or {"ce": .1, "tv": .9}
    distance = torch.arange(cfg.num_proposals, device=device)
    decay = torch.exp(-distance.float() / cfg.loss_decay)
    numerator = None
    stats1, stats2 = torch.zeros(3, device=device), torch.zeros(3, device=device)
    weight1, weight2 = torch.zeros((), device=device), torch.zeros((), device=device)
    kept_sum, predicted_sum = torch.zeros((), device=device), torch.zeros((), device=device)
    task.last_redraft_blocks = 0

    def supervision(starts, docs):
        positions = starts[:, None] + distance
        valid = positions < batch["lengths"][docs, None]
        idx = batch["offsets"][docs, None] + torch.minimum(positions, batch["lengths"][docs, None] - 1)
        return idx - 1, batch["weights"][idx] * valid * decay, batch["ids"][idx], valid

    def hidden(anchors, starts, limits, docs):
        return draft.packed_hidden(batch["features"][None], batch["positions"], batch["documents"],
                                    anchors, starts, limits, docs, memory=memory)

    for begin in range(0, len(selected), cfg.blocks_per_chunk):
        selection = slice(begin, begin + cfg.blocks_per_chunk)
        starts, docs = all_starts[selection], all_docs[selection]
        first = hidden(select_anchors(all_anchors, selection), starts, starts, docs)
        target_indices, w1, gold, valid = supervision(starts, docs)
        loss1, st1, tokens, logc = projected_loss(first, draft.lm_head, targets, target_indices, w1,
                                                mix, cfg.head_chunk_size)
        predicted = rejection_boundary(logc)
        # One host-to-device transfer per group instead of hundreds of scalar
        # assignment kernels. Preserve the reference RNG's draw order exactly.
        exploration = []
        for _ in range(len(starts)):
            value = -1
            if task.rng.random() < cfg.boundary_exploration:
                value = task.rng.choices(range(cfg.num_proposals),
                    weights=[cfg.boundary_beta ** j for j in range(cfg.num_proposals)])[0]
            exploration.append(value)
        explore = torch.tensor(exploration, device=device)
        keep = torch.where(explore >= 0, explore, predicted)
        keep = torch.minimum(keep, ((tokens == gold) & valid).long().cumprod(-1).sum(-1))
        keep = torch.minimum(keep, batch["lengths"][docs] - starts - 1)
        active = (keep > 0).nonzero().flatten()
        inactive = keep == 0
        # Identical zero-boundary views retain BOTH losses and their gradients.
        value = loss1.sum() + (loss1 * inactive).sum() * cfg.redraft_weight
        stats1 += st1.sum(0)
        stats2 += (st1 * inactive[:, None]).sum(0) * cfg.redraft_weight
        weight1 += w1.sum()
        weight2 += (w1 * inactive[:, None]).sum() * cfg.redraft_weight
        if active.numel():
            original = starts[active]
            second_starts, second_docs = original + keep[active], docs[active]
            anchors2 = packed_anchors(batch, original, second_starts, second_docs, draft.hidden_size,
                                      cfg.num_anchors, first[active], tokens[active])
            second = hidden(anchors2, second_starts, original, second_docs)
            target2, w2, _, _ = supervision(second_starts, second_docs)
            w2 = w2 * cfg.redraft_weight
            loss2, st2, _, _ = projected_loss(second, draft.lm_head, targets, target2, w2, mix, cfg.head_chunk_size)
            value = value + loss2.sum()
            stats2 += st2.sum(0)
            weight2 += w2.sum()
            task.last_redraft_blocks += active.numel()
        numerator = value if numerator is None else numerator + value
        kept_sum += keep.sum()
        predicted_sum += predicted.sum()
    denominator = weight1 + weight2
    metrics = torch.cat([numerator.detach()[None], denominator[None], stats1, weight1[None], stats2,
                         weight2[None], kept_sum[None], predicted_sum[None], denominator.new_tensor([len(selected)])])
    return numerator, denominator, metrics
