"""Train the 20/19 query layout used by BatchedBonusAddonDraft.

For first speculative position s, processed context ends at s-2. Four
processed tokens have target features; token s-1 is the known bonus with
embedding ONLY. Supervise s..s+14. Round two uses four first-pass DRAFT
states, including when the boundary falls within the original anchor span.
The dataset supplies the known bonus token under teacher forcing. Its target
features, and future target features, are allowed only for loss supervision.
"""
import torch

from .model import Anchors
from .cached_loss import teacher_cache, projected_loss
from .packed_objective import packed_anchors, rejection_boundary


def first_anchors(batch, starts, docs, hidden_size):
    processed = starts - 1
    anchors = packed_anchors(batch, processed, processed, docs, hidden_size, 4)
    bonus_ids = batch["ids"][batch["offsets"][docs] + processed][:, None]
    return anchors, bonus_ids


def redraft_anchors(anchors, bonus_ids, starts, keep, first_hidden, first_tokens):
    """Pick the four positions immediately before restart, all from pass one.

    first_hidden has 20 normalized states: four anchors, bonus, 15 proposals.
    Detach the reused states, matching the existing two-pass training recipe.
    """
    if first_hidden.shape[:2] != (len(starts), 20) or first_tokens.shape != (len(starts), 15):
        raise ValueError("Expected all 20 first-pass states and 15 proposed token IDs")
    if keep.shape != starts.shape or ((keep < 0) | (keep >= 15)).any():
        raise ValueError("Boundary must be a per-block candidate index in 0..14")
    positions = starts[:, None] + keep[:, None] + torch.arange(-4, 0, device=starts.device)
    indices = keep[:, None] + torch.arange(1, 5, device=starts.device)
    valid = positions >= 0
    ids = torch.cat((anchors.ids, bonus_ids, first_tokens), -1).gather(1, indices)
    df = first_hidden.detach().gather(1, indices[..., None].expand(-1, -1, first_hidden.shape[-1]))
    return Anchors(torch.where(valid, ids, 0), positions.clamp_min(0),
                   torch.zeros_like(anchors.target_features), df * valid[..., None],
                   torch.where(valid, 2, 0))


def bonus_addon_forward(task, batch):
    cfg, draft = task.cfg, task.drafter
    # Need at least one processed token before the pending bonus.
    eligible = ((batch["weights"] > 0) & (batch["positions"] >= 2)).nonzero().flatten().tolist()
    if not eligible:
        raise ValueError("No supervised position with a processed prefix and a bonus")
    chosen = task.rng.sample(eligible, min(cfg.max_anchors, len(eligible)))
    index = torch.tensor(chosen, device=batch["ids"].device)
    starts_all, docs_all = batch["positions"][index], batch["documents"][index]
    memory = (draft.packed_memory(batch["features"][None], batch["positions"])
              if cfg.packed_execution == "cached" else None)
    targets = teacher_cache(batch["final"], task.target_norm, draft.lm_head, cfg.head_chunk_size)
    distance = torch.arange(15, device=index.device)
    decay = torch.exp(-distance.float() / cfg.loss_decay)
    mix = cfg.loss_fn or {"ce": .1, "tv": .9}
    numerator = None
    stats1, stats2 = torch.zeros(3, device=index.device), torch.zeros(3, device=index.device)
    weight1, weight2 = torch.zeros((), device=index.device), torch.zeros((), device=index.device)
    kept, predicted_total = torch.zeros((), device=index.device), torch.zeros((), device=index.device)
    task.last_redraft_blocks = 0

    def supervision(starts, docs, remaining=15):
        positions = starts[:, None] + distance
        valid = positions < batch["lengths"][docs, None]
        i = batch["offsets"][docs, None] + torch.minimum(positions, batch["lengths"][docs, None] - 1)
        weights = batch["weights"][i] * valid * decay * (distance < remaining)
        return i - 1, weights, batch["ids"][i], valid

    def hidden(anchors, starts, limits, docs, bonus_ids=None, return_all=False):
        return draft.packed_hidden(batch["features"][None], batch["positions"], batch["documents"],
                                  anchors, starts, limits, docs, memory=memory, bonus_ids=bonus_ids, return_all=return_all)

    for begin in range(0, len(chosen), cfg.blocks_per_chunk):
        s = slice(begin, begin + cfg.blocks_per_chunk)
        starts, docs = starts_all[s], docs_all[s]
        a1, bonus_ids = first_anchors(batch, starts, docs, draft.hidden_size)
        first = hidden(a1, starts, starts - 1, docs, bonus_ids=bonus_ids, return_all=True)
        ti, w1, gold, valid = supervision(starts, docs)
        l1, st1, tokens, logc = projected_loss(first[:, 5:], draft.lm_head, targets, ti, w1,
                                             mix, cfg.head_chunk_size)
        predicted = rejection_boundary(logc)
        exploration = []
        for _ in range(len(starts)):
            value = -1
            if task.rng.random() < cfg.boundary_exploration:
                value = task.rng.choices(range(15), weights=[cfg.boundary_beta ** j for j in range(15)])[0]
            exploration.append(value)
        explore = torch.tensor(exploration, device=index.device)
        keep = torch.where(explore >= 0, explore, predicted)
        # Teacher forcing is valid only along the matching retained prefix.
        keep = torch.minimum(keep, ((tokens == gold) & valid).long().cumprod(-1).sum(-1))
        keep = torch.minimum(keep, batch["lengths"][docs] - starts - 1)
        restart = starts + keep
        a2 = redraft_anchors(a1, bonus_ids, starts, keep, first, tokens)
        # Even keep=0 changes all four anchors to drafter states: always run pass two.
        second = hidden(a2, restart, starts - 1, docs)
        ti2, w2, _, _ = supervision(restart, docs, (15 - keep)[:, None])
        w2 = w2 * cfg.redraft_weight
        l2, st2, _, _ = projected_loss(second, draft.lm_head, targets, ti2, w2, mix, cfg.head_chunk_size)
        value = l1.sum() + l2.sum()
        numerator = value if numerator is None else numerator + value
        stats1 += st1.sum(0)
        stats2 += st2.sum(0)
        weight1 += w1.sum()
        weight2 += w2.sum()
        kept += keep.sum()
        predicted_total += predicted.sum()
        task.last_redraft_blocks += len(starts)
    denominator = weight1 + weight2
    metrics = torch.cat([numerator.detach()[None], denominator[None], stats1, weight1[None],
                         stats2, weight2[None], kept[None], predicted_total[None],
                         denominator.new_tensor([len(chosen)])])
    return numerator, denominator, metrics
