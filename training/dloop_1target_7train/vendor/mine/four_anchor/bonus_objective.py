"""Train the 19-query bonus-in-four layout used by BonusInFourDraft.

At proposal start s, positions s-4..s-2 have target features; s-1 is
the known bonus, with embedding only. Teacher features at s-1 and beyond
are supervision ONLY. Both passes have four anchor queries + 15 masks.
"""
import torch

from .model import Anchors
from .cached_loss import teacher_cache, projected_loss
from .packed_objective import rejection_boundary


def bonus_anchors(batch, original, starts, docs, hidden_size, first_hidden=None, first_tokens=None):
    positions = starts[:, None] + torch.arange(-4, 0, device=starts.device)
    processed_limit = original - 1
    valid = positions >= 0
    known = valid & (positions < processed_limit[:, None])
    indices = batch["offsets"][docs, None] + torch.minimum(
        positions.clamp_min(0), batch["lengths"][docs, None] - 1)
    ids = batch["ids"][indices]
    tf = batch["features"][indices] * known[..., None]
    df = tf.new_zeros((*positions.shape, hidden_size))
    source = torch.where(valid, torch.where(known, 1, 3 if first_hidden is None else 2), 0)
    if first_hidden is not None:
        q = (positions - processed_limit[:, None]).clamp(0, first_hidden.shape[1] - 1)
        bonus_ids = batch["ids"][batch["offsets"][docs] + processed_limit][:, None]
        prior_ids = torch.cat((bonus_ids, first_tokens), -1)
        ids = torch.where(source == 2, prior_ids.gather(1, q), ids)
        df = first_hidden.detach().gather(1, q[..., None].expand(-1, -1, hidden_size))
        df = df * (source == 2)[..., None]
    elif not torch.equal(starts, original):
        raise ValueError("Redraft anchors require first-pass features")
    return Anchors(torch.where(valid, ids, 0), positions.clamp_min(0), tf, df, source)


def bonus_forward(task, batch):
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

    def hidden(anchors, starts, limits, docs, include_bonus=False):
        return draft.packed_hidden(batch["features"][None], batch["positions"], batch["documents"],
                                  anchors, starts, limits, docs, memory=memory, include_bonus=include_bonus)

    for begin in range(0, len(chosen), cfg.blocks_per_chunk):
        s = slice(begin, begin + cfg.blocks_per_chunk)
        starts, docs = starts_all[s], docs_all[s]
        a1 = bonus_anchors(batch, starts, starts, docs, draft.hidden_size)
        first = hidden(a1, starts, starts - 1, docs, include_bonus=True)
        ti, w1, gold, valid = supervision(starts, docs)
        l1, st1, tokens, logc = projected_loss(first[:, 1:], draft.lm_head, targets, ti, w1,
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
        a2 = bonus_anchors(batch, starts, restart, docs, draft.hidden_size, first, tokens)
        # Even keep=0 changes bonus conditioning from embedding to draft state.
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
