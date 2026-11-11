"""20/19-query two-round DSpark distillation with no inter-round teacher call.

Markov loss conditioning follows upstream teacher forcing. First-pass proposal
tokens, used to choose the restart and construct anchors, instead come from a
sequential greedy Markov chain. The original mismatch cap preserves validity
of cached teacher labels for the retained prefix. This is not on-policy RL.
"""
import torch

from .bonus_addon_objective import first_anchors, redraft_anchors
from .cached_loss import teacher_cache
from .dspark_loss import projected_dspark_loss
from .packed_objective import rejection_boundary


def dspark_forward(task, batch):
    cfg, draft = task.cfg, task.drafter
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
    conf1, conf2 = torch.zeros(2, device=index.device), torch.zeros(2, device=index.device)
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

    def loss(h, ti, w):
        # Token t-1 conditions the Markov bias for token t, and teacher hidden
        # state t-1 supplies its label. These IDs never enter the backbone.
        return projected_dspark_loss(draft, h, batch["ids"][ti], targets, ti, w,
                                    mix, cfg.confidence_alpha, cfg.head_chunk_size)

    for begin in range(0, len(chosen), cfg.blocks_per_chunk):
        s = slice(begin, begin + cfg.blocks_per_chunk)
        starts, docs = starts_all[s], docs_all[s]
        a1, bonus_ids = first_anchors(batch, starts, docs, draft.hidden_size)
        first = hidden(a1, starts, starts - 1, docs, bonus_ids=bonus_ids, return_all=True)
        ti, w1, gold, valid = supervision(starts, docs)
        tokens, logq, log_head = draft.greedy_proposals(
            first[:, 5:], bonus_ids[:, 0],
            need_token_probabilities=cfg.boundary_confidence == "draft_probability")
        logc = logq if cfg.boundary_confidence == "draft_probability" else log_head
        predicted = rejection_boundary(logc)
        exploration = []
        for _ in range(len(starts)):
            value = -1
            if task.rng.random() < cfg.boundary_exploration:
                value = task.rng.choices(range(15), weights=[cfg.boundary_beta ** j for j in range(15)])[0]
            exploration.append(value)
        explore = torch.tensor(exploration, device=index.device)
        keep = torch.where(explore >= 0, explore, predicted)
        keep = torch.minimum(keep, ((tokens == gold) & valid).long().cumprod(-1).sum(-1))
        keep = torch.minimum(keep, batch["lengths"][docs] - starts - 1)
        a2 = redraft_anchors(a1, bonus_ids, starts, keep, first, tokens)
        # Even keep=0 uses four detached first-pass states; always redraft.
        second = hidden(a2, starts + keep, starts - 1, docs)
        ti2, w2, _, _ = supervision(starts + keep, docs, (15 - keep)[:, None])
        w2 = w2 * cfg.redraft_weight
        l1, st1, c1 = loss(first[:, 5:], ti, w1)
        l2, st2, c2 = loss(second, ti2, w2)
        value = l1 + l2
        numerator = value if numerator is None else numerator + value
        stats1 += st1
        stats2 += st2
        conf1 += c1
        conf2 += c2
        weight1 += w1.sum()
        weight2 += w2.sum()
        kept += keep.sum()
        predicted_total += predicted.sum()
        task.last_redraft_blocks += len(starts)
    denominator = weight1 + weight2
    metrics = torch.cat([numerator.detach()[None], denominator[None], stats1, weight1[None],
        stats2, weight2[None], kept[None], predicted_total[None], denominator.new_tensor([len(chosen)]),
        conf1, conf2])
    return numerator, denominator, metrics
