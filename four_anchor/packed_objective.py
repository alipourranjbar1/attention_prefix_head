"""Two-pass packed distillation: DSpark's CE/TV loss, regular DFlash backbone.

Teacher targets are teacher-forced next-token distributions. Training caps the
predicted boundary at the first dataset-token mismatch so the reused teacher
states describe the actual retained prefix. Inference still uses confidence
alone and verifies only after both draft passes.
"""
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .model import Anchors


def distribution_terms(logits, teacher_logits):
    """Exact eager definitions used by Speculators (no top-k approximation)."""
    logq = F.log_softmax(logits.float(), -1)
    p = F.softmax(teacher_logits.detach().float(), -1)
    ce = F.nll_loss(logq, teacher_logits.argmax(-1), reduction="none")
    tv = (1 - torch.minimum(logq.exp(), p).sum(-1)).clamp_min(0)
    kl = F.kl_div(logq, p, reduction="none").sum(-1)
    return torch.stack([ce, tv, kl], -1)


def rejection_boundary(log_confidence):
    log_confidence = log_confidence.clamp(max=0)
    prior = torch.cat([torch.zeros_like(log_confidence[:, :1]), log_confidence.cumsum(-1)[:, :-1]], -1)
    return (prior + (-torch.expm1(log_confidence)).log()).argmax(-1)


def packed_anchors(batch, original_starts, starts, block_docs, hidden_size, count,
                   first_hidden=None, first_tokens=None):
    device = starts.device
    positions = starts[:, None] + torch.arange(-count, 0, device=device)
    global_indices = batch["offsets"][block_docs, None] + positions.clamp_min(0)
    source = torch.where(positions < 0, 0, torch.where(positions < original_starts[:, None], 1, 2))
    ids = batch["ids"][global_indices]
    tf = batch["features"][global_indices] * (source == 1)[..., None]
    df = torch.zeros((*ids.shape, hidden_size), device=device, dtype=tf.dtype)
    if first_hidden is not None:
        indices = (positions - original_starts[:, None]).clamp(0, first_hidden.shape[1] - 1)
        proposed = first_tokens.gather(1, indices)
        ids = torch.where(source == 2, proposed, ids)
        df = first_hidden.detach().gather(1, indices[..., None].expand(-1, -1, hidden_size))
        df = df * (source == 2)[..., None]
    elif (source == 2).any():
        raise ValueError("Proposal anchors require first-pass states")
    ids = torch.where(source == 0, 0, ids)
    return Anchors(ids, positions.clamp_min(0), tf, df, source)


class PackedTask(nn.Module):
    def __init__(self, draft, target_norm, cfg, rng):
        super().__init__()
        self.drafter, self.target_norm, self.cfg, self.rng = draft, target_norm, cfg, rng
        self.target_norm.requires_grad_(False)

    def loss_and_predictions(self, hidden, teacher_hidden, weights):
        """Checkpoint vocabulary chunks, never retain [512,16,151936] logits."""
        cfg = self.cfg
        parts, statistics, tokens, confidence = [], [], [], []
        mix = cfg.loss_fn or {"ce": 0.1, "tv": 0.9}
        coefficients = hidden.new_tensor([mix.get(k, 0) for k in ("ce", "tv", "kl_div")], dtype=torch.float32)

        def project(h, teacher, weight):
            logits = self.drafter.lm_head(h)
            with torch.no_grad():
                teacher_logits = self.drafter.lm_head(self.target_norm(teacher))
            terms = distribution_terms(logits, teacher_logits)
            sums = (terms * weight[:, None]).sum(0)
            with torch.no_grad():
                token = logits.argmax(-1)
                logc = logits.float().amax(-1) - torch.logsumexp(logits.float(), -1)
            return (sums * coefficients).sum(), sums.detach(), token, logc

        flat = hidden.reshape(-1, hidden.shape[-1])
        teacher = teacher_hidden.reshape(-1, teacher_hidden.shape[-1])
        for i in range(0, flat.shape[0], cfg.head_chunk_size):
            args = (flat[i:i + cfg.head_chunk_size], teacher[i:i + cfg.head_chunk_size],
                    weights.reshape(-1)[i:i + cfg.head_chunk_size])
            # Also when layer checkpointing is disabled: the vocabulary dominates
            # activation memory, and the frozen head still backpropagates into h.
            result = checkpoint(project, *args, use_reentrant=False) if self.training else project(*args)
            value, stats, token, logc = result
            parts.append(value)
            statistics.append(stats)
            tokens.append(token)
            confidence.append(logc)
        return (torch.stack(parts).sum(), torch.stack(statistics).sum(0),
                torch.cat(tokens).reshape(hidden.shape[:2]), torch.cat(confidence).reshape(hidden.shape[:2]))

    def forward(self, batch):
        cfg, draft = self.cfg, self.drafter
        eligible = ((batch["weights"] > 0) & (batch["positions"] > 0)).nonzero().flatten().tolist()
        if not eligible:
            raise ValueError("Packed batch has no trainable starts")
        selected = self.rng.sample(eligible, min(cfg.max_anchors, len(eligible)))
        device = batch["ids"].device
        indices = torch.tensor(selected, device=device)
        all_starts = batch["positions"][indices]
        all_docs = batch["documents"][indices]
        m = cfg.num_proposals
        distance = torch.arange(m, device=device)
        decay = torch.exp(-distance.float() / cfg.loss_decay)
        numerator = None
        denominator = torch.zeros((), device=device)
        stats1 = torch.zeros(3, device=device)
        stats2 = torch.zeros(3, device=device)
        weight1, weight2 = torch.zeros((), device=device), torch.zeros((), device=device)
        boundary_sum = torch.zeros((), device=device)
        predicted_sum = torch.zeros((), device=device)

        def supervision(starts, docs):
            positions = starts[:, None] + distance
            valid = positions < batch["lengths"][docs, None]
            idx = batch["offsets"][docs, None] + torch.minimum(positions, batch["lengths"][docs, None] - 1)
            weights = batch["weights"][idx] * valid * decay
            # Hidden at token t-1 predicts token t; never use hidden at t.
            teacher = batch["final"][idx - 1]
            return teacher, weights, batch["ids"][idx], valid

        for begin in range(0, len(selected), cfg.blocks_per_chunk):
            starts = all_starts[begin:begin + cfg.blocks_per_chunk]
            docs = all_docs[begin:begin + cfg.blocks_per_chunk]
            anchors = packed_anchors(batch, starts, starts, docs, draft.hidden_size, cfg.num_anchors)
            first = draft.packed_hidden(batch["features"][None], batch["positions"], batch["documents"],
                                        anchors, starts, starts, docs)
            teacher, w1, gold, valid = supervision(starts, docs)
            loss1, st1, tokens, logc = self.loss_and_predictions(first, teacher, w1)
            predicted = rejection_boundary(logc)
            keep = predicted.clone()
            # Python RNG is saved per rank; worker prefetch never advances it.
            for i in range(len(starts)):
                if self.rng.random() < cfg.boundary_exploration:
                    keep[i] = self.rng.choices(range(m), weights=[cfg.boundary_beta ** j for j in range(m)])[0]
            matches = (tokens == gold) & valid
            match_prefix = matches.long().cumprod(-1).sum(-1)
            keep = torch.minimum(keep, match_prefix)
            keep = torch.minimum(keep, batch["lengths"][docs] - starts - 1)
            second_starts = starts + keep
            if not keep.any():
                # With keep=0 the input, labels and masks are identical. DFlash
                # has no dropout here, so reuse the exact view and its gradient.
                w2, loss2, st2 = w1 * cfg.redraft_weight, loss1 * cfg.redraft_weight, st1 * cfg.redraft_weight
            else:
                anchors2 = packed_anchors(batch, starts, second_starts, docs, draft.hidden_size,
                                          cfg.num_anchors, first, tokens)
                second = draft.packed_hidden(batch["features"][None], batch["positions"], batch["documents"],
                                             anchors2, second_starts, starts, docs)
                teacher2, w2, _, _ = supervision(second_starts, docs)
                w2 = w2 * cfg.redraft_weight
                loss2, st2, _, _ = self.loss_and_predictions(second, teacher2, w2)
            value = loss1 + loss2
            numerator = value if numerator is None else numerator + value
            denominator = denominator + w1.sum() + w2.sum()
            stats1 += st1
            stats2 += st2
            weight1 += w1.sum()
            weight2 += w2.sum()
            boundary_sum += keep.sum()
            predicted_sum += predicted.sum()
        # Sufficient statistics permit correct token-weighted DDP normalization.
        metrics = torch.cat([numerator.detach()[None], denominator[None], stats1, weight1[None],
                             stats2, weight2[None], boundary_sum[None], predicted_sum[None],
                             denominator.new_tensor([len(selected)])])
        return numerator, denominator, metrics
