"""Chunked exact full-vocabulary DSpark loss with trainable Markov heads.

Unlike FrozenHeadLoss, this differentiates through both Markov factors and
the confidence head. Non-reentrant checkpointing discards vocabulary-sized
activations between chunks. The teacher distribution remains frozen.
"""
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def projected_dspark_loss(draft, hidden, previous_ids, cache, indices, weights,
                         mix, confidence_alpha, chunk_size):
    if getattr(draft, "head_loss_backend", "checkpoint") == "cached_vjp":
        from .dspark_cached_vjp import cached_vjp_loss
        return cached_vjp_loss(draft, hidden, previous_ids, cache, indices, weights,
                               mix, confidence_alpha, chunk_size)
    flat = hidden.reshape(-1, hidden.shape[-1])
    previous_ids, indices, weights = previous_ids.flatten(), indices.flatten(), weights.flatten()
    coefficients = flat.new_tensor([mix.get(k, 0.) for k in ("ce", "tv", "kl_div")], dtype=torch.float32)

    def project(h, previous, target_indices, w):
        logits, confidence = draft.project(h, previous)
        if getattr(draft, "loss_graph", "off") == "npu":
            from .dspark_loss_graph import distribution_loss
            return distribution_loss(logits, confidence, cache[0][target_indices],
                                     cache[1][target_indices], w, mix, confidence_alpha,
                                     chunk_size=chunk_size, graph=True)
        logq = F.log_softmax(logits.float(), -1)
        p = cache[0][target_indices]
        labels = cache[1][target_indices]
        ce = -logq.gather(1, labels[:, None]).squeeze(1)
        overlap = torch.minimum(logq.exp(), p).sum(-1).clamp(0., 1.)
        tv = 1. - overlap
        kl = F.kl_div(logq, p, reduction="none").sum(-1) if mix.get("kl_div", 0.) else torch.zeros_like(tv)
        terms = torch.stack((ce, tv, kl), -1)
        # Upstream DSpark's analytical acceptance label; stop gradients through
        # the label so BCE cannot train the distribution to imitate the head.
        bce = F.binary_cross_entropy_with_logits(confidence.float(), overlap.detach(), reduction="none")
        total = (((terms * coefficients).sum(-1) + confidence_alpha * bce) * w).sum()
        stats = (terms * w[:, None]).sum(0).detach()
        confidence_stats = torch.stack(((bce * w).sum(),
            ((confidence.float().sigmoid() - overlap).abs() * w).sum())).detach()
        return total, stats, confidence_stats

    losses, statistics, confidence_statistics = [], [], []
    for begin in range(0, len(flat), chunk_size):
        end = begin + chunk_size
        args = (flat[begin:end], previous_ids[begin:end], indices[begin:end], weights[begin:end])
        result = checkpoint(project, *args, use_reentrant=False) if draft.training else project(*args)
        loss, stats, confidence_stats = result
        losses.append(loss)
        statistics.append(stats)
        confidence_statistics.append(confidence_stats)
    return torch.stack(losses).sum(), torch.stack(statistics).sum(0), torch.stack(confidence_statistics).sum(0)
