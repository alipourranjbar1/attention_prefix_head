"""Exact first-order full-vocabulary loss for a frozen output projection.

Compute dL/dhidden while the vocabulary chunk is resident, saving only that
small gradient. Backward no longer recomputes either vocabulary projection.
Teacher probabilities are FP32 (no top-k or probability quantization).
"""
import torch
from torch.nn import functional as F
from torch.autograd.function import once_differentiable


@torch.no_grad()
def teacher_cache(final_hidden, norm, head, chunk_size):
    count = final_hidden.shape[0]
    probabilities = torch.empty((count, head.weight.shape[0]), device=final_hidden.device, dtype=torch.float32)
    labels = torch.empty(count, device=final_hidden.device, dtype=torch.long)
    for begin in range(0, count, chunk_size):
        logits = head(norm(final_hidden[begin:begin + chunk_size])).float()
        probabilities[begin:begin + chunk_size] = F.softmax(logits, -1)
        labels[begin:begin + chunk_size] = logits.argmax(-1)
    return probabilities, labels


class FrozenHeadLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, head_weight, probabilities, labels, indices, weights, coefficients, chunk_size):
        if head_weight.requires_grad:
            raise ValueError("Cached loss requires a frozen vocabulary head")
        ce_weight, tv_weight, kl_weight = coefficients
        count = hidden.shape[0]
        loss = hidden.new_empty(count, dtype=torch.float32)
        stats = hidden.new_zeros((count, 3), dtype=torch.float32)
        tokens = indices.new_empty(count)
        confidence = loss.new_empty(count)
        hidden_grad = torch.empty_like(hidden)
        for begin in range(0, count, chunk_size):
            end = min(begin + chunk_size, count)
            logits = F.linear(hidden[begin:end], head_weight)
            logq = F.log_softmax(logits.float(), -1)
            q = logq.exp()
            p = probabilities[indices[begin:end]]
            target = labels[indices[begin:end]]
            ce = -logq.gather(1, target[:, None]).squeeze(1)
            raw_tv = 1 - torch.minimum(q, p).sum(-1)
            tv = raw_tv.clamp_min(0)
            kl = F.kl_div(logq, p, reduction="none").sum(-1) if kl_weight else torch.zeros_like(tv)
            w = weights[begin:end]
            stats[begin:end] = torch.stack([ce, tv, kl], -1) * w[:, None]
            loss[begin:end] = (ce_weight * ce + tv_weight * tv + kl_weight * kl) * w
            tokens[begin:end] = logits.argmax(-1)
            confidence[begin:end] = logq.amax(-1)
            # CE: q - one_hot(y). TV: q * (sum(a*q) - a), with a=1
            # for q<p and a=1/2 on ties (torch.minimum's exact convention).
            grad = q * ce_weight
            grad.scatter_add_(1, target[:, None], torch.full_like(target[:, None], -ce_weight, dtype=q.dtype))
            if tv_weight:
                a = (q < p).to(q.dtype) + .5 * (q == p).to(q.dtype)
                tv_grad = q * ((a * q).sum(-1, keepdim=True) - a)
                grad.add_(tv_weight * tv_grad * (raw_tv >= 0)[:, None])
            if kl_weight:
                grad.add_(kl_weight * (q * p.sum(-1, keepdim=True) - p))
            grad.mul_(w[:, None])
            # Match the cast through the projected logits in ordinary autocast
            # backward, then return the gradient in the original hidden dtype.
            hidden_grad[begin:end] = (grad.to(logits.dtype) @ head_weight.to(logits.dtype)).to(hidden.dtype)
        ctx.save_for_backward(hidden_grad)
        ctx.mark_non_differentiable(stats, tokens, confidence)
        return loss, stats, tokens, confidence

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_loss, _stats, _tokens, _confidence):
        (gradient,) = ctx.saved_tensors
        return gradient * grad_loss[:, None], None, None, None, None, None, None, None


def projected_loss(hidden, head, cache, indices, weights, mix, chunk_size):
    shape = hidden.shape[:2]
    result = FrozenHeadLoss.apply(hidden.reshape(-1, hidden.shape[-1]), head.weight,
                                  cache[0], cache[1], indices.flatten(), weights.flatten(),
                                  tuple(mix.get(k, 0.) for k in ("ce", "tv", "kl_div")), chunk_size)
    loss, stats, tokens, confidence = result
    return loss.reshape(shape).sum(-1), stats.reshape(*shape, 3).sum(1), tokens.reshape(shape), confidence.reshape(shape)
