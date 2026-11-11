"""First-order DSpark loss without checkpointing/recomputing vocabulary logits.

The frozen LM projection, trainable vanilla Markov head, and confidence head
retain the same full-vocabulary objective. Save small input gradients and one
Markov-weight gradient per call, rather than vocabulary-sized activations.
This trades some persistent gradient memory for less recomputation. Benchmark
on the actual device before selecting it; no higher-order AD is supported.
"""
import torch
from torch.nn import functional as F
from torch.autograd.function import once_differentiable

from .dspark_loss_graph import _kernel


class _CachedVJP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, previous, lm_weight, markov_weight, conf_weight,
                conf_bias, probabilities, labels, indices, weights,
                coefficients, alpha, chunk_size, compute_dtype):
        if lm_weight.requires_grad or probabilities.requires_grad or weights.requires_grad:
            raise ValueError("LM head, teacher probabilities and weights must be frozen")
        if conf_weight.shape != (1, hidden.shape[-1] + previous.shape[-1]):
            raise ValueError("Expected the vanilla DSpark confidence head")
        dh, de = torch.empty_like(hidden), torch.empty_like(previous)
        dw, dcw = torch.zeros_like(markov_weight), torch.zeros_like(conf_weight)
        dcb = torch.zeros_like(conf_bias)
        total = hidden.new_zeros((), dtype=torch.float32)
        stats = hidden.new_zeros(3, dtype=torch.float32)
        cs = hidden.new_zeros(2, dtype=torch.float32)
        # Explicit casts are local to this call; never retain FSDP parameter
        # views or stale weight copies across optimizer updates.
        with torch.autocast(hidden.device.type, enabled=False):
            lm, mw = lm_weight.to(compute_dtype), markov_weight.to(compute_dtype)
            cw, cb = conf_weight.to(compute_dtype), conf_bias.to(compute_dtype)
            width = hidden.shape[-1]
            for begin in range(0, len(hidden), chunk_size):
                s = slice(begin, min(begin + chunk_size, len(hidden)))
                h, e = hidden[s].to(compute_dtype), previous[s].to(compute_dtype)
                joined = torch.cat((h, e), -1)
                logits = F.linear(h, lm) + F.linear(e, mw)
                confidence = F.linear(joined, cw, cb).squeeze(-1)
                value, st, cst, dz, dc = _kernel(
                    logits, confidence, probabilities[indices[s]], labels[indices[s]],
                    weights[s], coefficients, alpha)
                total.add_(value)
                stats.add_(st)
                cs.add_(cst)
                dc = dc[:, None]
                # Cast each branch before accumulation, matching autocast's
                # linear-backward conversion to the original input dtype.
                conf_input_grad = dc @ cw
                dh[s] = (dz @ lm).to(hidden.dtype) + conf_input_grad[:, :width].to(hidden.dtype)
                de[s] = (dz @ mw).to(previous.dtype) + conf_input_grad[:, width:].to(previous.dtype)
                dw.add_((dz.T @ e).to(markov_weight.dtype))
                dcw.add_((dc.T @ joined).to(conf_weight.dtype))
                dcb.add_(dc.sum(0).to(conf_bias.dtype))
        ctx.save_for_backward(dh, de, dw, dcw, dcb)
        ctx.mark_non_differentiable(stats, cs)
        return total, stats, cs

    @staticmethod
    @once_differentiable
    def backward(ctx, scale, _stats, _cs):
        dh, de, dw, dcw, dcb = ctx.saved_tensors
        return (dh * scale, de * scale, None, dw * scale, dcw * scale,
                dcb * scale, None, None, None, None, None, None, None, None)


def cached_vjp_loss(draft, hidden, previous_ids, cache, indices, weights,
                    mix, confidence_alpha, chunk_size):
    if getattr(draft, "loss_graph", "off") != "off":
        raise ValueError("cached_vjp requires LOSS_GRAPH=off")
    if draft.lm_head.bias is not None or draft.markov_head.markov_w2.bias is not None:
        raise ValueError("cached_vjp supports the existing bias-free DSpark projections")
    flat = hidden.reshape(-1, hidden.shape[-1])
    previous = draft.markov_head.prev_embeddings(previous_ids.flatten())
    dtype = (torch.get_autocast_dtype(hidden.device.type)
             if torch.is_autocast_enabled(hidden.device.type) else hidden.dtype)
    return _CachedVJP.apply(flat, previous, draft.lm_head.weight,
        draft.markov_head.markov_w2.weight, draft.confidence_head.proj.weight,
        draft.confidence_head.proj.bias, cache[0], cache[1], indices.flatten(), weights.flatten(),
        tuple(float(mix.get(k, 0.)) for k in ("ce", "tv", "kl_div")),
        float(confidence_alpha), chunk_size, dtype)
