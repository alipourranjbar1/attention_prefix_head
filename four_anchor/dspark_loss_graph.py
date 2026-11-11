"""Optional NPUGraph for DSpark's distribution loss and its exact first derivatives.

Only tensor-only CE/TV/KL/confidence work is captured. Projection, attention,
optimizer, packing, HTTP and FSDP/HCCL remain eager. Parameters never enter a
capture, so FSDP may freely reshard or replace its full parameter buffers.
The surrounding non-reentrant checkpoint discards vocabulary-sized gradients
until backward recomputation. No higher-order gradients are supported.
"""
import torch
from torch.nn import functional as F
from torch.autograd.function import once_differentiable

_CACHE = {}


def _kernel(logits, confidence, p, labels, weights, coefficients, alpha):
    ce_weight, tv_weight, kl_weight = coefficients
    logq = F.log_softmax(logits.float(), -1)
    q = logq.exp()
    raw_overlap = torch.minimum(q, p).sum(-1)
    overlap = raw_overlap.clamp(0., 1.)
    ce = -logq.gather(1, labels[:, None]).squeeze(1)
    tv = 1. - overlap
    kl = F.kl_div(logq, p, reduction="none").sum(-1) if kl_weight else torch.zeros_like(tv)
    c = confidence.float()
    bce = F.binary_cross_entropy_with_logits(c, overlap, reduction="none")
    terms = torch.stack((ce, tv, kl), -1)
    total = ((ce_weight * ce + tv_weight * tv + kl_weight * kl + alpha * bce) * weights).sum()
    stats = (terms * weights[:, None]).sum(0)
    conf_stats = torch.stack(((bce * weights).sum(), ((c.sigmoid() - overlap).abs() * weights).sum()))
    # The confidence BCE label is stop-gradient, exactly as in upstream DSpark.
    grad = ce_weight * q
    grad.scatter_add_(1, labels[:, None], torch.full_like(labels[:, None], -ce_weight, dtype=q.dtype))
    if tv_weight:
        a = (q < p).to(q.dtype) + .5 * (q == p).to(q.dtype)
        grad = grad + tv_weight * q * ((a * q).sum(-1, keepdim=True) - a) * (
            (raw_overlap >= 0) & (raw_overlap <= 1))[:, None]
    if kl_weight:
        grad = grad + kl_weight * (q * p.sum(-1, keepdim=True) - p)
    grad = (grad * weights[:, None]).to(logits.dtype)
    grad_confidence = (alpha * (c.sigmoid() - overlap) * weights).to(confidence.dtype)
    return total, stats, conf_stats, grad, grad_confidence


class _Replay:
    def __init__(self, inputs, coefficients, alpha):
        import torch_npu  # registers torch.npu; no global CUDA API substitution
        backend = torch.npu
        if not hasattr(backend, "NPUGraph"):
            raise RuntimeError("NPUGraph is unavailable; use LOSS_GRAPH=off or a supported torch_npu build")
        self.inputs = tuple(x.detach().clone() for x in inputs)
        self.replays = 0
        backend.synchronize()
        stream = backend.Stream()
        stream.wait_stream(backend.current_stream())
        with backend.stream(stream), torch.no_grad(), torch.autocast("npu", enabled=False):
            for _ in range(3):
                warm = _kernel(*self.inputs, coefficients, alpha)
            del warm
        backend.current_stream().wait_stream(stream)
        backend.synchronize()
        self.graph = backend.NPUGraph()
        # Capture failure propagates. Never silently advertise graphs while eager.
        with torch.no_grad(), torch.autocast("npu", enabled=False), backend.graph(self.graph):
            self.outputs = _kernel(*self.inputs, coefficients, alpha)

    def __call__(self, inputs):
        for fixed, value in zip(self.inputs, inputs):
            fixed.copy_(value)
        self.graph.replay()
        self.replays += 1
        # Outstanding autograd nodes must not alias buffers overwritten by replay.
        return tuple(value.clone() for value in self.outputs)


class _Loss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, confidence, probabilities, labels, weights, coefficients, alpha, rows, graph):
        count, vocab = logits.shape
        if not 0 < count <= rows or confidence.shape != (count,) or probabilities.shape != logits.shape:
            raise ValueError("Invalid loss chunk dimensions")
        if probabilities.requires_grad or weights.requires_grad:
            raise ValueError("Teacher probabilities and loss weights must be frozen")
        # Pad only the final vocabulary chunk; zero weights exclude padded rows.
        inputs = (F.pad(logits, (0, 0, 0, rows - count)),
                  F.pad(confidence, (0, rows - count)),
                  F.pad(probabilities, (0, 0, 0, rows - count)),
                  F.pad(labels, (0, rows - count)), F.pad(weights, (0, rows - count)))
        if graph:
            if logits.device.type != "npu":
                raise ValueError("NPUGraph loss requires NPU tensors")
            key = (str(logits.device), tuple((x.dtype, tuple(x.shape)) for x in inputs), coefficients, alpha)
            if key not in _CACHE:
                if len(_CACHE) >= 4:
                    raise RuntimeError("Too many loss graph signatures; fixed head_chunk_size/dtype is required")
                _CACHE[key] = _Replay(inputs, coefficients, alpha)
            outputs = _CACHE[key](inputs)
        else:
            outputs = _kernel(*inputs, coefficients, alpha)
        total, stats, conf_stats, grad, grad_confidence = outputs
        ctx.save_for_backward(grad[:count], grad_confidence[:count])
        ctx.mark_non_differentiable(stats, conf_stats)
        return total, stats, conf_stats

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_total, _stats, _confidence):
        grad, grad_confidence = ctx.saved_tensors
        return grad * grad_total, grad_confidence * grad_total, None, None, None, None, None, None, None


def distribution_loss(logits, confidence, probabilities, labels, weights, mix, alpha, chunk_size, graph=True):
    coefficients = tuple(float(mix.get(k, 0.)) for k in ("ce", "tv", "kl_div"))
    return _Loss.apply(logits, confidence, probabilities, labels, weights, coefficients, float(alpha), chunk_size, graph)


def graph_status():
    return {"loss_graph_captures": len(_CACHE), "loss_graph_replays": sum(g.replays for g in _CACHE.values())}
