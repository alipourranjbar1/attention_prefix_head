"""Target-selected, restricted top-k KL in FP32 (teacher -> student)."""
import torch


def topk_kl(student, teacher, valid, k=16):
    if student.shape != teacher.shape or valid.shape != student.shape[:-1]:
        raise ValueError("Expected matching logits and a per-position mask")
    if not 1 <= k <= student.shape[-1]:
        raise ValueError("k must be in [1, vocabulary size]")
    values, ids = teacher.detach().float().topk(k, dim=-1)
    logp = values.log_softmax(-1)
    logq = student.float().gather(-1, ids).log_softmax(-1)
    divergence = (logp.exp() * (logp - logq)).sum(-1)
    weights = valid.float() * torch.exp(
        -torch.arange(valid.shape[-1], device=valid.device).float() / 4.0
    )
    # Match stored DFlash KL's valid-token denominator and position decay.
    return (divergence * weights).sum() / valid.sum().clamp_min(1)
