"""Optional memory-bounded evaluation of the native DLoop weighted CE.

Only prediction rows are chunked. Every row still uses the full vocabulary.
Non-reentrant checkpointing releases each chunk's logits/log-softmax after
forward and recomputes them for backward. Carrier gradients are not detached.
The frozen head is reacquired from the module during recomputation, so no
stale FSDP parameter view is cached. The upstream source file stays unchanged.
"""
from types import MethodType

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def _chunk_stats(head, rows, targets, weights, loss_weights):
    logits = head(rows)
    ce = F.cross_entropy(logits, targets, reduction="none")
    with torch.no_grad():
        correct = ((logits.argmax(dim=-1) == targets) & (weights > 0.5)).sum().float()
    return (ce * loss_weights).sum(), correct


def _block_loss_chunked(self, pred_hidden, input_ids, loss_mask, anchors, keep, label_offset):
    bsz, n_blocks, n_rows, width = pred_hidden.shape
    seq_len = input_ids.shape[1]
    label_idx = anchors.unsqueeze(-1) + torch.arange(
        label_offset, label_offset + n_rows, device=input_ids.device)
    safe_idx = label_idx.clamp(max=seq_len - 1)
    targets = torch.gather(input_ids.unsqueeze(1).expand(-1, n_blocks, -1), 2, safe_idx)
    weights = keep.unsqueeze(-1).float() * (label_idx < seq_len).float()
    weights = weights * torch.gather(loss_mask.unsqueeze(1).expand(-1, n_blocks, -1), 2, safe_idx)
    loss_weights = weights
    if self.loss_decay_gamma is not None and self.loss_decay_gamma > 0:
        depth = torch.arange(n_rows, device=input_ids.device).view(1, 1, -1)
        loss_weights = loss_weights * torch.exp(-depth.float() / self.loss_decay_gamma)
    rows = pred_hidden.reshape(-1, width)
    targets, weights, loss_weights = (x.reshape(-1) for x in (targets, weights, loss_weights))
    ce_sum = torch.zeros((), dtype=loss_weights.dtype, device=rows.device)
    correct = torch.zeros_like(ce_sum)

    # The function captures the module only. Chunk tensors are explicit inputs;
    # no mutable loop index or saved parameter view is captured by the closure.
    def evaluate(r, t, w, lw):
        return _chunk_stats(self.lm_head, r, t, w, lw)

    for start in range(0, rows.shape[0], self._head_loss_chunk_size):
        end = start + self._head_loss_chunk_size
        args = (rows[start:end], targets[start:end], weights[start:end], loss_weights[start:end])
        if torch.is_grad_enabled() and rows.requires_grad:
            value, n_correct = checkpoint(evaluate, *args, use_reentrant=False, preserve_rng_state=False)
        else:
            value, n_correct = evaluate(*args)
        ce_sum = ce_sum + value
        correct = correct + n_correct
    return ce_sum, loss_weights.sum(), correct, weights.sum()


def enable_chunked_ce(model, chunk_size):
    from specforge.core.dflash import OnlineDFlashModel
    if type(model) is not OnlineDFlashModel or model.loss_type != "dflash" or model.loop_max_stage < 1:
        raise ValueError("Chunked CE supports the native regular DFlash DLoop objective only")
    if chunk_size <= 0:
        raise ValueError("head_loss_chunk_size must be positive")
    if any(p.requires_grad for p in model.lm_head.parameters()):
        raise ValueError("Chunked CE expects the native frozen target head")
    model._head_loss_chunk_size = int(chunk_size)
    model._block_loss = MethodType(_block_loss_chunked, model)
    return model
