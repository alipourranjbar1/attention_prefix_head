"""Joint stored-response training with genuine prefix-selected verification."""
import torch

from speculators.model import SpeculatorModel
from speculators.models.dflash_prefix.core import reachable_greedy_prefix
from speculators.models.dflash_prefix.selector_v2 import unary_topk
from speculators.models.retrace.stored_training import ReTraceDraftModel as StoredReTrace

from .core import ReTracePrefixDraftModel
from .loss import topk_kl


@SpeculatorModel.register("retrace_prefix_stored_training")
class ReTracePrefixTraining(StoredReTrace, ReTracePrefixDraftModel):
    def auxiliary_loss(self, block, targets, valid, anchors, reference):
        config = self.config
        kl = topk_kl(block.logits, targets, valid, config.distill_top_k)
        hidden, unary = block.hidden, block.logits
        if config.prefix_detach_backbone:
            hidden, unary = hidden.detach(), unary.detach()
        values, candidates = unary_topk(unary, config.prefix_top_k)
        scores = self.prefix_head.score_teacher_prefix(
            hidden, candidates, values, anchors, reference, **self._prefix_kwargs()
        ).float()
        target_ids = targets.argmax(-1)
        matches = candidates.eq(target_ids[..., None])
        covered = matches.any(-1)
        mask = valid
        if config.prefix_loss_kind == "target_ce":
            ce = -scores.log_softmax(-1).gather(-1, matches.long().argmax(-1)[..., None]).squeeze(-1)
            mask = mask & covered
        else:
            teacher = targets.detach().float().gather(-1, candidates).softmax(-1)
            ce = -(teacher * scores.log_softmax(-1)).sum(-1)
        if config.prefix_loss_scope == "reachable_greedy_prefix":
            mask = mask & reachable_greedy_prefix(
                reference, target_ids, candidates.eq(reference[..., None]).any(-1)
            )
        weights = mask.float() * torch.exp(-torch.arange(valid.shape[-1], device=valid.device).float() / 4.0)
        weights = weights * torch.where(unary.argmax(-1).eq(target_ids), config.prefix_damage_weight, 1.0)
        # Fixed denominator makes accumulation invariant to block chunk size.
        selector = (ce * weights).sum() / valid.sum().clamp_min(1)
        return config.prefix_loss_alpha * selector + config.distill_kl_weight * kl, {
            "selector_ce": selector, "top16_kl": kl,
            "weighted_top16_kl": config.distill_kl_weight * kl,
        }
