"""Cross-entropy plus asymmetric accepted-length prediction cost."""

import math

import torch
from torch.nn import functional as F


def acceptance_boundary_loss(
    logits: torch.Tensor,
    accepted_lengths: torch.Tensor,
    *,
    under_weight: float = 3.0,
    over_weight: float = 1.0,
    cost_weight: float = 1.0,
    validate: bool = False,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Train a classifier over accepted lengths 0, 1, ..., K.

    Args:
        logits:
            Raw head outputs: [B, K + 1].
            Do not apply softmax before calling this function.

        accepted_lengths:
            Ground-truth accepted draft lengths: [B], dtype torch.long.
            Excludes the anchor and target-generated correction/bonus token.

        under_weight:
            Cost per missed acceptable token.

        over_weight:
            Cost per extra proposed token beyond the true boundary.

        cost_weight:
            Strength of the asymmetric cost relative to cross-entropy.
            Set to zero for ordinary cross-entropy training.

        validate:
            Enable value checks during initial debugging.
            These checks may synchronize the accelerator.

    Returns:
        loss:
            Differentiable scalar used for backpropagation.

        metrics:
            Detached scalar tensors for logging.
    """
    if logits.ndim != 2:
        raise ValueError("logits must have shape [B, K + 1].")

    batch_size, num_classes = logits.shape
    num_proposals = num_classes - 1

    if batch_size < 1 or num_proposals < 1:
        raise ValueError("A nonempty batch and at least two classes are required.")

    if accepted_lengths.shape != (batch_size,):
        raise ValueError("accepted_lengths must have shape [B].")

    if accepted_lengths.dtype != torch.long:
        raise TypeError("accepted_lengths must have dtype torch.long.")

    if accepted_lengths.device != logits.device:
        raise ValueError("Labels and logits must be on the same device.")

    weights = (under_weight, over_weight, cost_weight)
    if any(not math.isfinite(w) or w < 0 for w in weights):
        raise ValueError("Loss weights must be finite and nonnegative.")

    if under_weight <= over_weight:
        raise ValueError(
            "under_weight must exceed over_weight "
            "to penalize underprediction more strongly."
        )

    if validate:
        if not bool(torch.isfinite(logits).all()):
            raise ValueError("logits contain nonfinite values.")

        valid_labels = (
            (accepted_lengths >= 0)
            & (accepted_lengths <= num_proposals)
        )
        if not bool(valid_labels.all()):
            raise ValueError(
                f"accepted_lengths must be between 0 and {num_proposals}."
            )

    with torch.autocast(
        device_type=logits.device.type,
        enabled=False,
    ):
        scores = logits.float()

        # Rewards assigning probability to the exact accepted length.
        ce_loss = F.cross_entropy(
            scores,
            accepted_lengths,
        )

        probabilities = scores.softmax(dim=-1)

        possible_lengths = torch.arange(
            num_classes,
            device=logits.device,
            dtype=torch.float32,
        ).unsqueeze(0)

        true_lengths = accepted_lengths.float().unsqueeze(-1)

        # [B, K + 1]: distance cost for every possible predicted length.
        under_distance = (
            true_lengths - possible_lengths
        ).clamp_min(0)

        over_distance = (
            possible_lengths - true_lengths
        ).clamp_min(0)

        cost_matrix = (
            under_weight * under_distance
            + over_weight * over_distance
        )

        # Normalize by block proposal count to control the loss scale.
        expected_cost = (
            probabilities * cost_matrix
        ).sum(dim=-1).mean() / num_proposals

        loss = ce_loss + cost_weight * expected_cost

        # Argmax is used only for reporting, never to compute gradients.
        with torch.no_grad():
            predictions = scores.argmax(dim=-1)
            errors = predictions - accepted_lengths

            under_tokens = (-errors).clamp_min(0).float()
            over_tokens = errors.clamp_min(0).float()

            metrics = {
                "loss": loss.detach(),
                "cross_entropy": ce_loss.detach(),
                "expected_cost_normalized": expected_cost.detach(),
                "exact_accuracy": errors.eq(0).float().mean(),
                "mean_under_tokens": under_tokens.mean(),
                "mean_over_tokens": over_tokens.mean(),
                "decision_cost": (
                    under_weight * under_tokens
                    + over_weight * over_tokens
                ).mean(),
            }

    return loss, metrics