"""Shared acceptance-head features for data collection and inference."""

import torch


FEATURE_SCHEMA = "draft_hidden_plus_five_statistics_v1"


@torch.no_grad()
def build_acceptance_features(
    hidden_states: torch.Tensor,
    candidate_ids: torch.Tensor,
    unary_scores: torch.Tensor,
    selected_token_ids: torch.Tensor,
    final_scores: torch.Tensor,
    *,
    validate: bool = False,
) -> torch.Tensor:
    """
    Construct features after the prefix selector chooses draft tokens.

    Args:
        hidden_states:
            Draft hidden states: [B, K, H].

        candidate_ids:
            Shortlisted token IDs: [B, K, C].

        unary_scores:
            Original DFlash scores for those candidates: [B, K, C].

        selected_token_ids:
            Tokens chosen by the prefix selector: [B, K].

        final_scores:
            Candidate scores AFTER prefix re-ranking: [B, K, C].
            These must be calculated along the actual selected prefix,
            not a teacher-forced prefix.

        validate:
            Check candidate membership and finite values.
            Enable for initial validation. These checks can synchronize
            the accelerator, so disable in the normal serving path.

    Returns:
        Detached BF16 features: [B, K, H + 5].

    Notes:
        K excludes the anchor token.
        C is the shortlist size, not the vocabulary size.
        Shortlist-normalized scores are not full-vocabulary probabilities.
    """
    if hidden_states.ndim != 3:
        raise ValueError("hidden_states must have shape [B, K, H].")

    if candidate_ids.ndim != 3:
        raise ValueError("candidate_ids must have shape [B, K, C].")

    if unary_scores.shape != candidate_ids.shape:
        raise ValueError("unary_scores and candidate_ids must match.")

    if final_scores.shape != candidate_ids.shape:
        raise ValueError("final_scores and candidate_ids must match.")

    batch_size, num_proposals, num_candidates = candidate_ids.shape

    if num_proposals < 1:
        raise ValueError("At least one draft proposal is required.")

    if num_candidates < 2:
        raise ValueError("At least two candidates are needed for the margin.")

    expected_positions = (batch_size, num_proposals)

    if hidden_states.shape[:2] != expected_positions:
        raise ValueError("Hidden states and candidate positions must match.")

    if selected_token_ids.shape != expected_positions:
        raise ValueError("selected_token_ids must have shape [B, K].")

    tensors = (
        candidate_ids,
        unary_scores,
        selected_token_ids,
        final_scores,
    )
    if any(t.device != hidden_states.device for t in tensors):
        raise ValueError("All feature inputs must be on the same device.")

    # Locate the selected token within each position's shortlist.
    matches = candidate_ids.eq(selected_token_ids.unsqueeze(-1))

    if validate:
        if not bool(matches.sum(dim=-1).eq(1).all()):
            raise ValueError(
                "Each selected token must occur exactly once "
                "in its candidate shortlist."
            )

        for name, tensor in (
            ("hidden_states", hidden_states),
            ("unary_scores", unary_scores),
            ("final_scores", final_scores),
        ):
            if not bool(torch.isfinite(tensor).all()):
                raise ValueError(f"{name} contains nonfinite values.")

    selected_indices = matches.long().argmax(
        dim=-1,
        keepdim=True,
    )

    with torch.autocast(
        device_type=hidden_states.device.type,
        enabled=False,
    ):
        unary = unary_scores.float()
        final = final_scores.float()

        selected_unary = unary.gather(
            dim=-1,
            index=selected_indices,
        )

        selected_final = final.gather(
            dim=-1,
            index=selected_indices,
        )

        # 1. How far the selected candidate is below the unary maximum.
        unary_relative_score = (
            selected_unary - unary.amax(dim=-1, keepdim=True)
        )

        # 2. Log-softmax score within the final candidate shortlist.
        selected_log_score = (
            selected_final - final.logsumexp(dim=-1, keepdim=True)
        )

        # 3. Margin between the two highest final candidate scores.
        top_two = final.topk(k=2, dim=-1).values
        final_margin = (
            top_two[..., 0:1] - top_two[..., 1:2]
        )

        # 4. Correction applied by the prefix selector.
        prefix_correction = selected_final - selected_unary

        # 5. Positions 1/K, 2/K, ..., 1.
        positions = torch.arange(
            1,
            num_proposals + 1,
            device=hidden_states.device,
            dtype=torch.float32,
        ) / num_proposals

        positions = positions.view(
            1, num_proposals, 1
        ).expand(batch_size, -1, -1)

        statistics = torch.cat(
            [
                unary_relative_score,
                selected_log_score,
                final_margin,
                prefix_correction,
                positions,
            ],
            dim=-1,
        )

        # Use identical feature precision during collection and serving.
        features = torch.cat(
            [
                hidden_states.to(torch.bfloat16),
                statistics.to(torch.bfloat16),
            ],
            dim=-1,
        )

    return features.detach()