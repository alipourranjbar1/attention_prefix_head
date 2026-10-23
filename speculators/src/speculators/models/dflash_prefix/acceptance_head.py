"""Small acceptance-length predictor for a frozen DFlash-Prefix model."""

import torch
from torch import nn
from torch.nn import functional as F


class AcceptedLengthHead(nn.Module):
    """
    Predict the number of consecutive accepted draft tokens.

    Input:
        features: [batch_size, num_proposals, hidden_size + 5]

    Output:
        logits: [batch_size, num_proposals + 1]

    Output class meanings:
        0: No draft tokens accepted.
        1: First draft token accepted.
        ...
        K: All K draft tokens accepted.

    The anchor token is excluded from K.
    """

    def __init__(
        self,
        hidden_size: int,
        num_proposals: int,
        projection_size: int = 32,
        mlp_hidden_size: int = 64,
    ):
        super().__init__()

        dimensions = (
            hidden_size,
            num_proposals,
            projection_size,
            mlp_hidden_size,
        )
        if any(value < 1 for value in dimensions):
            raise ValueError("All head dimensions must be positive.")

        self.hidden_size = hidden_size
        self.num_proposals = num_proposals
        self.num_extra_features = 5

        # Save these values alongside the trained weights so the head
        # can be reconstructed with exactly the same architecture.
        self.architecture = {
            "hidden_size": hidden_size,
            "num_proposals": num_proposals,
            "projection_size": projection_size,
            "mlp_hidden_size": mlp_hidden_size,
        }

        # Normalize draft hidden states separately from score features.
        self.hidden_norm = nn.LayerNorm(hidden_size)

        # The same projection is used at every draft position.
        self.position_projection = nn.Linear(
            hidden_size + self.num_extra_features,
            projection_size,
        )

        # Flattening preserves the order of the draft positions.
        self.boundary_classifier = nn.Sequential(
            nn.Linear(
                num_proposals * projection_size,
                mlp_hidden_size,
            ),
            nn.SiLU(),
            nn.Linear(
                mlp_hidden_size,
                num_proposals + 1,
            ),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        expected_shape = (
            self.num_proposals,
            self.hidden_size + self.num_extra_features,
        )

        if features.ndim != 3:
            raise ValueError(
                "Expected features with shape [batch, K, hidden_size + 5], "
                f"received {tuple(features.shape)}."
            )

        if tuple(features.shape[1:]) != expected_shape:
            raise ValueError(
                f"Expected trailing dimensions {expected_shape}, "
                f"received {tuple(features.shape[1:])}."
            )

        # Keep the small head in FP32, even when draft features are BF16.
        with torch.autocast(
            device_type=features.device.type,
            enabled=False,
        ):
            features = features.float()

            hidden = features[..., :self.hidden_size]
            statistics = features[..., self.hidden_size:]

            hidden = self.hidden_norm(hidden)

            combined = torch.cat(
                [hidden, statistics],
                dim=-1,
            )

            projected = F.silu(
                self.position_projection(combined)
            )

            block_features = projected.flatten(start_dim=1)

            # Return raw logits; cross_entropy applies log-softmax.
            logits = self.boundary_classifier(block_features)

        return logits