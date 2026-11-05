"""Checkpoint and fixed-window contracts shared by the vLLM adapter and tests."""
from dataclasses import dataclass
from pathlib import Path
import json

import torch

from .model import make_anchors, select_boundary


@dataclass(frozen=True)
class ServingSettings:
    checkpoint: str
    num_proposals: int = 15
    mode: str = "two_pass"
    trace_steps: bool = False
    profile_steps: int = 0
    draft_graph: bool = False
    anchor_policy: str = "trained"

    def __post_init__(self):
        if self.profile_steps < 0:
            raise ValueError("profile_steps must be nonnegative")
        if self.num_proposals < 1:
            raise ValueError("num_proposals must be positive")
        if self.mode not in {"two_pass", "one_pass"}:
            raise ValueError("mode must be two_pass or one_pass")
        if self.anchor_policy not in {"trained", "accepted_prefix"}:
            raise ValueError("Unknown anchor_policy")
        if self.anchor_policy == "accepted_prefix" and self.mode != "two_pass":
            raise ValueError("accepted_prefix requires two_pass")

    def metadata(self):
        path = Path(self.checkpoint)
        metadata = json.loads((path / "four_anchor_config.json").read_text())
        if metadata.get("architecture") != "FourAnchorDFlash":
            raise ValueError("Expected an existing FourAnchorDFlash checkpoint")
        cfg = metadata["draft_config"]
        if cfg["num_anchors"] != 4:
            raise ValueError("This adapter requires four conditioning anchors")
        layout = cfg.get("anchor_layout", "past")
        if self.anchor_policy == "accepted_prefix" and layout != "target_four":
            raise ValueError("accepted_prefix requires a format-3 target_four checkpoint")
        if layout in {"bonus_in_four", "target_four"}:
            version = 3 if layout == "target_four" else 2
            if (cfg.get("format_version"), cfg["num_proposals"], self.num_proposals) != (version, 15, 15):
                raise ValueError(f"{layout} requires a format-{version} checkpoint trained for 15 proposals")
        elif layout != "past":
            raise ValueError("Unknown checkpoint anchor layout")
        elif self.num_proposals + 1 > cfg["num_proposals"]:
            raise ValueError("Serving proposals plus the known bonus cannot exceed the trained slot count")
        if not (path / "model.safetensors").is_file():
            raise ValueError("Checkpoint model.safetensors is missing")
        return metadata

    def inference_version(self):
        layout = self.metadata()["draft_config"].get("anchor_layout", "past")
        if self.anchor_policy == "accepted_prefix":
            return "accepted_prefix19_v1"
        return {"past": "known_bonus_cached_v1", "bonus_in_four": "bonus_in_four_cached_v2",
                "target_four": "target_four_materialized_v3"}[layout]

    def apply(self, drafter):
        # The versioned proposer owns the query layout; never rewrite a
        # checkpoint's trained slot count or hidden-state alignment here.
        self.metadata()


def retained_forward_tokens(query_tokens, draft_tokens, sampled_tokens):
    """How many current target inputs belong to the accepted prefix?

    vLLM processes the previous bonus followed by draft candidates. Its output
    contains accepted candidates and ONE new correction/bonus. That NEW bonus
    is not among the inputs. Drafting uses its embedding, not target features.
    """
    if sampled_tokens < 1 or draft_tokens < 0 or query_tokens <= draft_tokens:
        raise ValueError("Invalid target verification layout")
    if sampled_tokens > draft_tokens + 1:
        raise ValueError("More sampled tokens than candidates plus one bonus")
    return query_tokens - draft_tokens + sampled_tokens - 1


class FeatureHistory:
    """Single-request, absolute-position history; rejected features are discarded."""
    def __init__(self, capacity, width, device, dtype):
        self.ids = torch.empty((1, capacity), device=device, dtype=torch.long)
        self.features = torch.empty((1, capacity, width), device=device, dtype=dtype)
        self.request_id = None
        self.length = 0

    def update(self, request_id, start, ids, features):
        if request_id != self.request_id or start == 0:
            self.request_id, self.length = request_id, 0
        if start < 0 or start > self.length:
            raise ValueError("Missing target features: prefix caching/chunk gaps are unsupported")
        count = ids.numel()
        end = start + count
        if end > self.ids.shape[1] or features.shape != (count, self.features.shape[-1]):
            raise ValueError("Invalid target feature shape or context capacity")
        self.ids[0, start:end].copy_(ids.reshape(-1))
        self.features[0, start:end].copy_(features)
        self.length = end

    def prefix(self):
        return self.ids[:, :self.length], self.features[:, :self.length]


@torch.inference_mode()
def fixed_window_proposal(drafter, ids, features, mode="two_pass"):
    """Refine M positions twice; return exactly M candidates, never M + boundary."""
    if mode not in {"one_pass", "two_pass"}:
        raise ValueError("Unknown drafting mode")
    cfg = drafter.cfg
    n = ids.shape[1]
    anchors = make_anchors(ids, features, cfg.num_anchors, drafter.hidden_size)
    first = drafter(features, anchors, n)
    if mode == "one_pass":
        return first.tokens, None
    boundary = select_boundary(first.logits)
    anchors = make_anchors(ids, features, cfg.num_anchors, drafter.hidden_size,
                           first, boundary)
    second = drafter(features, anchors, n + boundary)
    # The second pass still sees M mask slots, matching the chosen serving
    # shape. Only the suffix inside the original M-position window is kept.
    tokens = torch.cat([first.tokens[:, :boundary],
                        second.tokens[:, :cfg.num_proposals - boundary]], dim=1)
    if tokens.shape != (1, cfg.num_proposals):
        raise RuntimeError("Fixed-window proposer returned the wrong token count")
    return tokens, boundary
