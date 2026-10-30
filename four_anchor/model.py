"""Regular DFlash decoder layers, with an explicit multi-anchor input interface.

No prefix-attention model, recurrent head, or MTP layer is instantiated here.
Target features are decoder-layer outputs before the target's final norm.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
import json

import torch
from torch import nn
from safetensors.torch import load_file, save_file
from transformers import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm, Qwen3RotaryEmbedding
from speculators.models.dflash.model_definitions import Qwen3DFlashDecoderLayer


@dataclass
class DraftConfig:
    target_name: str
    target_layer_ids: list[int]
    mask_token_id: int
    num_draft_layers: int = 5
    num_anchors: int = 4
    num_proposals: int = 16
    attention: str = "sdpa"
    target_revision: str | None = None
    format_version: int = 1

    def __post_init__(self):
        if self.num_anchors < 1 or self.num_proposals < 1 or self.num_draft_layers < 1:
            raise ValueError("Anchor, proposal, and layer counts must be positive")
        if not self.target_layer_ids or len(set(self.target_layer_ids)) != len(self.target_layer_ids):
            raise ValueError("target_layer_ids must be a nonempty list of distinct decoder indices")
        if self.attention not in {"sdpa", "eager"}:
            raise ValueError("This reference backend supports sdpa and eager attention")
        if self.format_version != 1:
            raise ValueError("Unsupported four-anchor checkpoint version")


@dataclass
class Anchors:
    ids: torch.Tensor             # [1, A]
    positions: torch.Tensor       # [1, A], original absolute positions
    target_features: torch.Tensor # [1, A, F]
    draft_features: torch.Tensor  # [1, A, H]
    source: torch.Tensor          # [1, A]: 0=padding, 1=target, 2=first-pass drafter


@dataclass
class DraftOutput:
    logits: torch.Tensor          # [1, M, vocab], anchors excluded
    hidden: torch.Tensor          # [1, M, H], final normalized drafter states

    @property
    def tokens(self):
        return self.logits.argmax(dim=-1)


def boundary_scores(logits: torch.Tensor) -> torch.Tensor:
    """D-Loop section 3.1 / Algorithm 1 boundary score, K=1, in log space.

    Returned index l means keep l proposals; proposal l (zero based) is the
    predicted first error. Includes l=0 and l=M-1, not the all-accepted case.
    Confidence is the untempered softmax probability of the greedy token.
    """
    x = logits.detach().float()
    log_c = x.amax(-1) - torch.logsumexp(x, dim=-1)
    log_c = log_c.clamp(max=0)
    prior = torch.cat([torch.zeros_like(log_c[..., :1]), log_c.cumsum(-1)[..., :-1]], -1)
    # -expm1 is accurate even when confidence is very close to one.
    rejection = (-torch.expm1(log_c)).log()
    return prior + rejection


def select_boundary(logits: torch.Tensor) -> int:
    if logits.shape[0] != 1:
        raise ValueError("Reference decoding is batch size one")
    scores = boundary_scores(logits)[0]
    # Exact confidence=1 everywhere: all rejection scores are zero. Stable first tie.
    return int(scores.argmax().item())


def make_anchors(prefix_ids, target_features, num_anchors, hidden_size,
                 first: DraftOutput | None = None, keep: int = 0) -> Anchors:
    """Use ONLY the four positions preceding the restart, never its bad token."""
    n = prefix_ids.shape[1]
    if n < 1 or target_features.shape[:2] != prefix_ids.shape:
        raise ValueError("Need a nonempty prefix and one target feature per prefix token")
    if keep < 0 or (keep and (first is None or keep > first.hidden.shape[1])):
        raise ValueError("Invalid retained prefix length")
    device, dtype = target_features.device, target_features.dtype
    ids = torch.zeros((1, num_anchors), device=device, dtype=torch.long)
    positions = torch.arange(n + keep - num_anchors, n + keep, device=device)[None]
    source = torch.zeros_like(ids)
    tf = torch.zeros((1, num_anchors, target_features.shape[-1]), device=device, dtype=dtype)
    df = torch.zeros((1, num_anchors, hidden_size), device=device, dtype=dtype)
    proposed = first.tokens if first is not None else None
    for slot, pos in enumerate(positions[0].tolist()):
        if 0 <= pos < n:
            ids[0, slot] = prefix_ids[0, pos]
            tf[0, slot] = target_features[0, pos]
            source[0, slot] = 1
        elif pos >= n:
            ids[0, slot] = proposed[0, pos - n]
            df[0, slot] = first.hidden[0, pos - n].detach()
            source[0, slot] = 2
    return Anchors(ids, positions.clamp_min(0), tf, df, source)


class FourAnchorDFlash(nn.Module):
    def __init__(self, cfg: DraftConfig, target):
        super().__init__()
        self.cfg = cfg
        tc = target.config
        if tc.model_type != "qwen3":
            raise ValueError("This implementation currently supports Qwen3 causal-LM targets")
        if any(i < 0 or i >= tc.num_hidden_layers for i in cfg.target_layer_ids):
            raise ValueError("target_layer_ids are zero-based target decoder layer indices")
        if not 0 <= cfg.mask_token_id < tc.vocab_size:
            raise ValueError("mask_token_id is outside the target vocabulary")
        self.target_signature = self.signature(tc)
        dc = deepcopy(tc)
        dc.num_hidden_layers = cfg.num_draft_layers
        dc.layer_types = ["full_attention"] * cfg.num_draft_layers
        dc.sliding_window = None
        dc.use_sliding_window = False
        dc.attention_dropout = 0.0
        dc._attn_implementation = cfg.attention
        self.layers = nn.ModuleList(Qwen3DFlashDecoderLayer(dc, i) for i in range(cfg.num_draft_layers))
        self.norm = Qwen3RMSNorm(tc.hidden_size, eps=tc.rms_norm_eps)
        self.hidden_norm = Qwen3RMSNorm(tc.hidden_size, eps=tc.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(dc)
        self.fc = nn.Linear(len(cfg.target_layer_ids) * tc.hidden_size, tc.hidden_size, bias=False)
        self.draft_anchor_proj = nn.Linear(tc.hidden_size, tc.hidden_size, bias=False)
        self.draft_anchor_norm = Qwen3RMSNorm(tc.hidden_size, eps=tc.rms_norm_eps)
        self.hidden_size = tc.hidden_size
        # Share the actual frozen target weights; no second full vocabulary allocation.
        self.embed_tokens = target.get_input_embeddings()
        self.lm_head = target.get_output_embeddings()
        self.embed_tokens.requires_grad_(False)
        self.lm_head.requires_grad_(False)
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear) and not name.startswith("lm_head"):
                nn.init.normal_(module.weight, mean=0, std=tc.initializer_range)
        nn.init.eye_(self.draft_anchor_proj.weight)
        self.to(device=self.embed_tokens.weight.device, dtype=self.embed_tokens.weight.dtype)
        self.gradient_checkpointing = False

    @staticmethod
    def signature(tc):
        keys = ("model_type", "vocab_size", "hidden_size", "num_hidden_layers", "num_attention_heads",
                "num_key_value_heads", "head_dim", "rope_theta", "rope_scaling", "rope_parameters",
                "rms_norm_eps", "max_position_embeddings")
        return {k: getattr(tc, k, None) for k in keys}

    def forward(self, context_features, anchors: Anchors, start: int) -> DraftOutput:
        a, m = self.cfg.num_anchors, self.cfg.num_proposals
        if anchors.ids.shape != (1, a) or context_features.shape[0] != 1:
            raise ValueError("Each reference pass needs one sequence and exactly num_anchors slots")
        if start < context_features.shape[1]:
            raise ValueError("Target features beyond the generation boundary would leak future tokens")
        device = context_features.device
        masked = torch.full((1, m), self.cfg.mask_token_id, device=device, dtype=torch.long)
        x = self.embed_tokens(torch.cat([anchors.ids, masked], dim=1))
        target_anchor = self.hidden_norm(self.fc(anchors.target_features))
        draft_anchor = self.draft_anchor_norm(self.draft_anchor_proj(anchors.draft_features))
        feature = (target_anchor * (anchors.source == 1)[..., None]
                   + draft_anchor * (anchors.source == 2)[..., None])
        x = x + torch.cat([feature, torch.zeros_like(x[:, a:])], dim=1)
        context = self.hidden_norm(self.fc(context_features))
        context_positions = torch.arange(context.shape[1], device=device)[None]
        slots = torch.cat([anchors.positions, torch.arange(start, start + m, device=device)[None]], 1)
        positions = torch.cat([context_positions, slots], dim=1)
        rope = self.rotary_emb(x, positions)
        key_valid = torch.cat([torch.ones_like(context_positions, dtype=torch.bool),
                               anchors.source != 0, torch.ones_like(masked, dtype=torch.bool)], 1)
        # Explicit additive mask works for eager and SDPA. Attention remains bidirectional.
        mask = torch.zeros((1, 1, a + m, key_valid.shape[1]), device=device, dtype=x.dtype)
        mask = mask.masked_fill(~key_valid[:, None, None], torch.finfo(x.dtype).min)
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                from torch.utils.checkpoint import checkpoint
                def run(h, ctx, layer=layer):
                    return layer(hidden_states=h, target_hidden=ctx, attention_mask=mask,
                                 position_ids=positions, position_embeddings=rope, use_cache=False)
                x = checkpoint(run, x, context, use_reentrant=False)
            else:
                x = layer(hidden_states=x, target_hidden=context, attention_mask=mask,
                          position_ids=positions, position_embeddings=rope, use_cache=False)
        hidden = self.norm(x[:, a:])
        return DraftOutput(self.lm_head(hidden), hidden)

    def save(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        metadata = {"architecture": "FourAnchorDFlash", "draft_config": asdict(self.cfg),
                    "target_signature": self.target_signature}
        (path / "four_anchor_config.json").write_text(json.dumps(metadata, indent=2) + "\n")
        state = {k: v.detach().cpu().contiguous() for k, v in self.state_dict().items()
                 if not k.startswith(("embed_tokens.", "lm_head."))}
        save_file(state, str(path / "model.safetensors"))

    @classmethod
    def load(cls, path, target):
        path = Path(path)
        metadata = json.loads((path / "four_anchor_config.json").read_text())
        if metadata["architecture"] != "FourAnchorDFlash":
            raise ValueError("Not a FourAnchorDFlash checkpoint")
        if metadata["target_signature"] != cls.signature(target.config):
            raise ValueError("Checkpoint and target architectures differ")
        model = cls(DraftConfig(**metadata["draft_config"]), target)
        state = load_file(str(path / "model.safetensors"))
        expected = {k for k in model.state_dict() if not k.startswith(("embed_tokens.", "lm_head."))}
        if set(state) != expected:
            raise ValueError(f"Checkpoint keys mismatch: {set(state) ^ expected}")
        model.load_state_dict(state, strict=False)
        return model

    def initialize_regular_dflash(self, path):
        """Import regular full-vocab speculators DFlash, rejecting prefix variants.

        The new state bridge is deliberately left newly initialized. This is a
        warm start, not a claim that a one-anchor checkpoint needs no finetuning.
        """
        path = Path(path)
        config = json.loads((path / "config.json").read_text())
        if config.get("speculators_model_type") != "dflash":
            raise ValueError("Warm start must be regular speculators DFlash, not prefix/DFlash2/MTP")
        if config.get("aux_hidden_state_layer_ids") != self.cfg.target_layer_ids:
            raise ValueError("Warm-start target feature layer IDs must match exactly")
        if config.get("draft_vocab_size") != self.embed_tokens.num_embeddings:
            raise ValueError("Only full-vocabulary warm starts are supported")
        source_config = Qwen3Config(**config["transformer_layer_config"])
        if source_config.num_hidden_layers != self.cfg.num_draft_layers:
            raise ValueError("Warm-start drafter depth must match num_draft_layers")
        if config.get("mask_token_id") != self.cfg.mask_token_id:
            raise ValueError("Warm-start mask_token_id must match")
        expected = {k for k in self.state_dict() if k.startswith(("layers.", "norm.", "fc.", "hidden_norm."))}
        state = {}
        index = path / "model.safetensors.index.json"
        files = sorted(set(json.loads(index.read_text())["weight_map"].values())) if index.exists() else ["model.safetensors"]
        for file in files:
            state.update({k: v for k, v in load_file(str(path / file)).items() if k in expected})
        if set(state) != expected:
            raise ValueError(f"Regular DFlash checkpoint is missing keys: {expected - set(state)}")
        self.load_state_dict(state, strict=False)
