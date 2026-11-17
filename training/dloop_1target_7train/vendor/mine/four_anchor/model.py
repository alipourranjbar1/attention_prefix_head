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
    anchor_layout: str = "past"

    def __post_init__(self):
        if self.num_anchors < 1 or self.num_proposals < 1 or self.num_draft_layers < 1:
            raise ValueError("Anchor, proposal, and layer counts must be positive")
        if not self.target_layer_ids or len(set(self.target_layer_ids)) != len(self.target_layer_ids):
            raise ValueError("target_layer_ids must be a nonempty list of distinct decoder indices")
        if self.attention not in {"sdpa", "eager"}:
            raise ValueError("This reference backend supports sdpa and eager attention")
        if self.format_version not in {1, 2, 3, 4}:
            raise ValueError("Unsupported four-anchor checkpoint version")
        if self.anchor_layout not in {"past", "bonus_in_four", "target_four", "bonus_addon"}:
            raise ValueError("Unknown anchor_layout")
        if self.anchor_layout == "bonus_in_four":
            if (self.format_version, self.num_anchors, self.num_proposals) != (2, 4, 15):
                raise ValueError("bonus_in_four requires format_version=2, four anchors and 15 proposals")
        elif self.anchor_layout == "target_four":
            if (self.format_version, self.num_anchors, self.num_proposals) != (3, 4, 15):
                raise ValueError("target_four requires format_version=3, four target anchors and 15 proposals")
        elif self.anchor_layout == "bonus_addon":
            if (self.format_version, self.num_anchors, self.num_proposals) != (4, 4, 15):
                raise ValueError("bonus_addon requires format_version=4, four anchors and 15 proposals")
        elif self.format_version != 1:
            raise ValueError("Legacy past-anchor checkpoints use format_version=1")


@dataclass
class Anchors:
    ids: torch.Tensor             # [1, A]
    positions: torch.Tensor       # [1, A], original absolute positions
    target_features: torch.Tensor # [1, A, F]
    draft_features: torch.Tensor  # [1, A, H]
    source: torch.Tensor          # [1, A]: 0=padding, 1=target, 2=draft, 3=embedding-only bonus


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
        if self.cfg.anchor_layout == "bonus_addon":
            raise ValueError("bonus_addon requires the packed training path or bonus-addon vLLM proposer")
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

    def save(self, path, state_dict=None):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        metadata = {"architecture": "FourAnchorDFlash", "draft_config": asdict(self.cfg),
                    "target_signature": self.target_signature}
        (path / "four_anchor_config.json").write_text(json.dumps(metadata, indent=2) + "\n")
        source = self.state_dict() if state_dict is None else state_dict
        state = {k: v.detach().cpu().contiguous() for k, v in source.items()
                 if not k.startswith(("embed_tokens.", "lm_head."))}
        save_file(state, str(path / "model.safetensors"))

    def packed_hidden(self, features, context_positions, document_ids, anchors,
                      starts, context_limits, block_documents, memory=None, include_bonus=False,
                      bonus_ids=None, return_all=False):
        """Independent blocks sharing packed context memory; no cross-document KV.

        starts/context_limits are *document-relative* positions. Pass two moves
        starts, but keeps the original context_limits: teacher states of the
        retained proposals must never leak into either its queries or memory.
        Returns [blocks, proposals, H], before the frozen vocabulary projection.
        bonus_ids adds ONE embedding-only query at starts-1; it is not a fifth
        target anchor. return_all retains all query states for draft-only redrafting.
        """
        a, m = self.cfg.num_anchors, self.cfg.num_proposals
        g = starts.numel()
        device = features.device
        masked = torch.full((g, m), self.cfg.mask_token_id, device=device, dtype=torch.long)
        if bonus_ids is not None:
            if self.cfg.anchor_layout != "bonus_addon" or include_bonus or bonus_ids.shape != (g, 1):
                raise ValueError("The addon layout needs one bonus ID per block")
            if not torch.equal(context_limits, starts - 1):
                raise ValueError("Bonus and future target states must remain outside context")
        slots = masked if bonus_ids is None else torch.cat((bonus_ids, masked), -1)
        q = a + slots.shape[1]
        x = self.embed_tokens(torch.cat([anchors.ids, slots], -1))
        tf = self.hidden_norm(self.fc(anchors.target_features))
        df = self.draft_anchor_norm(self.draft_anchor_proj(anchors.draft_features))
        feature = tf * (anchors.source == 1)[..., None] + df * (anchors.source == 2)[..., None]
        x = (x + torch.cat([feature, torch.zeros_like(x[:, a:])], 1)).reshape(1, g * q, -1)
        context = self.hidden_norm(self.fc(features)) if memory is None else None
        slot_positions = starts[:, None] + torch.arange(m, device=device)
        if bonus_ids is not None:
            slot_positions = torch.cat((starts[:, None] - 1, slot_positions), -1)
        query_positions = torch.cat([anchors.positions, slot_positions], -1)
        positions = torch.cat([context_positions[None], query_positions.reshape(1, -1)], -1)
        rope = self.rotary_emb(x, positions)
        ctx_ok = ((document_ids[None] == block_documents[:, None]) &
                  (context_positions[None] < context_limits[:, None]))
        owners = torch.arange(g, device=device).repeat_interleave(q)
        key_valid = torch.cat([anchors.source != 0, torch.ones_like(slots, dtype=torch.bool)], -1).flatten()
        query_ok = (owners[:, None] == owners[None]) & key_valid[None]
        allowed = torch.cat([ctx_ok.repeat_interleave(q, 0), query_ok], -1)
        mask = torch.zeros((1, 1, *allowed.shape), device=device, dtype=x.dtype)
        mask.masked_fill_(~allowed[None, None], torch.finfo(x.dtype).min)
        for index, layer in enumerate(self.layers):
            if memory is None:
                def run(h, ctx, layer=layer):
                    return layer(hidden_states=h, target_hidden=ctx, attention_mask=mask,
                                 position_ids=positions, position_embeddings=rope, use_cache=False)
                args = (x, context)
            else:
                def run(h, k, v, layer=layer):
                    return self._cached_layer(layer, h, k, v, mask, rope)
                args = (x, *memory[index])
            if self.gradient_checkpointing and self.training:
                from torch.utils.checkpoint import checkpoint
                x = checkpoint(run, *args, use_reentrant=False)
            else:
                x = run(*args)
        # The last anchor's first-pass state conditions the next pass. It is
        # NOT a target state and is never supervised as a speculative token.
        hidden = self.norm(x.reshape(g, q, -1))
        if return_all:
            return hidden
        return hidden[:, a + int(bonus_ids is not None) - int(include_bonus):]

    def packed_memory(self, features, positions):
        """Project shared context and each layer's K/V once, with gradients intact."""
        from speculators.models.dflash.model_definitions import _rotate_half
        context = self.hidden_norm(self.fc(features))
        cos, sin = self.rotary_emb(context, positions[None])
        cos, sin = cos[:, None], sin[:, None]
        memory = []
        for layer in self.layers:
            attention = layer.self_attn
            shape = (*context.shape[:2], -1, attention.head_dim)
            k = attention.k_norm(attention.k_proj(context).view(shape)).transpose(1, 2)
            k = k * cos + _rotate_half(k) * sin
            v = attention.v_proj(context).view(shape).transpose(1, 2)
            memory.append((k, v))
        return memory

    @staticmethod
    def _cached_layer(layer, hidden, context_k, context_v, mask, rope):
        """The pinned DFlash layer equations, reusing only its context projections.

        No parameters or state-dict keys are added. The normal inference layer
        remains unchanged. Context K/V retain their graph for shared gradients.
        """
        from speculators.models.dflash.model_definitions import (
            ALL_ATTENTION_FUNCTIONS, eager_attention_forward, apply_rotary_pos_emb,
        )
        residual = hidden
        x = layer.input_layernorm(hidden)
        attention = layer.self_attn
        b, length = x.shape[:2]
        shape = (b, length, -1, attention.head_dim)
        q = attention.q_norm(attention.q_proj(x).view(shape)).transpose(1, 2)
        k = attention.k_norm(attention.k_proj(x).view(shape)).transpose(1, 2)
        v = attention.v_proj(x).view(shape).transpose(1, 2)
        cos, sin = rope
        q, k = apply_rotary_pos_emb(q, k, cos[:, -length:], sin[:, -length:])
        k, v = torch.cat([context_k, k], -2), torch.cat([context_v, v], -2)
        implementation = attention.config._attn_implementation
        fn = eager_attention_forward if implementation in {None, "eager"} else ALL_ATTENTION_FUNCTIONS[implementation]
        value = fn(attention, q, k, v, mask, dropout=0., scaling=attention.scaling,
                   sliding_window=attention.sliding_window)[0]
        hidden = residual + attention.o_proj(value.reshape(b, length, -1))
        return hidden + layer.mlp(layer.post_attention_layernorm(hidden))

    @classmethod
    def load(cls, path, target, trainable_dtype=None):
        path = Path(path)
        metadata = json.loads((path / "four_anchor_config.json").read_text())
        if metadata["architecture"] != "FourAnchorDFlash":
            raise ValueError("Not a FourAnchorDFlash checkpoint")
        if metadata["target_signature"] != cls.signature(target.config):
            raise ValueError("Checkpoint and target architectures differ")
        model = cls(DraftConfig(**metadata["draft_config"]), target)
        # Resume must not round saved FP32 master weights through the frozen
        # target's BF16 dtype before the optimizer is restored.
        if trainable_dtype is not None:
            for parameter in model.parameters():
                if parameter.requires_grad:
                    parameter.data = parameter.data.to(trainable_dtype)
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
