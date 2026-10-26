"""Autoregressive DFlash backbone: identical parameters and prefix feature fusion."""
from typing import ClassVar, Literal

import torch
from pydantic import Field, model_validator
from transformers.cache_utils import DynamicCache

from speculators import SpeculatorModelConfig
from speculators.model import SpeculatorModel
from speculators.models.dflash.config import DFlashSpeculatorConfig
from speculators.models.dflash.core import DFlashDraftModel
from speculators.models.dflash.attention import create_anchor_block_mask_mod
from speculators.models.dflash.utils import get_base_indices_for_anchored_blocks


@SpeculatorModelConfig.register("dflash_ar")
class DFlashARSpeculatorConfig(DFlashSpeculatorConfig):
    has_no_defaults_at_init: ClassVar[bool] = True
    speculators_model_type: Literal["dflash_ar"] = "dflash_ar"
    architectures: list[str] = Field(default_factory=lambda: ["DFlashARDraftModel"])
    sample_from_anchor: bool = True

    @model_validator(mode="after")
    def validate_ar(self):
        if not self.sample_from_anchor:
            raise ValueError("AR predicts the next token from every input slot, including the anchor")
        if self.block_size < 1:
            raise ValueError("AR block_size is the positive number of new proposals")
        if self.draft_vocab_size != self.transformer_layer_config.vocab_size:
            raise ValueError("This matched AR experiment requires the full target vocabulary")
        return self


@SpeculatorModel.register("dflash_ar")
class DFlashARDraftModel(DFlashDraftModel):
    config_class: ClassVar[type[DFlashARSpeculatorConfig]] = DFlashARSpeculatorConfig

    @classmethod
    def from_training_args(cls, verifier_config, t2d=None, d2t=None, **kwargs):
        kwargs = dict(kwargs, sample_from_anchor=True)
        cfg = DFlashARSpeculatorConfig(**cls._build_base_config_kwargs(
            "dflash_ar", verifier_config, **kwargs))
        model = cls(cfg)
        model.load_vocab_mappings(t2d, d2t)
        model.load_verifier_weights()
        return model

    @torch.compiler.disable
    def _create_attention_mask(self, document_ids, total_seq_len, anchor_positions,
                               device, sliding_window=None,
                               sliding_window_non_causal=False):
        base, q_len, kv_len = create_anchor_block_mask_mod(
            document_ids.squeeze(0).to(device), total_seq_len, anchor_positions,
            self.block_size, sliding_window, False)

        def causal(b, h, q, kv):
            # Preserve exactly the baseline's target-prefix visibility. Make
            # the synthetic token block causal even for full-attention layers.
            return base(b, h, q, kv) & ((kv < total_seq_len) | (kv <= total_seq_len + q))

        return self._create_mask_fn(causal, B=None, H=None, Q_LEN=q_len,
                                    KV_LEN=kv_len, device=device)

    def _backbone_forward(self, hidden_states, input_ids, loss_mask,
                          verifier_last_hidden_states, document_ids,
                          position_ids=None, **kwargs):
        device = hidden_states.device
        length = input_ids.shape[1]
        if input_ids.shape[0] != 1 or length <= self.block_size:
            raise ValueError("Expected one packed sequence longer than the AR block")
        if position_ids is None:
            position_ids = torch.arange(length, device=device).unsqueeze(0)
        full, sliding, anchors, valid = self._build_attention_mask(
            loss_mask, kwargs.pop("max_anchors", 512), document_ids, device)
        indices = get_base_indices_for_anchored_blocks(anchors, self.block_size)
        # x[a:a+K] -> target distributions p(x[a+1:a+K+1]). No mask tokens.
        query = self.embed_tokens(input_ids[:, indices])
        context = self.hidden_norm(self.fc(hidden_states))
        positions = torch.cat((position_ids, position_ids[:, indices]), dim=1)
        rotary = self.rotary_emb(hidden_states, positions)
        with torch.no_grad():
            targets = self.verifier_lm_head(self.verifier_norm(
                verifier_last_hidden_states[:, indices]))
        for i, layer in enumerate(self.layers):
            query = layer(hidden_states=query, target_hidden=context,
                          attention_mask=sliding if i in self.sliding_window_indices else full,
                          position_ids=positions, use_cache=False,
                          position_embeddings=rotary, **kwargs)
        hidden = self.norm(query)
        logits = self.lm_head(hidden)
        # Loss belongs to the NEXT token. Never train across a packed document
        # boundary, including blocks that start before such a boundary.
        anchor_docs = document_ids[:, anchors].repeat_interleave(self.block_size, dim=1)
        covered = ((document_ids[:, indices] == anchor_docs)
                   & (document_ids[:, indices + 1] == anchor_docs)
                   & (anchor_docs >= 0))
        aligned = loss_mask[:, indices + 1] * covered * valid.repeat_interleave(self.block_size)[None]
        return hidden, logits, targets, aligned, indices

    def forward(self, *args, **kwargs):
        result, loss, metrics = super().forward(*args, **kwargs)
        # These predictions use ground-truth previous tokens; do not label them
        # as measured free-running acceptance length.
        for suffix in ("sum", "total"):
            metrics["teacher_forced_eal_" + suffix] = metrics.pop("eal_" + suffix)
        return result, loss, metrics

    @torch.no_grad()
    def propose(self, target_hidden, anchor, num_tokens=None):
        """Single request, full verified prefix excluding anchor; fresh cache per round.

        Reuse identical fc/hidden_norm, embedding, decoder and lm_head tensors.
        No target forward is performed here. Target prefix K/V is inserted once.
        """
        if self.training:
            raise RuntimeError("Call eval() before proposing")
        count = self.block_size if num_tokens is None else num_tokens
        if not 1 <= count <= self.block_size:
            raise ValueError("num_tokens must be in [1, checkpoint block_size]")
        if target_hidden.shape[0] != 1 or anchor.shape != (1, 1):
            raise ValueError("Reference proposer supports one unpadded request")
        if self._attn_impl not in {"sdpa", "eager"}:
            raise ValueError("Reference proposer requires sdpa or eager attention")
        prefix_len = target_hidden.shape[1]
        fused = self.hidden_norm(self.fc(target_hidden))
        cache = DynamicCache()
        current = anchor
        proposals = []
        for step in range(count):
            query = self.embed_tokens(current)
            context = fused if step == 0 else fused[:, :0]
            positions = (torch.arange(prefix_len + 1, device=anchor.device)[None]
                         if step == 0 else
                         torch.tensor([[prefix_len + step]], device=anchor.device))
            rotary = self.rotary_emb(query, positions)
            key_positions = torch.arange(prefix_len + step + 1, device=anchor.device)
            for i, layer in enumerate(self.layers):
                allowed = torch.ones_like(key_positions, dtype=torch.bool)
                if i in self.sliding_window_indices:
                    # Match training: window on target prefix only; all earlier
                    # draft tokens remain visible within this anchor's block.
                    allowed &= ((key_positions >= prefix_len) |
                                (key_positions >= prefix_len - self.sliding_window))
                mask = torch.zeros((1, 1, 1, len(key_positions)), device=query.device,
                                   dtype=query.dtype).masked_fill(~allowed[None, None, None], -torch.inf)
                query = layer(hidden_states=query, target_hidden=context,
                              attention_mask=mask, position_ids=positions,
                              past_key_value=cache, use_cache=True,
                              position_embeddings=rotary)
            current = self.lm_head(self.norm(query[:, -1:])).argmax(-1)
            proposals.append(current)
        return torch.cat(proposals, dim=1)
