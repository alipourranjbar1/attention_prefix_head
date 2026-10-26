"""ReTrace conditions DFlash; the prefix head selects the actual verified path."""
from typing import ClassVar

import torch

from speculators.model import SpeculatorModel
from speculators.models.dflash_prefix.core import DFlashPrefixDraftModel
from speculators.models.dflash_prefix.selector_v2 import make_prefix_head, unary_topk
from speculators.models.retrace.core import DraftBlock, ReTraceDraftModel

from .config import ReTracePrefixSpeculatorConfig


@SpeculatorModel.register("retrace_prefix")
class ReTracePrefixDraftModel(ReTraceDraftModel):
    config_class: ClassVar[type[ReTracePrefixSpeculatorConfig]] = ReTracePrefixSpeculatorConfig

    def __init__(self, config):
        # ReTrace constructs dense additive block masks. HF does not serialize
        # the private attention backend; never fall back to FlexAttention here.
        if config.transformer_layer_config._attn_implementation is None:
            config.transformer_layer_config._attn_implementation = "sdpa"
        super().__init__(config)
        with torch.random.fork_rng(devices=[]):
            self.prefix_head = make_prefix_head(
                self.draft_vocab_size, self.hidden_size, config.prefix_rank,
                config.prefix_top_k, self.block_size - 1, config.prefix_gate_init,
                kind=config.prefix_selector_kind, heads=config.prefix_attention_heads,
                layers=config.prefix_attention_layers, history_scale=config.prefix_history_scale,
                compiled_logit_cap=config.prefix_compiled_logit_cap,
            )
        if hasattr(self.prefix_head, "set_history_trainable"):
            self.prefix_head.set_history_trainable(config.prefix_history_scale > 0)

    _prefix_kwargs = DFlashPrefixDraftModel._prefix_kwargs

    def save_pretrained(self, save_directory, **kwargs):
        result = super().save_pretrained(save_directory, **kwargs)
        self.config.architectures = ["ReTracePrefixDraftModel"]
        self.config.save_pretrained(save_directory)
        return result

    def make_block(self, hidden, logits, anchors):
        # A no-grad greedy walk must not populate the autocast weight cache:
        # the differentiable teacher-prefix pass uses these same head weights.
        device_type = hidden.device.type
        with torch.no_grad(), torch.autocast(
            device_type=device_type, dtype=torch.get_autocast_dtype(device_type),
            enabled=torch.is_autocast_enabled(device_type), cache_enabled=False,
        ):
            scores, candidates = unary_topk(logits, self.config.prefix_top_k)
            # Match the source checkpoint's inference calibration.
            old_scale = self.prefix_head.correction_scale
            try:
                self.prefix_head.correction_scale = old_scale * self.config.prefix_inference_gate_scale
                tokens = self.prefix_head.greedy(
                    hidden, candidates, scores, anchors, **self._prefix_kwargs()
                )
            finally:
                self.prefix_head.correction_scale = old_scale
        return DraftBlock(hidden, logits, tokens)
