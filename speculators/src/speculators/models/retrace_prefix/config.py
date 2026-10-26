"""Checkpoint identity for ReTrace + the existing DFlash prefix selector."""
from typing import Literal

from pydantic import Field, model_validator

from speculators.config import SpeculatorModelConfig
from speculators.models.dflash_prefix.config import DFlashPrefixSpeculatorConfig
from speculators.models.retrace.config import ReTraceSpeculatorConfig


@SpeculatorModelConfig.register("retrace_prefix")
class ReTracePrefixSpeculatorConfig(ReTraceSpeculatorConfig, DFlashPrefixSpeculatorConfig):
    speculators_model_type: Literal["retrace_prefix"] = "retrace_prefix"
    architectures: list[str] = Field(default_factory=lambda: ["ReTracePrefixDraftModel"])
    distill_top_k: int = Field(default=16, ge=1)
    distill_kl_weight: float = Field(default=0.1, ge=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_combined_training(self):
        if self.distill_top_k > self.draft_vocab_size:
            raise ValueError("distill_top_k exceeds the vocabulary")
        if self.prefix_freeze_backbone:
            raise ValueError("ReTrace prefix retraining requires an unfrozen backbone")
        return self
