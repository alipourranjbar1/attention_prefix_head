"""Explicit checkpoint dispatch; never silently drop a trained prefix head."""
import json
from pathlib import Path

from speculators.models.retrace.config import ReTraceSpeculatorConfig
from .config import ReTracePrefixSpeculatorConfig


def load_config(checkpoint):
    kind = json.loads((Path(checkpoint) / "config.json").read_text()).get("speculators_model_type")
    classes = {"retrace": ReTraceSpeculatorConfig, "retrace_prefix": ReTracePrefixSpeculatorConfig}
    if kind not in classes:
        raise ValueError(f"Expected a converted ReTrace checkpoint, got {kind!r}")
    return classes[kind].from_pretrained(checkpoint, local_files_only=True)
