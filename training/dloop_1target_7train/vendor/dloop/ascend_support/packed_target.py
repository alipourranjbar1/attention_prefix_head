"""Skip unused teacher vocabulary logits; retain identical Qwen3 hidden states."""
from types import MethodType

import torch


@torch.no_grad()
def _hidden_only(self, input_ids, attention_mask, loss_mask, position_offsets=None):
    from specforge.inference.target_engine.dflash_target_model import DFlashTargetOutput
    position_ids = None
    if position_offsets is not None:
        position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
        position_ids = position_ids + position_offsets.view(-1, 1)
    outputs = self.model.model(input_ids=input_ids, attention_mask=attention_mask,
                               position_ids=position_ids, output_hidden_states=True,
                               use_cache=False, return_dict=True)
    hidden_states = (torch.cat([outputs.hidden_states[i+1] for i in self.capture_layer_ids], dim=-1)
                     if self.capture_layer_ids is not None else outputs.hidden_states[-1])
    return DFlashTargetOutput(hidden_states=hidden_states, input_ids=input_ids,
        attention_mask=attention_mask, loss_mask=loss_mask,
        last_hidden_states=outputs.hidden_states[-1] if self.return_last_hidden_states else None)


def enable_hidden_only_target(engine):
    from transformers import Qwen3ForCausalLM
    from specforge.inference.target_engine.dflash_target_model import HFDFlashTargetEngine
    if type(engine) is not HFDFlashTargetEngine or type(engine.model) is not Qwen3ForCausalLM:
        raise ValueError("Packed hidden-only target supports the HF Qwen3 target only")
    engine.generate_dflash_data = MethodType(_hidden_only, engine)
    return engine
