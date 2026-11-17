import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from four_anchor.model import DraftConfig, FourAnchorDFlash
from four_anchor.runtime import TargetRunner


@pytest.fixture
def components():
    torch.set_num_threads(1)
    torch.manual_seed(1234)
    config = Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64,
                        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                        head_dim=8, max_position_embeddings=512, eos_token_id=None,
                        pad_token_id=0, attention_dropout=0.0)
    config._attn_implementation = "sdpa"
    target = Qwen3ForCausalLM(config).eval()
    draft = FourAnchorDFlash(DraftConfig("tiny", [0, 1], 63, num_draft_layers=2), target)
    runner = TargetRunner(target, [0, 1])
    yield target, draft, runner
    runner.close()
