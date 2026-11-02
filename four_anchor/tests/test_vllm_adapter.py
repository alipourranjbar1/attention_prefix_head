"""Exercise the adapter's orchestration on CPU with a minimal vLLM boundary.

These are interface/position tests, not evidence of working Ascend kernels.
"""
from contextlib import nullcontext
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS

import pytest
import torch

from four_anchor.vllm_support import FeatureHistory


@pytest.fixture
def adapter(monkeypatch):
    modules = {
        "vllm.config": {"CUDAGraphMode": NS(NONE="none")},
        "vllm.v1.worker.workspace": {"init_workspace_manager": lambda *_: None},
        "vllm_ascend.ascend_forward_context": {"set_ascend_forward_context": lambda *a, **k: nullcontext()},
        "vllm_ascend.attention.attention_v1": {"AscendAttentionState": NS(DecodeOnly="decode")},
        "vllm_ascend.attention.utils": {"AscendCommonAttentionMetadata": NS},
        "vllm_ascend.worker.model_runner_v1": {"NPUModelRunner": type("NPUModelRunner", (), {})},
        "vllm_ascend.worker.worker": {"NPUWorker": type("NPUWorker", (), {})},
    }
    for name, attrs in modules.items():
        module = ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
    spec = importlib.util.spec_from_file_location("four_anchor._adapter_test",
                                                 Path(__file__).parents[1] / "vllm_ascend.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("accepted", [0, 1, 7, 15])
def test_native_propose_excludes_rejected_states_and_materializes_bonus(adapter, monkeypatch, accepted):
    runner = adapter.FourAnchorNPUModelRunner.__new__(adapter.FourAnchorNPUModelRunner)
    history = FeatureHistory(128, 2, "cpu", torch.float32)
    history.update("request", 0, torch.arange(5), torch.zeros(5, 2))
    runner.drafter = NS(history=history, layer_ids=[0, 1], model=object(),
                        settings=NS(num_proposals=15, mode="two_pass"))
    runner.input_batch = NS(num_reqs=1, req_ids=["request"])
    runner.input_ids = NS(gpu=torch.arange(40, 56))
    runner._get_positions = lambda _: torch.arange(4, 20)
    runner.max_model_len = 128
    observed = {}

    def materialize(token, position, common):
        observed["bonus"] = (token, position)
        return torch.tensor([token]), torch.tensor([[900., 901.]])

    def proposal(model, ids, features, mode):
        observed["ids"] = ids.clone()
        observed["features"] = features.clone()
        return torch.arange(15)[None], 3

    runner._materialize_bonus = materialize
    monkeypatch.setattr(adapter, "fixed_window_proposal", proposal)
    output = runner.propose_draft_token_ids(
        [list(range(41, 41 + accepted)) + [60]], NS(all_greedy=True), None,
        NS(num_draft_tokens=[15]), object(), None, 16, None,
        [torch.arange(16).float()[:, None], (torch.arange(16) + 100).float()[:, None]],
    )
    assert output == [list(range(15))]
    assert observed["bonus"] == (60, 5 + accepted)
    assert observed["ids"].tolist() == [list(range(4)) + list(range(40, 41 + accepted)) + [60]]
    assert observed["features"].shape[1] == 6 + accepted
    assert observed["features"][0, -1].tolist() == [900., 901.]
    assert observed["features"][0, 4:-1, 0].tolist() == list(range(accepted + 1))


def test_bonus_uses_allocated_slot_and_shortened_kv_length(adapter):
    runner = adapter.FourAnchorNPUModelRunner.__new__(adapter.FourAnchorNPUModelRunner)
    calls = []

    class Builder:
        def build(self, **kwargs):
            calls.append(kwargs["common_attn_metadata"])
            return object()

    runner.attn_groups = [[NS(get_metadata_builder=lambda _: Builder(), layer_names=["layer0"])]]
    runner.input_batch = NS(block_table=[NS(num_blocks_per_row=[2], block_size=4)])
    runner.device, runner.vllm_config = "cpu", object()
    runner.drafter = NS(layer_ids=[0, 1])
    runner.model = lambda **kwargs: (None, [torch.ones(1, 2), torch.full((1, 2), 2.)])
    common = NS(block_table_tensor=torch.tensor([[10, 20]], dtype=torch.int32))
    ids, features = runner._materialize_bonus(51, 5, common)
    assert ids.tolist() == [51]
    assert features.tolist() == [[1., 1., 2., 2.]]
    cm = calls[0]
    assert cm.slot_mapping.tolist() == [81]  # page 20, offset 1
    assert cm.seq_lens.tolist() == [6]       # rejected positions 6+ are invisible
    assert cm.query_start_loc.tolist() == [0, 1]
    assert cm.positions.tolist() == [5]
    assert cm.causal
    assert runner._materialize_bonus(52, 8, common) is None
    assert len(calls) == 1  # never writes into an unallocated page


def test_custom_class_constructor_accepts_upstream_config_only(adapter, components, tmp_path):
    _target, draft, _runner = components
    draft.save(tmp_path)
    config = NS(model_config=NS(hf_config=NS(four_anchor_serving={"checkpoint": str(tmp_path)})))
    initial = adapter.FourAnchorProposer(config)
    assert initial.device is None and initial.model is None
