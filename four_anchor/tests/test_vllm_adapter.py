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


@pytest.mark.parametrize("accepted", range(16))
def test_propose_uses_only_past_states_and_never_calls_target(adapter, monkeypatch, accepted):
    runner = adapter.FourAnchorNPUModelRunner.__new__(adapter.FourAnchorNPUModelRunner)
    history = FeatureHistory(128, 2, "cpu", torch.float32)
    history.update("request", 0, torch.arange(4), torch.zeros(4, 2))
    observed = {}
    def update(req, start, features):
        observed["cache"] = (req, start, features.clone())
    def propose(ids, bonus, mode):
        observed["ids"] = ids.clone()
        observed["bonus"] = bonus.clone()
        return torch.arange(15)[None], torch.tensor([3])
    inference = NS(cache=NS(update=update), propose=propose,
                   profiler=NS(region=lambda _: nullcontext(), finish=lambda: None))
    runner.drafter = NS(history=history, layer_ids=[0, 1], inference=inference,
                        settings=NS(num_proposals=15, mode="two_pass", trace_steps=False))
    runner.input_batch = NS(num_reqs=1, req_ids=["request"])
    runner.input_ids = NS(gpu=torch.arange(40, 56))
    runner.max_model_len, runner.device = 128, "cpu"
    def forbidden(*args, **kwargs):
        raise AssertionError("No extra target forward or device position readback")
    runner.model = forbidden
    runner._get_positions = forbidden
    output = runner.propose_draft_token_ids(
        [list(range(41, 41 + accepted)) + [60]], NS(all_greedy=True), None,
        NS(num_draft_tokens=[15]), NS(num_computed_tokens_cpu=torch.tensor([4])), None, 16, None,
        [torch.arange(16).float()[:, None], (torch.arange(16) + 100).float()[:, None]],
    )
    assert output == [list(range(15))]
    assert observed["bonus"].tolist() == [[60]]
    assert observed["ids"].tolist() == [list(range(4)) + list(range(40, 41 + accepted))]
    assert observed["cache"][1] == 4
    assert observed["cache"][2][0, :, 0].tolist() == list(range(accepted + 1))
    assert not hasattr(runner, "_materialize_bonus")


def test_custom_class_constructor_accepts_upstream_config_only(adapter, components, tmp_path):
    _target, draft, _runner = components
    draft.save(tmp_path)
    config = NS(model_config=NS(hf_config=NS(four_anchor_serving={"checkpoint": str(tmp_path)})))
    initial = adapter.FourAnchorProposer(config)
    assert initial.device is None and initial.model is None
