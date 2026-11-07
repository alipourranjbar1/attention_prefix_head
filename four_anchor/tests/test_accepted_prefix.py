"""CPU contracts, including real tiny-Qwen greedy verification (not NPU timing)."""
from dataclasses import replace
from contextlib import nullcontext
from types import SimpleNamespace as NS, ModuleType
from pathlib import Path
import json
import sys

import pytest
import torch

from four_anchor.inference import AcceptedPrefixDraft
from four_anchor.model import make_anchors
from four_anchor.vllm_support import FeatureHistory, ServingSettings
from four_anchor.tests.test_vllm_adapter import adapter
from four_anchor.tests.test_cached_benchmarks import suite, tiny_model, fixture_sources, restore_prepare_environment


def target19(components):
    _, draft, _ = components
    draft.cfg = replace(draft.cfg, num_proposals=15, anchor_layout="target_four", format_version=3)
    return draft.eval()


@pytest.mark.parametrize("keep", range(1, 15))
@pytest.mark.parametrize("length", [1, 4, 10])
@torch.inference_mode()
def test_both_passes_match_training_equations_and_return_exact_future_window(components, monkeypatch, keep, length):
    draft = target19(components)
    _, _, runner = components
    ids = torch.arange(1, length + 1)[None]
    state = runner.prefill(ids)
    bonus = state.next_logits.argmax(-1, keepdim=True)
    first = draft(state.features, make_anchors(ids, state.features, 4, draft.hidden_size), length)
    prior = NS(tokens=torch.cat((bonus, first.tokens[:, 1:]), -1), hidden=first.hidden)
    a2 = make_anchors(ids, state.features, 4, draft.hidden_size, prior, keep)
    expected_sources = [0 if p < 0 else 1 if p < length else 2
                        for p in range(length + keep - 4, length + keep)]
    assert a2.source.tolist() == [expected_sources]
    second = draft(state.features, a2, length + keep)
    fast = AcceptedPrefixDraft(draft, 128)
    fast.cache.update("x", 0, state.features)
    observed = []
    original = fast.pass_hidden
    def observe(*args):
        hidden = original(*args)
        observed.append((args[2].item(), args[-1].clone(), hidden.clone()))
        return hidden
    monkeypatch.setattr(fast, "pass_hidden", observe)
    def scores(logits):
        assert logits.shape == (1, 14, 64)  # pending bonus is already known
        out = torch.zeros(1, 14)
        out[0, keep-1] = 1
        return out
    monkeypatch.setattr("four_anchor.inference.boundary_scores", scores)
    calls_before = runner.calls
    tokens, boundary = fast.propose(ids, bonus)
    assert runner.calls == calls_before  # draft passes cannot invoke the target
    assert boundary.item() == keep
    assert fast.requires_materialized_bonus is False
    assert len(observed) == 2
    assert [p[0] for p in observed] == [length, length + keep]
    assert all(p[1].shape == (1, 15) and (p[1] == draft.cfg.mask_token_id).all() for p in observed)
    assert all(4 + p[1].shape[1] == 19 for p in observed)
    torch.testing.assert_close(observed[0][2], first.hidden, atol=3e-6, rtol=3e-5)
    torch.testing.assert_close(observed[1][2], second.hidden, atol=3e-6, rtol=3e-5)
    # Original slot 0 is the pending bonus; n+15 comes from round two.
    expected = torch.cat((prior.tokens[:, 1:keep], second.tokens[:, :16-keep]), -1)
    assert tokens.shape == (1, 15) and tokens.tolist() == expected.tolist()
    assert fast.cache.length == length  # no insertion of pending/rejected IDs


@pytest.mark.parametrize("budget", [1, 2, 17, 37])
@torch.inference_mode()
def test_one_target_forward_per_cycle_greedy_parity_and_output_budget(components, budget):
    draft = target19(components)
    _, _, runner = components
    prompt = torch.tensor([[1, 2, 3, 4]])
    state = runner.prefill(prompt)
    fast = AcceptedPrefixDraft(draft, 256)
    fast.cache.update("x", 0, state.features)
    bonus = state.next_logits.argmax(-1, keepdim=True)
    emitted = bonus[0].tolist()
    calls_before, cycles = runner.calls, 0
    while len(emitted) < budget:
        n = state.ids.shape[1]
        candidates, _ = fast.propose(state.ids, bonus)
        inputs = torch.cat((bonus, candidates), -1)
        out, features = runner.forward(inputs, state.cache, all_logits=True)
        predictions = out.logits[:, :-1].argmax(-1)
        bad = (predictions != candidates)[0].nonzero()
        accepted = int(bad[0]) if bad.numel() else 15
        bonus = out.logits[:, accepted].argmax(-1, keepdim=True)
        keep = accepted + 1
        out.past_key_values.crop(n + keep)
        fast.cache.update("x", n, features[:, :keep])
        state.ids = torch.cat((state.ids, inputs[:, :keep]), -1)
        state.cache = out.past_key_values
        emitted += candidates[0, :accepted].tolist() + bonus[0].tolist()
        cycles += 1
    assert runner.calls - calls_before == cycles
    ar = runner.prefill(prompt)
    expected = []
    for _ in range(budget):
        token = ar.next_logits.argmax(-1, keepdim=True)
        expected += token[0].tolist()
        ar = runner.append(ar, token)
    assert expected == emitted[:budget]
    # Native stop handling truncates the same stream at the first EOS too.
    eos = expected[-1]
    stop = expected.index(eos) + 1
    assert emitted[:stop] == expected[:stop]


@pytest.mark.parametrize("accepted", range(16))
def test_adapter_discards_rejected_suffix_and_new_bonus_without_materializing(adapter, monkeypatch, accepted):
    runner = adapter.FourAnchorNPUModelRunner.__new__(adapter.FourAnchorNPUModelRunner)
    history = FeatureHistory(128, 2, "cpu", torch.float32)
    history.update("request", 0, torch.arange(20), torch.zeros(20, 2))
    observed = {}
    def update(req, start, features):
        observed["features"] = features.clone()
    def propose(ids, bonus, mode):
        assert bonus.tolist() == [[60]]
        assert ids.tolist() == [list(range(4)) + list(range(40, 41+accepted))]
        assert ids.shape[1] == 5 + accepted
        assert history.length == 5 + accepted
        return torch.arange(15)[None], torch.tensor([4])
    def forbidden(*args, **kwargs):
        raise AssertionError("Extra target forward is forbidden")
    inference = NS(requires_materialized_bonus=False, cache=NS(update=update), propose=propose,
                   profiler=NS(region=lambda _: nullcontext(), finish=lambda: None))
    runner.drafter = NS(history=history, layer_ids=[0, 1], inference=inference,
        counters={"extra_target_forwards": 0, "bonus_page_fallbacks": 0},
        settings=NS(num_proposals=15, mode="two_pass", trace_steps=False))
    runner.input_batch = NS(num_reqs=1, req_ids=["request"])
    runner.input_ids = NS(gpu=torch.arange(40, 56))
    runner.max_model_len, runner.device, runner.model = 128, "cpu", forbidden
    monkeypatch.setattr(adapter, "materialize_bonus", forbidden)
    output = runner.propose_draft_token_ids(
        [list(range(41, 41+accepted)) + [60]], NS(all_greedy=True), None,
        NS(num_draft_tokens=[15]), NS(num_computed_tokens_cpu=torch.tensor([4])), None, 16, None,
        [torch.arange(16).float()[:, None], (torch.arange(16)+100).float()[:, None]])
    assert output == [list(range(15))]
    assert observed["features"][0, :, 0].tolist() == list(range(accepted+1))
    assert runner.drafter.counters == {"extra_target_forwards": 0, "bonus_page_fallbacks": 0}


@torch.inference_mode()
def test_policy_is_explicit_and_checkpoint_is_not_rewritten(components, tmp_path):
    from four_anchor.vllm_entry import engine_options
    draft = target19(components)
    draft.save(tmp_path)
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir() if p.is_file()}
    assert ServingSettings(str(tmp_path)).inference_version() == "target_four_materialized_v3"
    settings = ServingSettings(str(tmp_path), anchor_policy="accepted_prefix")
    assert settings.inference_version() == "accepted_prefix19_v1"
    args = NS(target="tiny", checkpoint=str(tmp_path), mode="two_pass", proposals=15,
              anchor_policy="accepted_prefix", max_model_len=128, memory_fraction=.8)
    opts = engine_options(args)
    assert opts["hf_overrides"]["four_anchor_serving"]["anchor_policy"] == "accepted_prefix"
    assert opts["speculative_config"]["num_speculative_tokens"] == 15
    fast = AcceptedPrefixDraft(draft, 64)
    ids = (torch.arange(34)[None] % 60) + 1
    fast.cache.update("x", 0, torch.randn(1, 34, 64))
    assert fast.propose(ids, torch.tensor([[5]]))[0].shape == (1, 15)
    with pytest.raises(ValueError, match="two_pass"):
        fast.propose(ids, torch.tensor([[5]]), "one_pass")
    with pytest.raises(ValueError, match="pending bonus"):
        fast.propose(ids)
    with pytest.raises(ValueError, match="two_pass"):
        ServingSettings(str(tmp_path), mode="one_pass", anchor_policy="accepted_prefix")
    with pytest.raises(ValueError, match="15 proposals"):
        ServingSettings(str(tmp_path), num_proposals=14, anchor_policy="accepted_prefix").metadata()
    assert before == {p.name: p.read_bytes() for p in tmp_path.iterdir() if p.is_file()}


@pytest.mark.parametrize("extra", [0, 1])
def test_evaluator_reads_actual_counters_and_rejects_extra_calls(components, tmp_path, monkeypatch, extra):
    from four_anchor.vllm_entry import evaluate
    draft = target19(components)
    draft.save(tmp_path / "checkpoint")
    module = ModuleType("vllm")
    calls = []
    class FakeLLM:
        def __init__(self, **kwargs):
            self.extra = 0
        def collective_rpc(self, method):
            calls.append(method)
            return [{"extra_target_forwards": self.extra, "bonus_page_fallbacks": 0}]
        def generate(self, *args, **kwargs):
            self.extra += extra
            metrics = NS(to_dict=lambda: {"num_spec_tokens": 15, "per_step_accepted": [1, 0],
                "num_spec_steps": 2, "num_accepted_draft_tokens": 1})
            return [NS(outputs=[NS(token_ids=[4, 5, 6], spec_decode_metrics=metrics)])]
    module.LLM, module.SamplingParams = FakeLLM, lambda **kwargs: kwargs
    monkeypatch.setitem(sys.modules, "vllm", module)
    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", lambda *a, **kw: None)
    monkeypatch.setattr("four_anchor.data.load_records", lambda _: [{"input_ids": [1, 2, 3]}])
    args = NS(checkpoint=str(tmp_path / "checkpoint"), proposals=15, target="tiny", data="unused",
              output=str(tmp_path / "out"), limit=1, max_new_tokens=32, max_model_len=128,
              baseline=None, warmup=0, mode="two_pass", anchor_policy="accepted_prefix")
    if extra:
        with pytest.raises(RuntimeError, match="extra target forward"):
            evaluate(args, {})
        assert not (tmp_path / "out/summary.json").exists()
    else:
        evaluate(args, {})
        result = json.loads((tmp_path / "out/summary.json").read_text())
        assert result["extra_target_forwards_measured"] and result["extra_target_forwards"] == 0
        assert result["query_positions"] == 19 and result["speculative_tokens"] == 15
        assert result["acceptance_length_including_bonus"] == 1.5
    assert calls == ["four_anchor_counters"] * 2


def test_all_eight_benchmarks_reuse_frozen_prompts_in_separate_output(suite, tmp_path, monkeypatch):
    model = tmp_path / "model"
    tiny_model(model)
    cache = fixture_sources(suite, tmp_path)
    repo = Path(__file__).parents[2]
    monkeypatch.setattr(suite, "cache_roots", lambda _: [cache])
    common = ["--root", str(repo), "--code-root", str(repo), "--inference-layout", "accepted_prefix",
              "--model", str(model), "--prepare-only", "--limit", "2"]
    frozen = tmp_path / "frozen"
    assert suite.main(common + ["--output", str(frozen)]) == 0
    before = {str(p.relative_to(frozen)): p.read_bytes() for p in frozen.rglob("*") if p.is_file()}
    output = tmp_path / "new_eval"
    assert suite.main(common + ["--output", str(output), "--prompt-plan", str(frozen / "plan.json")]) == 0
    plan = json.loads((output / "plan.json").read_text())
    assert plan["method"] == "accepted_prefix19_v1" and plan["query_positions"] == 19
    assert plan["bonus_target_forward_required"] is False and len(plan["benchmarks"]) == 8
    for name in plan["benchmarks"]:
        assert (output / name / "prompts.jsonl").read_bytes() == (frozen / name / "prompts.jsonl").read_bytes()
    assert before == {str(p.relative_to(frozen)): p.read_bytes() for p in frozen.rglob("*") if p.is_file()}
