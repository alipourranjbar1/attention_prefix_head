from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace as NS

import pytest
import torch

from four_anchor.inference import TargetFourDraft
from four_anchor.model import DraftOutput, make_anchors
from four_anchor.packed_objective import packed_anchors
from four_anchor.vllm_support import FeatureHistory, ServingSettings
from four_anchor.tests.test_packed_training import batch_for
from four_anchor.tests.test_vllm_adapter import adapter


def target19(components):
    _, draft, _ = components
    draft.cfg = replace(draft.cfg, num_proposals=15, anchor_layout="target_four", format_version=3)
    return draft


@pytest.mark.parametrize("n", [1, 2, 4, 8])
def test_four_actual_target_states_and_nineteen_queries_match_training(components, n):
    draft = target19(components)
    batch = batch_for(components)
    starts, docs = torch.tensor([n]), torch.tensor([0])
    a = packed_anchors(batch, starts, starts, docs, draft.hidden_size, 4)
    assert a.source[0, -1] == 1
    torch.testing.assert_close(a.target_features[0, -1], batch["features"][n-1])
    trained = draft.packed_hidden(batch["features"][None], batch["positions"], batch["documents"],
                                  a, starts, starts, docs)
    with torch.inference_mode():
        fast = TargetFourDraft(draft, 128)
        fast.cache.update("x", 0, batch["features"][:n][None])
        actual = fast.pass_hidden(batch["ids"][:n][None], n, starts[0], fast.zero_ids, fast.zero_hidden, fast.masks)
    assert actual.shape == (1, 15, draft.hidden_size)
    assert a.ids.shape[1] + fast.masks.shape[1] == 19
    torch.testing.assert_close(actual, trained, atol=3e-6, rtol=3e-5)
    altered = batch["features"].clone()
    altered[n:] = 100 * torch.randn_like(altered[n:])
    other = draft.packed_hidden(altered[None], batch["positions"], batch["documents"], a, starts, starts, docs)
    torch.testing.assert_close(trained, other, atol=0, rtol=0)


@pytest.mark.parametrize("keep", range(15))
@torch.inference_mode()
def test_second_round_preserves_original_mixed_source_rule_and_fixed_window(components, monkeypatch, keep):
    draft = target19(components)
    _, _, runner = components
    ids = torch.tensor([[1, 2, 3, 4]])  # t4 is already processed, including its bonus state
    state = runner.prefill(ids)
    n = ids.shape[1]
    first_anchors = make_anchors(ids, state.features, 4, draft.hidden_size)
    first = draft(state.features, first_anchors, n)
    a2 = make_anchors(ids, state.features, 4, draft.hidden_size, first, keep)
    assert a2.source.tolist() == [[1] * max(0, 4-keep) + [2] * min(4, keep)]
    if keep == 8:  # last retained token t12 -> anchors t9..t12, all drafter states
        assert a2.positions.tolist() == [[8, 9, 10, 11]]
        torch.testing.assert_close(a2.draft_features, first.hidden[:, 4:8])
    second = draft(state.features, a2, n + keep)
    fast = TargetFourDraft(draft, 128)
    fast.cache.update("x", 0, state.features)
    actual = fast.pass_hidden(ids, n, torch.tensor(n + keep), first.tokens, first.hidden, fast.masks)
    torch.testing.assert_close(second.hidden, actual, atol=3e-6, rtol=3e-5)
    def scores(logits):
        assert logits.shape[1] == 15
        values = torch.zeros(1, 15)
        values[:, keep] = 1
        return values
    monkeypatch.setattr("four_anchor.inference.boundary_scores", scores)
    proposals, boundary = fast.propose(ids)
    expected = torch.cat((first.tokens[:, :keep], second.tokens[:, :15-keep]), -1)
    assert proposals.shape == (1, 15) and proposals.tolist() == expected.tolist()
    assert boundary.item() == keep
    with pytest.raises(ValueError, match="already be part"):
        fast.propose(ids, torch.tensor([[5]]))


@torch.inference_mode()
def test_actual_bonus_materialization_and_verification_match_greedy_target(components):
    draft = target19(components)
    _, _, runner = components
    prompt = torch.tensor([[1, 2, 3, 4]])
    state = runner.prefill(prompt)
    fast = TargetFourDraft(draft, 256)
    fast.cache.update("x", 0, state.features)
    bonus = state.next_logits.argmax(-1, keepdim=True)
    emitted = bonus[0].tolist()
    before = runner.calls
    for _ in range(6):
        n = state.ids.shape[1]
        # Target input B computes its actual h[B]. Its next-token logits are
        # not used to replace draft candidates or to shift anchor features.
        materialized, bonus_features = runner.forward(bonus, state.cache, all_logits=True)
        fast.cache.update("x", n, bonus_features)
        processed = torch.cat((state.ids, bonus), -1)
        candidates, _ = fast.propose(processed)
        # Native vLLM deliberately schedules the same bonus again. Simulate
        # overwriting its KV slot by cropping to the original processed prefix.
        materialized.past_key_values.crop(n)
        inputs = torch.cat((bonus, candidates), -1)
        out, features = runner.forward(inputs, materialized.past_key_values, all_logits=True)
        predictions = out.logits[:, :-1].argmax(-1)
        bad = (predictions != candidates)[0].nonzero()
        accepted = int(bad[0]) if bad.numel() else 15
        keep = accepted + 1
        bonus = out.logits[:, accepted].argmax(-1, keepdim=True)
        out.past_key_values.crop(n + keep)
        fast.cache.update("x", n, features[:, :keep])
        state.ids = torch.cat((state.ids, inputs[:, :keep]), -1)
        state.cache = out.past_key_values
        emitted += candidates[0, :accepted].tolist() + bonus[0].tolist()
    assert runner.calls - before == 12  # 6 bonus forwards + 6 verification forwards
    ar = runner.prefill(prompt)
    expected = []
    for _ in emitted:
        token = ar.next_logits.argmax(-1, keepdim=True)
        expected += token[0].tolist()
        ar = runner.append(ar, token)
    assert expected == emitted


@pytest.mark.parametrize("accepted", range(16))
def test_adapter_materializes_correct_bonus_and_caches_only_valid_positions(adapter, monkeypatch, accepted):
    runner = adapter.FourAnchorNPUModelRunner.__new__(adapter.FourAnchorNPUModelRunner)
    history = FeatureHistory(128, 2, "cpu", torch.float32)
    history.update("request", 0, torch.arange(4), torch.zeros(4, 2))
    updates, calls = [], []
    def update(req, start, features):
        updates.append((start, features.clone()))
    def materialize(r, token, position, common):
        assert token == 60 and position == 5 + accepted
        calls.append(position)
        return torch.tensor([60]), torch.tensor([[900., 901.]])
    def propose(ids, bonus, mode):
        assert bonus is None
        assert ids[0, -1] == 60
        assert history.features[0, history.length - 1].tolist() == [900., 901.]
        return torch.arange(15)[None], torch.tensor([8])
    inference = NS(requires_materialized_bonus=True, cache=NS(update=update), propose=propose,
                   profiler=NS(region=lambda _: nullcontext(), finish=lambda: None))
    runner.drafter = NS(history=history, layer_ids=[0, 1], inference=inference,
        counters={"extra_target_forwards": 0, "bonus_page_fallbacks": 0},
        settings=NS(num_proposals=15, mode="two_pass", trace_steps=False))
    runner.input_batch = NS(num_reqs=1, req_ids=["request"])
    runner.input_ids = NS(gpu=torch.arange(40, 56))
    runner.max_model_len, runner.device = 128, "cpu"
    monkeypatch.setattr(adapter, "materialize_bonus", materialize)
    output = runner.propose_draft_token_ids(
        [list(range(41, 41 + accepted)) + [60]], NS(all_greedy=True), None,
        NS(num_draft_tokens=[15]), NS(num_computed_tokens_cpu=torch.tensor([4])), None, 16, None,
        [torch.arange(16).float()[:, None], (torch.arange(16) + 100).float()[:, None]])
    assert output == [list(range(15))]
    assert calls == [5 + accepted]
    assert [u[0] for u in updates] == [4, 5 + accepted]
    assert history.length == 6 + accepted
    assert runner.drafter.counters == {"extra_target_forwards": 1, "bonus_page_fallbacks": 0}


def test_bonus_forward_attention_is_single_token_and_stops_before_rejected_suffix(adapter, monkeypatch):
    import four_anchor.vllm_target_bonus as helper
    seen = {}
    def build(**kw):
        seen["common"] = kw["common_attn_metadata"]
        return kw["common_attn_metadata"]
    runner = NS(device="cpu", vllm_config=NS(), drafter=NS(layer_ids=[0, 1]),
        attn_groups=[[NS(layer_names=["layer"], get_metadata_builder=lambda _: NS(build=build))]],
        input_batch=NS(block_table=[NS(num_blocks_per_row=[2], block_size=8)]))
    def model(input_ids, positions):
        assert input_ids.tolist() == [60] and positions.tolist() == [9]
        return None, [torch.tensor([[9.]]), torch.tensor([[90.]])]
    runner.model = model
    ids, features = helper.materialize_bonus(runner, 60, 9, NS(block_table_tensor=torch.tensor([[3, 4]])))
    assert ids.tolist() == [60] and features.tolist() == [[9., 90.]]
    cm = seen["common"]
    assert cm.seq_lens.tolist() == [10] and cm.num_computed_tokens_cpu.tolist() == [9]
    assert cm.slot_mapping.tolist() == [33] and cm.query_start_loc_cpu.tolist() == [0, 1]
    assert cm.num_actual_tokens == 1 and cm.causal
    assert helper.materialize_bonus(runner, 60, 16, NS(block_table_tensor=torch.tensor([[3, 4]]))) is None


def test_counter_rpc_returns_actual_counts_and_new_checkpoint_contract(components, tmp_path):
    from four_anchor.vllm_entry import worker_counters
    draft = target19(components)
    draft.save(tmp_path)
    assert ServingSettings(str(tmp_path)).inference_version() == "target_four_materialized_v3"
    assert worker_counters(NS(collective_rpc=lambda _: [{"extra_target_forwards": 23, "bonus_page_fallbacks": 1}])) == {
        "extra_target_forwards": 23, "bonus_page_fallbacks": 1}
    with pytest.raises(RuntimeError):
        worker_counters(NS(collective_rpc=lambda _: [{"extra_target_forwards": -1, "bonus_page_fallbacks": 0}]))


@torch.inference_mode()
def test_target_layout_has_room_at_final_valid_context_boundary(components):
    draft = target19(components)
    fast = TargetFourDraft(draft, 64)
    ids = (torch.arange(34)[None] % 60) + 1
    fast.cache.update("x", 0, torch.randn(1, 34, 64))
    tokens, _ = fast.propose(ids)
    assert tokens.shape == (1, 15)


def test_evaluation_reports_extra_forwards_excluding_warmup(components, tmp_path, monkeypatch):
    import json
    import sys
    from types import ModuleType
    from four_anchor.vllm_entry import evaluate
    draft = target19(components)
    checkpoint = tmp_path / "checkpoint"
    draft.save(checkpoint)
    module = ModuleType("vllm")
    class FakeLLM:
        def __init__(self, **kwargs):
            self.extra = 0
        def collective_rpc(self, method):
            assert method == "four_anchor_counters"
            return [{"extra_target_forwards": self.extra, "bonus_page_fallbacks": 0}]
        def generate(self, *args, **kwargs):
            self.extra += 3
            metrics = NS(to_dict=lambda: {"num_spec_tokens": 15, "per_step_accepted": [1, 0],
                "num_spec_steps": 2, "num_accepted_draft_tokens": 1})
            return [NS(outputs=[NS(token_ids=[4, 5, 6], text="tiny response", spec_decode_metrics=metrics)])]
    module.LLM, module.SamplingParams = FakeLLM, lambda **kwargs: kwargs
    monkeypatch.setitem(sys.modules, "vllm", module)
    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", lambda *a, **kw: None)
    monkeypatch.setattr("four_anchor.data.load_records", lambda _: [{"input_ids": [1, 2, 3]}] * 2)
    args = NS(checkpoint=str(checkpoint), proposals=15, target="tiny", data="unused", output=str(tmp_path / "out"),
              limit=2, max_new_tokens=32, max_model_len=128, baseline=None, warmup=2, mode="two_pass")
    evaluate(args, {})
    summary = json.loads((tmp_path / "out/summary.json").read_text())
    assert summary["extra_target_forwards"] == 6  # excludes the 6 warmup calls
    assert summary["verification_cycles"] == 4
    assert summary["query_positions"] == 19 and summary["first_round_target_anchor_states"] == 4
    assert summary["inference_version"] == "target_four_materialized_v3"
