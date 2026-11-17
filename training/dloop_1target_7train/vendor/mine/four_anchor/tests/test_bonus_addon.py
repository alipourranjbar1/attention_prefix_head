"""Bonus input, strict four-drafter-state redrafting, and verifier accounting."""
from types import SimpleNamespace as NS, ModuleType
import sys
import json

import pytest
import torch

from four_anchor.inference import BonusAddonDraft
from four_anchor.model import Anchors, make_anchors
from four_anchor.vllm_support import ServingSettings
from four_anchor.tests.test_accepted_prefix import target19
from four_anchor.tests.test_cached_benchmarks import suite, tiny_model, fixture_sources, restore_prepare_environment
from four_anchor.tests.test_vllm_adapter import adapter


@torch.inference_mode()
def dense_all(model, features, anchors, start, slots):
    """Uncached original decoder equations; includes outputs at anchor slots."""
    x = model.embed_tokens(torch.cat((anchors.ids, slots), 1))
    f = model.hidden_norm(model.fc(anchors.target_features)) * (anchors.source == 1)[..., None]
    f += model.draft_anchor_norm(model.draft_anchor_proj(anchors.draft_features)) * (anchors.source == 2)[..., None]
    x += torch.cat((f, torch.zeros_like(x[:, 4:])), 1)
    ctx = model.hidden_norm(model.fc(features))
    pos = torch.cat((torch.arange(ctx.shape[1])[None], anchors.positions,
                     torch.arange(start, start + slots.shape[1])[None]), 1)
    valid = torch.cat((torch.ones((1, ctx.shape[1]), dtype=torch.bool),
                       anchors.source != 0, torch.ones_like(slots, dtype=torch.bool)), 1)
    mask = torch.zeros((1, 1, x.shape[1], valid.shape[1]), dtype=x.dtype)
    mask.masked_fill_(~valid[:, None, None], torch.finfo(x.dtype).min)
    rope = model.rotary_emb(x, pos)
    for layer in model.layers:
        x = layer(hidden_states=x, target_hidden=ctx, attention_mask=mask,
                  position_ids=pos, position_embeddings=rope, use_cache=False)
    return model.norm(x)


@pytest.mark.parametrize("boundary", range(15))
@pytest.mark.parametrize("length", [1, 2, 4, 10])
@torch.inference_mode()
def test_all_boundaries_and_padding_use_four_first_pass_states(components, monkeypatch, boundary, length):
    model = target19(components)
    _, _, runner = components
    ids = torch.arange(1, length + 1)[None]
    state = runner.prefill(ids)
    bonus = state.next_logits.argmax(-1, keepdim=True)
    fast = BonusAddonDraft(model, 128)
    fast.cache.update("x", 0, state.features)
    a1 = make_anchors(ids, state.features, 4, model.hidden_size)
    slots = torch.cat((bonus, fast.masks), 1)
    h1 = dense_all(model, state.features, a1, length, slots)
    first = model.lm_head(h1[:, 5:]).argmax(-1)
    restart = length + 1 + boundary
    positions = torch.arange(restart-4, restart)[None]
    valid = positions >= 0
    select = torch.arange(boundary+1, boundary+5)
    a2 = Anchors(torch.cat((a1.ids, bonus, first), 1)[:, select], positions.clamp_min(0),
                 torch.zeros(1, 4, state.features.shape[-1]), h1[:, select], valid.long()*2)
    second = model(state.features, a2, restart)
    observed = []
    original = fast.query_hidden
    def record(*args):
        out = original(*args)
        observed.append((args, out.clone()))
        return out
    monkeypatch.setattr(fast, "query_hidden", record)
    def scores(logits):
        assert logits.shape == (1, 15, 64)
        out = torch.zeros(1, 15); out[0, boundary] = 1
        return out
    monkeypatch.setattr("four_anchor.inference.boundary_scores", scores)
    calls = runner.calls
    actual, chosen = fast.propose(ids, bonus)
    assert runner.calls == calls and fast.cache.length == length
    assert chosen.item() == boundary and len(observed) == 2
    assert observed[0][0][-2].tolist() == slots.tolist()  # bonus is a real input
    assert [out.shape[1] for _, out in observed] == [20, 19]
    torch.testing.assert_close(observed[0][1], h1, rtol=3e-5, atol=3e-6)
    torch.testing.assert_close(observed[1][1][:, 4:], second.hidden, rtol=3e-5, atol=3e-6)
    torch.testing.assert_close(observed[1][0][4],
        model.draft_anchor_norm(model.draft_anchor_proj(h1[:, select])) * valid[..., None],
        rtol=3e-5, atol=3e-6)
    assert observed[1][0][2].tolist() == a2.ids.tolist()
    assert observed[1][0][3].tolist() == positions.clamp_min(0).tolist()
    expected = torch.cat((first[:, :boundary], second.tokens[:, :15-boundary]), 1)
    assert actual.shape == (1, 15) and actual.tolist() == expected.tolist()


@torch.inference_mode()
def test_bonus_changes_first_pass_and_future_cache_is_invisible(components):
    model = target19(components)
    _, _, runner = components
    ids = torch.tensor([[1, 2, 3, 4]])
    state = runner.prefill(ids)
    fast = BonusAddonDraft(model, 128)
    fast.cache.update("x", 0, state.features)
    # Simulate graph context padding and dirty, rejected cache beyond n.
    a = fast.anchors(ids, torch.tensor(4), torch.tensor(4), fast.zero_ids, fast.zero_hidden)
    def compute(b, input_ids=ids):
        return fast.query_hidden(input_ids, torch.tensor(4), *a,
                                  torch.cat((torch.tensor([[b]]), fast.masks), 1), torch.tensor(4))
    first = compute(5)
    assert not torch.allclose(first[:, 5:], compute(6)[:, 5:])
    for k, v in fast.cache.memory:
        k[:, :, 4:].normal_(); v[:, :, 4:].normal_()
    padded_ids = torch.cat((ids, torch.zeros(1, 16, dtype=torch.long)), 1)
    torch.testing.assert_close(first, compute(5, padded_ids), rtol=3e-5, atol=3e-6)


@pytest.mark.parametrize("budget", [1, 2, 17, 53])
@torch.inference_mode()
def test_real_tiny_target_greedy_parity_one_verify_per_cycle(components, budget):
    model = target19(components)
    _, _, runner = components
    prompt = torch.tensor([[1, 2, 3, 4]])
    state = runner.prefill(prompt)
    fast = BonusAddonDraft(model, 256)
    fast.cache.update("x", 0, state.features)
    bonus = state.next_logits.argmax(-1, keepdim=True)
    emitted = bonus[0].tolist()
    before, cycles = runner.calls, 0
    while len(emitted) < budget:
        n = state.ids.shape[1]
        candidates, _ = fast.propose(state.ids, bonus)
        inputs = torch.cat((bonus, candidates), 1)
        assert inputs.shape[1] == 16
        out, features = runner.forward(inputs, state.cache, all_logits=True)
        bad = (out.logits[:, :-1].argmax(-1) != candidates)[0].nonzero()
        accepted = int(bad[0]) if bad.numel() else 15
        bonus = out.logits[:, accepted].argmax(-1, keepdim=True)
        keep = accepted + 1
        out.past_key_values.crop(n + keep)
        fast.cache.update("x", n, features[:, :keep])
        state.ids = torch.cat((state.ids, inputs[:, :keep]), 1)
        state.cache = out.past_key_values
        emitted += candidates[0, :accepted].tolist() + bonus[0].tolist()
        cycles += 1
    assert runner.calls - before == cycles
    ar = runner.prefill(prompt)
    expected = []
    for _ in range(budget):
        token = ar.next_logits.argmax(-1, keepdim=True)
        expected += token[0].tolist(); ar = runner.append(ar, token)
    assert emitted[:budget] == expected
    eos = expected[-1]
    assert emitted[:expected.index(eos)+1] == expected[:expected.index(eos)+1]


@torch.inference_mode()
def test_policy_checkpoint_immutable_and_native_adapter_selects_it(components, adapter, monkeypatch, tmp_path):
    target, _, _ = components
    model = target19(components)
    model.save(tmp_path)
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    settings = ServingSettings(str(tmp_path), anchor_policy="bonus_addon")
    assert settings.inference_version() == "bonus_addon20_draft4_v1"
    config = NS(model_config=NS(hf_config=NS(four_anchor_serving={"checkpoint": str(tmp_path),
        "anchor_policy": "bonus_addon"}), model="tiny", dtype=torch.float32, max_model_len=128))
    monkeypatch.setattr(adapter, "FrozenTargetInterface", lambda *a: target)
    monkeypatch.setattr(adapter.FourAnchorDFlash, "load", lambda *a: model)
    proposer = adapter.FourAnchorProposer(config, "cpu")
    proposer.load_model(None)
    assert isinstance(proposer.inference, BonusAddonDraft)
    assert proposer.inference.requires_materialized_bonus is False
    proposer.dummy_run()
    assert before == {p.name: p.read_bytes() for p in tmp_path.iterdir()}


def test_all_benchmark_frozen_prompts_are_preserved(suite, tmp_path, monkeypatch):
    model = tmp_path / "model"; tiny_model(model)
    cache = fixture_sources(suite, tmp_path)
    from pathlib import Path
    repo = Path(__file__).parents[2]
    monkeypatch.setattr(suite, "cache_roots", lambda _: [cache])
    common = ["--root", str(repo), "--code-root", str(repo), "--inference-layout", "bonus_addon",
              "--model", str(model), "--prepare-only", "--limit", "2"]
    frozen, output = tmp_path / "frozen", tmp_path / "new_eval"
    assert suite.main(common + ["--output", str(frozen)]) == 0
    before = {str(p.relative_to(frozen)): p.read_bytes() for p in frozen.rglob("*") if p.is_file()}
    assert suite.main(common + ["--output", str(output), "--prompt-plan", str(frozen / "plan.json")]) == 0
    plan = json.loads((output / "plan.json").read_text())
    assert plan["method"] == "bonus_addon20_draft4_v1"
    assert plan["query_positions"] == {"round_one": 20, "round_two": 19}
    assert not plan["with_baseline"] and len(plan["benchmarks"]) == 8
    assert before == {str(p.relative_to(frozen)): p.read_bytes() for p in frozen.rglob("*") if p.is_file()}

@pytest.mark.parametrize("extra", [0, 1])
def test_bonus_addon_evaluator_counts_actual_forwards(components, tmp_path, monkeypatch, extra):
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
              baseline=None, warmup=0, mode="two_pass", anchor_policy="bonus_addon")
    if extra:
        with pytest.raises(RuntimeError, match="extra target forward"):
            evaluate(args, {})
        assert not (tmp_path / "out/summary.json").exists()
    else:
        evaluate(args, {})
        result = json.loads((tmp_path / "out/summary.json").read_text())
        assert result["extra_target_forwards_measured"] and result["extra_target_forwards"] == 0
        assert result["query_positions"] == {"round_one": 20, "round_two": 19} and result["speculative_tokens"] == 15
        assert result["acceptance_length_including_bonus"] == 1.5
    assert calls == ["four_anchor_counters"] * 2
