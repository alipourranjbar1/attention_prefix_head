from types import SimpleNamespace as NS
import importlib.util
from pathlib import Path

import pytest
import torch

from four_anchor.batch_inference import BatchedBonusAddonDraft
from four_anchor.inference import BonusAddonDraft
from four_anchor.tests.test_accepted_prefix import target19
from four_anchor.tests.test_vllm_adapter import adapter
from four_anchor.vllm_batch_entry import summarize


@pytest.mark.parametrize("batch_size", [1, 2, 8, 32])
@pytest.mark.parametrize("mode", ["one_pass", "two_pass"])
@torch.inference_mode()
def test_batched_matches_independent_requests_and_reordering(components, batch_size, mode):
    model = target19(components)
    cache = BatchedBonusAddonDraft(model, 128, 32)
    names = [str(i) for i in range(batch_size)]
    features, ids, singles, bonuses, updates = [], [], [], [], []
    for i, name in enumerate(names):
        n = 1 + i % 17
        features.append(torch.randn(1, n, model.fc.in_features))
        ids.append(torch.randint(1, 63, (1, n)))
        bonuses.append((i + 11) % 63)
        single = BonusAddonDraft(model, 128)
        single.cache.update(name, 0, features[-1])
        singles.append(single)
        updates.append((name, 0, ids[-1][0], features[-1][0]))
    cache.cache.update(updates, names)
    # Poison masked cache storage: padding must not read it.
    for k, v in cache.cache.memory:
        for name in names:
            s, n = cache.cache.slots[name], cache.cache.lengths[name]
            k[s, :, n:].normal_(); v[s, :, n:].normal_()
    observed = []
    original = cache.query_hidden
    def record(*a):
        value = original(*a); observed.append(value.clone()); return value
    cache.query_hidden = record
    batch, boundaries = cache.propose(names[::-1], bonuses[::-1], mode)
    for row, i in enumerate(reversed(range(batch_size))):
        outputs = []
        fn = singles[i].query_hidden
        def capture(*a):
            value = fn(*a); outputs.append(value.clone()); return value
        singles[i].query_hidden = capture
        expected, b = singles[i].propose(ids[i], torch.tensor([[bonuses[i]]]), mode)
        assert batch[row].tolist() == expected[0].tolist()
        if mode == "two_pass":
            assert boundaries[row].item() == b.item()
        for j, value in enumerate(outputs):
            torch.testing.assert_close(observed[j][row:row+1], value, atol=3e-6, rtol=4e-5)
    assert observed[0].shape == (batch_size, 20, model.hidden_size)
    if mode == "two_pass":
        assert observed[1].shape == (batch_size, 19, model.hidden_size)


@torch.inference_mode()
def test_slot_reuse_rollback_and_missing_prefix_fail_closed(components):
    model = target19(components)
    fast = BatchedBonusAddonDraft(model, 128, 2)
    c = fast.cache
    def update(rid, start, n, active):
        c.update([(rid, start, torch.arange(n), torch.randn(n, model.fc.in_features))], active)
    update("a", 0, 10, ["a"]); update("b", 0, 20, ["a", "b"])
    slot = c.slots["a"]
    c.release(["a"])
    update("c", 0, 3, ["b", "c"])
    assert c.slots["c"] == slot and c.lengths == {"b": 20, "c": 3}
    update("b", 8, 2, ["b", "c"])
    assert c.lengths["b"] == 10
    with pytest.raises(RuntimeError, match="Gap"):
        update("b", 11, 1, ["b", "c"])
    c.release(["b"])
    with pytest.raises(RuntimeError, match="Missing"):
        update("b", 10, 1, ["b", "c"])


@pytest.fixture
def batch_adapter(adapter):
    path = Path(__file__).parents[1] / "vllm_batch_ascend.py"
    spec = importlib.util.spec_from_file_location("four_anchor._batch_adapter_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("accepted", range(16))
@pytest.mark.parametrize("input_dtype", [torch.int32, torch.int64])
@torch.inference_mode()
def test_native_flattened_rows_prefill_decode_and_rejected_features(batch_adapter, components, accepted, input_dtype):
    model = target19(components)
    fast = BatchedBonusAddonDraft(model, 128, 32)
    fast.cache.update([("decode", 0, torch.tensor([1, 2, 3, 4]), torch.randn(4, model.fc.in_features))], ["decode"])
    r = batch_adapter.BatchedFourAnchorRunner.__new__(batch_adapter.BatchedFourAnchorRunner)
    r.drafter = NS(layer_ids=[0, 1], inference=fast, settings=NS(mode="two_pass", trace_steps=False),
                   batch_stats={"draft_batches": 0, "draft_requests": 0, "max_draft_batch": 0})
    r.input_batch = NS(num_reqs=3, req_ids=["new", "decode", "chunk"])
    r.input_ids = NS(gpu=torch.arange(25, dtype=input_dtype) % 63)
    r.max_model_len = 128
    width = model.fc.in_features // 2
    aux = [torch.randn(25, width), torch.randn(25, width)]
    seen = {}
    def propose(names, bonuses, mode):
        seen.update(names=names, bonuses=bonuses)
        return torch.arange(30).view(2, 15), None
    fast.propose = propose
    result = r.propose_draft_token_ids([[9], list(range(accepted)) + [50], []], NS(all_greedy=True),
        NS(finished_req_ids=[]), NS(num_draft_tokens=[0, 15, 0]),
        NS(num_computed_tokens_cpu=torch.tensor([0, 4, 0]), query_start_loc_cpu=torch.tensor([0, 6, 22, 25])),
        None, 25, None, aux)
    assert result == [list(range(15)), list(range(15, 30)), []]
    assert seen == {"names": ["new", "decode"], "bonuses": [9, 50]}
    assert fast.cache.lengths == {"decode": 5 + accepted, "new": 6, "chunk": 3}
    s = fast.cache.slots["decode"]
    assert fast.cache.ids.dtype == torch.int64
    assert fast.cache.ids[s, 4:5+accepted].tolist() == list(range(6, 7+accepted))
    features = torch.cat([x[6:7+accepted] for x in aux], -1)
    torch.testing.assert_close(fast.cache.projected[s, 4:5+accepted], model.hidden_norm(model.fc(features)))
    assert r.drafter.batch_stats["max_draft_batch"] == 2
    # The unfinished prefill is cached and can continue without a gap.
    fast.cache.update([("chunk", 3, torch.tensor([4]), torch.randn(1, model.fc.in_features))], ["chunk"])
    assert fast.cache.lengths["chunk"] == 4


def test_throughput_uses_shared_wall_time_and_pooled_cycles():
    rows = [{"output_tokens": 100, "latency_including_queue_seconds": 10,
             "speculative_decoding": {"num_spec_steps": 10, "num_accepted_draft_tokens": 70}},
            {"output_tokens": 200, "latency_including_queue_seconds": 12,
             "speculative_decoding": {"num_spec_steps": 20, "num_accepted_draft_tokens": 110}}]
    report = summarize(rows, 12, 32)
    assert report["tokens_per_second_including_prefill"] == 25
    assert report["acceptance_length_including_bonus"] == 7
    assert report["concurrency"] == 32 and report["seconds"] == 12


@torch.inference_mode()
def test_batch_proposer_profiles_32_and_preserves_checkpoint(batch_adapter, components, monkeypatch, tmp_path):
    target, _, _ = components
    model = target19(components)
    model.save(tmp_path)
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    cfg = NS(model_config=NS(model="tiny", dtype=torch.float32, max_model_len=128,
        hf_config=NS(four_anchor_serving={"checkpoint": str(tmp_path), "anchor_policy": "bonus_addon"})),
        scheduler_config=NS(max_num_seqs=32))
    monkeypatch.setattr(batch_adapter, "FrozenTargetInterface", lambda *a: target)
    monkeypatch.setattr(batch_adapter.FourAnchorDFlash, "load", lambda *a: model)
    p = batch_adapter.BatchedFourAnchorProposer(cfg, "cpu")
    p.load_model(None)
    observed = []
    original = p.inference.propose
    def propose(names, bonuses, mode):
        observed.append(len(names)); return original(names, bonuses, mode)
    p.inference.propose = propose
    p.dummy_run()
    assert observed == [32] and p.inference.cache.slots == {}
    assert p.counters == {"extra_target_forwards": 0, "bonus_page_fallbacks": 0}
    assert before == {p.name: p.read_bytes() for p in tmp_path.iterdir()}
