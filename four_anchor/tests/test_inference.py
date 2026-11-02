"""Numerical tests for cached known-bonus drafting, without Ascend hardware."""
import pytest
import torch

from four_anchor.inference import KnownBonusDraft
from four_anchor.model import Anchors, DraftOutput, make_anchors


@torch.inference_mode()
def dense_reference(model, features, anchors, start, slot_ids):
    """Independent full-context route through the original decoder layers."""
    a, width = 4, slot_ids.shape[1]
    x = model.embed_tokens(torch.cat((anchors.ids, slot_ids), 1))
    tf = model.hidden_norm(model.fc(anchors.target_features))
    df = model.draft_anchor_norm(model.draft_anchor_proj(anchors.draft_features))
    f = tf * (anchors.source == 1)[..., None] + df * (anchors.source == 2)[..., None]
    x = x + torch.cat((f, torch.zeros_like(x[:, a:])), 1)
    ctx = model.hidden_norm(model.fc(features))
    pos = torch.cat((torch.arange(ctx.shape[1])[None], anchors.positions,
                     torch.arange(start, start + width)[None]), 1)
    rope = model.rotary_emb(x, pos)
    valid = torch.cat((torch.ones((1, ctx.shape[1]), dtype=torch.bool),
                       anchors.source != 0, torch.ones_like(slot_ids, dtype=torch.bool)), 1)
    mask = torch.zeros((1, 1, a + width, valid.shape[1]))
    mask.masked_fill_(~valid[:, None, None], torch.finfo(x.dtype).min)
    for layer in model.layers:
        x = layer(hidden_states=x, target_hidden=ctx, attention_mask=mask,
                  position_ids=pos, position_embeddings=rope, use_cache=False)
    return model.norm(x[:, a:])


@pytest.mark.parametrize("n", [1, 2, 4, 9, 127])
@torch.inference_mode()
def test_known_bonus_cached_first_pass_matches_dense(components, n):
    _, model, runner = components
    ids = (torch.arange(n)[None] % 60) + 1
    state = runner.prefill(ids)
    fast = KnownBonusDraft(model, 256)
    fast.cache.update("a", 0, state.features)
    bonus = state.next_logits.argmax(-1, keepdim=True)
    slots = torch.cat((bonus, fast.masks[:, 1:]), 1)
    anchors = make_anchors(ids, state.features, 4, model.hidden_size)
    expected = dense_reference(model, state.features, anchors, n, slots)
    actual = fast.pass_hidden(ids, n, torch.tensor(n), fast.zero_ids, fast.zero_hidden, slots)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    # This is conditioning on a known token, not a fake target feature.
    changed_slots = slots.clone()
    changed_slots[0, 0] = (bonus[0, 0] + 1) % 60
    changed = fast.pass_hidden(ids, n, torch.tensor(n), fast.zero_ids, fast.zero_hidden, changed_slots)
    assert not torch.allclose(actual[:, 1:], changed[:, 1:])


@pytest.mark.parametrize("boundary", range(15))
@torch.inference_mode()
def test_second_pass_anchor_positions_sources_and_dense_parity(components, monkeypatch, boundary):
    _, model, runner = components
    ids = torch.tensor([[2, 3, 4, 5, 6, 7, 8, 9]])
    state = runner.prefill(ids)
    n = ids.shape[1]
    fast = KnownBonusDraft(model, 128)
    fast.cache.update("a", 0, state.features)
    bonus = state.next_logits.argmax(-1, keepdim=True)
    slots = torch.cat((bonus, fast.masks[:, 1:]), 1)
    a1 = make_anchors(ids, state.features, 4, model.hidden_size)
    h1 = dense_reference(model, state.features, a1, n, slots)
    first = model.lm_head(h1[:, 1:]).argmax(-1)
    # Include the known bonus in the temporary first-pass sequence, but not
    # in target memory or in the selector's 15 future-position probabilities.
    prior_ids = torch.cat((bonus, first), 1)
    fake_logits = torch.full((1, 16, 64), -100.)
    fake_logits.scatter_(-1, prior_ids[..., None], 100.)
    a2 = make_anchors(ids, state.features, 4, model.hidden_size,
                      DraftOutput(fake_logits, h1), 1 + boundary)
    restart = n + 1 + boundary
    token_ids, positions, projected, valid = fast.anchors(
        ids, n, torch.tensor(restart), prior_ids, h1)
    assert positions.tolist() == [list(range(restart - 4, restart))]
    assert token_ids.tolist() == a2.ids.tolist()
    assert valid.all()
    expected_features = (model.hidden_norm(model.fc(a2.target_features)) * (a2.source == 1)[..., None]
                         + model.draft_anchor_norm(model.draft_anchor_proj(a2.draft_features))
                         * (a2.source == 2)[..., None])
    torch.testing.assert_close(projected, expected_features, rtol=2e-5, atol=2e-6)
    expected_h2 = dense_reference(model, state.features, a2, restart, fast.masks)
    actual_h2 = fast.pass_hidden(ids, n, torch.tensor(restart), prior_ids, h1, fast.masks)
    torch.testing.assert_close(actual_h2, expected_h2, rtol=2e-5, atol=2e-6)
    observed = []
    def scores(logits):
        observed.append(logits.shape)
        result = torch.zeros(1, 15)
        result[0, boundary] = 1
        return result
    monkeypatch.setattr("four_anchor.inference.boundary_scores", scores)
    got, chosen = fast.propose(ids, bonus)
    expected_second = model.lm_head(expected_h2[:, :15]).argmax(-1)
    expected = torch.cat((first[:, :boundary], expected_second[:, :15-boundary]), 1)
    assert chosen.tolist() == [boundary]
    assert observed == [torch.Size([1, 15, 64])]
    assert got.shape == (1, 15)
    assert got.tolist() == expected.tolist()


@torch.inference_mode()
def test_cache_projects_only_delta_and_rolls_back_or_resets(components):
    _, model, _ = components
    fast = KnownBonusDraft(model, 128)
    features = torch.randn(1, 15, 64)
    seen = []
    hook = model.fc.register_forward_pre_hook(lambda _, args: seen.append(args[0].shape[1]))
    try:
        fast.cache.update("a", 0, features[:, :8])
        fast.cache.update("a", 8, features[:, 8:12])
        before = [k.clone() for k, _ in fast.cache.memory]
        fast.cache.update("a", 12, features[:, 12:])
        assert seen == [8, 4, 3]
        for old, (k, _) in zip(before, fast.cache.memory):
            torch.testing.assert_close(old[:, :, :12], k[:, :, :12], rtol=0, atol=0)
        expected = model.packed_memory(features, torch.arange(15))
        for (k, v), (rk, rv) in zip(fast.cache.memory, expected):
            torch.testing.assert_close(k[:, :, :15], rk, rtol=2e-5, atol=2e-6)
            torch.testing.assert_close(v[:, :, :15], rv, rtol=2e-5, atol=2e-6)
        fast.cache.update("a", 10, features[:, 10:11])
        assert fast.cache.length == 11
        with pytest.raises(ValueError, match="gap"):
            fast.cache.update("a", 12, features[:, :1])
        fast.cache.update("b", 0, features[:, :2])
        assert fast.cache.length == 2
    finally:
        hook.remove()


@pytest.mark.parametrize("mode", ["one_pass", "two_pass"])
@torch.inference_mode()
def test_native_style_verification_matches_ar_without_bonus_forward(components, mode):
    _, model, runner = components
    ids = torch.tensor([[2, 3, 4, 5]])
    state = runner.prefill(ids)
    fast = KnownBonusDraft(model, 256)
    fast.cache.update("request", 0, state.features)
    bonus = state.next_logits.argmax(-1, keepdim=True)
    emitted = bonus[0].tolist()
    calls_before = runner.calls
    for _ in range(6):
        candidates, _ = fast.propose(state.ids, bonus, mode)
        n = state.ids.shape[1]
        # Exactly one target forward: pending bonus + 15 proposals.
        inputs = torch.cat((bonus, candidates), 1)
        out, features = runner.forward(inputs, state.cache, all_logits=True)
        expected = out.logits[:, :-1].argmax(-1)
        bad = (expected != candidates)[0].nonzero()
        accepted = int(bad[0]) if bad.numel() else 15
        keep = 1 + accepted
        bonus = out.logits[:, accepted].argmax(-1, keepdim=True)
        out.past_key_values.crop(n + keep)
        fast.cache.update("request", n, features[:, :keep])
        state.ids = torch.cat((state.ids, inputs[:, :keep]), 1)
        state.cache = out.past_key_values
        emitted += candidates[0, :accepted].tolist() + bonus[0].tolist()
    assert runner.calls - calls_before == 6
    baseline = runner.prefill(ids)
    expected_ids = []
    for _ in emitted:
        token = baseline.next_logits.argmax(-1, keepdim=True)
        expected_ids += token[0].tolist()
        baseline = runner.append(baseline, token)
    assert emitted == expected_ids


def test_no_host_scalar_readback_inside_drafting(components, monkeypatch):
    _, model, runner = components
    ids = torch.tensor([[2, 3, 4, 5]])
    state = runner.prefill(ids)
    fast = KnownBonusDraft(model, 128)
    fast.cache.update("a", 0, state.features)
    def forbidden(*args, **kwargs):
        raise AssertionError("No scalar/list readbacks inside drafting")
    monkeypatch.setattr(torch.Tensor, "item", forbidden)
    monkeypatch.setattr(torch.Tensor, "tolist", forbidden)
    tokens, boundary = fast.propose(ids, torch.tensor([[8]]))
    assert tokens.shape == (1, 15) and boundary.shape == (1,)


@pytest.mark.parametrize("n", [1, 4, 31, 63])
@torch.inference_mode()
def test_graph_tensor_body_masks_padded_and_stale_context(components, n):
    _, model, runner = components
    ids = (torch.arange(n)[None] % 60) + 1
    state = runner.prefill(ids)
    fast = KnownBonusDraft(model, 128)
    fast.cache.update("previous_request", 0, torch.randn(1, 100, 64))
    fast.cache.update("a", 0, state.features)
    bonus = state.next_logits.argmax(-1, keepdim=True)
    expected, boundary = fast.propose(ids, bonus)
    padded_ids = torch.full((1, 64), 57, dtype=torch.long)
    padded_ids[:, :n] = ids
    actual, actual_boundary = fast.tensor_propose(padded_ids, bonus, torch.tensor(n), "two_pass")
    assert actual.tolist() == expected.tolist()
    assert actual_boundary.tolist() == boundary.tolist()


@torch.inference_mode()
def test_graph_replay_refreshes_bonus_prefix_and_length(components):
    from four_anchor.inference import NpuDraftGraphs
    from types import SimpleNamespace
    _, model, _ = components
    fast = KnownBonusDraft(model, 128)
    manager = NpuDraftGraphs.__new__(NpuDraftGraphs)
    manager.draft, manager.buckets, manager.entries = fast, [32, 128], {}
    fixed_ids = torch.zeros(1, 32, dtype=torch.long)
    fixed_bonus = torch.zeros(1, 1, dtype=torch.long)
    length = torch.zeros((), dtype=torch.long)
    outputs = (torch.zeros(1, 15, dtype=torch.long), torch.zeros(1, dtype=torch.long))
    def replay():
        result = fast.tensor_propose(fixed_ids, fixed_bonus, length, "two_pass")
        for dest, src in zip(outputs, result):
            dest.copy_(src)
    manager.entries[32, "two_pass"] = (SimpleNamespace(replay=replay), fixed_ids, fixed_bonus, length, outputs)
    for n, token in [(5, 10), (9, 18), (2, 27)]:
        ids = torch.arange(n)[None]
        fast.cache.update(str(n), 0, torch.randn(1, n, 64))
        bonus = torch.tensor([[token]])
        expected, boundary = fast.propose(ids, bonus)
        actual, actual_boundary = manager.run(ids, bonus, "two_pass")
        assert int(length) == n and int(fixed_bonus[0, 0]) == token
        assert fixed_ids[:, :n].tolist() == ids.tolist()
        assert actual.tolist() == expected.tolist()
        assert actual_boundary.tolist() == boundary.tolist()


def test_graph_mode_requires_npu(components):
    _, model, _ = components
    with pytest.raises(ValueError, match="Ascend NPU"):
        KnownBonusDraft(model, 128, draft_graph=True)
