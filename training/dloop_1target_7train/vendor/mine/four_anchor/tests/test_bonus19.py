from copy import deepcopy
from dataclasses import replace
import json
import random

import pytest
import torch

from four_anchor.bonus_objective import bonus_anchors
from four_anchor.inference import BonusInFourDraft, KnownBonusDraft
from four_anchor.model import DraftConfig, FourAnchorDFlash
from four_anchor.packed_objective import PackedTask
from four_anchor.vllm_support import ServingSettings
from four_anchor.tests.test_packed_training import batch_for, tiny_config


def new_draft(components):
    _, draft, _ = components
    draft.cfg = replace(draft.cfg, num_proposals=15, anchor_layout="bonus_in_four", format_version=2)
    return draft


@pytest.mark.parametrize("start", [2, 3, 4, 8])
@pytest.mark.parametrize("cached", [False, True])
def test_training_first_pass_matches_serving_with_no_bonus_target_state(components, start, cached):
    draft = new_draft(components)
    batch = batch_for(components)
    starts, docs = torch.tensor([start]), torch.tensor([0])
    a = bonus_anchors(batch, starts, starts, docs, draft.hidden_size)
    n = start - 1
    assert a.source[0, -1] == 3
    assert a.ids[0, -1] == batch["ids"][n]
    assert not a.target_features[0, -1].any()
    assert not a.draft_features[0, -1].any()
    memory = draft.packed_memory(batch["features"][None], batch["positions"]) if cached else None
    trained = draft.packed_hidden(batch["features"][None], batch["positions"], batch["documents"],
                                  a, starts, starts - 1, docs, memory=memory, include_bonus=True)
    with torch.inference_mode():
        fast = BonusInFourDraft(draft, 128)
        fast.cache.update("x", 0, batch["features"][:n][None])
        prior_ids = fast.zero_ids.clone()
        prior_ids[:, 0] = batch["ids"][n]
        actual = fast.pass_hidden(batch["ids"][:n][None], n, torch.tensor(start),
                                  prior_ids, fast.zero_hidden, fast.masks)
    assert trained.shape == actual.shape == (1, 16, draft.hidden_size)
    torch.testing.assert_close(trained, actual, atol=3e-6, rtol=3e-5)
    changed = batch["features"].clone()
    changed[n:] = 100 * torch.randn_like(changed[n:])
    altered_memory = draft.packed_memory(changed[None], batch["positions"]) if cached else None
    other = draft.packed_hidden(changed[None], batch["positions"], batch["documents"],
                               a, starts, starts - 1, docs, memory=altered_memory, include_bonus=True)
    torch.testing.assert_close(trained, other, atol=0, rtol=0)


@pytest.mark.parametrize("boundary", range(15))
def test_second_pass_states_positions_and_final_window_match_serving(components, boundary, monkeypatch):
    draft = new_draft(components)
    # Longer synthetic document allows testing every possible boundary.
    n, size = 6, 40
    batch = {"ids": torch.arange(1, size + 1), "features": torch.randn(size, 64),
             "offsets": torch.tensor([0]), "lengths": torch.tensor([size]),
             "positions": torch.arange(size), "documents": torch.zeros(size, dtype=torch.long)}
    starts, docs = torch.tensor([n + 1]), torch.tensor([0])
    a1 = bonus_anchors(batch, starts, starts, docs, draft.hidden_size)
    h1 = draft.packed_hidden(batch["features"][None], batch["positions"], batch["documents"],
                             a1, starts, starts - 1, docs, include_bonus=True)
    first = draft.lm_head(h1[:, 1:]).argmax(-1)
    restart = starts + boundary
    a2 = bonus_anchors(batch, starts, restart, docs, draft.hidden_size, h1, first)
    assert a2.positions.tolist() == [list(range(n + 1 + boundary - 4, n + 1 + boundary))]
    assert a2.source[0, -1] == 2
    assert not a2.draft_features.requires_grad
    for slot, pos in enumerate(a2.positions[0]):
        if pos >= n:
            torch.testing.assert_close(a2.draft_features[0, slot], h1[0, pos - n].detach())
    expected = draft.packed_hidden(batch["features"][None], batch["positions"], batch["documents"],
                                   a2, restart, starts - 1, docs)
    with torch.inference_mode():
        fast = BonusInFourDraft(draft, 128)
        fast.cache.update("x", 0, batch["features"][:n][None])
        prior_ids = torch.cat((batch["ids"][n].reshape(1, 1), first), -1)
        got = fast.pass_hidden(batch["ids"][:n][None], n, restart[0], prior_ids, h1, fast.masks)[:, 1:]
        torch.testing.assert_close(got, expected, atol=3e-6, rtol=3e-5)
        def choose(logits):
            assert logits.shape[1] == 15
            scores = torch.zeros(1, 15)
            scores[:, boundary] = 1
            return scores
        monkeypatch.setattr("four_anchor.inference.boundary_scores", choose)
        candidates, keep = fast.propose(batch["ids"][:n][None], batch["ids"][n].reshape(1, 1))
        redraft = draft.lm_head(expected).argmax(-1)
        assert candidates.tolist() == torch.cat((first[:, :boundary], redraft[:, :15-boundary]), -1).tolist()
        assert candidates.shape == (1, 15) and keep.item() == boundary


@pytest.mark.parametrize("checkpointing", [False, True])
def test_zero_boundary_runs_both_views_and_cached_gradients_match(components, checkpointing, monkeypatch):
    target, _, _ = components
    draft = new_draft(components)
    batch = batch_for(components)
    draft.gradient_checkpointing = checkpointing
    clone = deepcopy(draft)
    cfg = tiny_config(num_proposals=15, anchor_layout="bonus_in_four", boundary_exploration=0.)
    monkeypatch.setattr("four_anchor.bonus_objective.rejection_boundary", lambda x: torch.zeros(len(x), dtype=torch.long))
    values = []
    for model, mode in ((draft, "reference"), (clone, "cached")):
        task = PackedTask(model, target.model.norm, replace(cfg, packed_execution=mode), random.Random(4))
        calls = []
        hook = model.layers[0].register_forward_pre_hook(lambda *args: calls.append(1))
        num, den, stats = task(batch)
        hook.remove()
        assert task.last_redraft_blocks == cfg.max_anchors
        assert stats[10] == 0
        if mode == "reference":
            assert len(calls) == 2 * 3  # three chunks, including the final single block
        (num / den).backward()
        assert model.draft_anchor_proj.weight.grad.abs().sum() > 0
        values.append((num, den, stats))
    for a, b in zip(*values):
        torch.testing.assert_close(a, b, atol=2e-5, rtol=2e-5)
    for (name, p), (_, q) in zip(draft.named_parameters(), clone.named_parameters()):
        if p.grad is not None:
            torch.testing.assert_close(p.grad, q.grad, atol=2e-5, rtol=3e-4, msg=name)


@torch.inference_mode()
def test_new_native_loop_emits_identical_greedy_tokens_with_one_target_forward(components):
    _, _, runner = components
    draft = new_draft(components)
    prompt = torch.tensor([[2, 3, 4, 5]])
    state = runner.prefill(prompt)
    fast = BonusInFourDraft(draft, 256)
    fast.cache.update("x", 0, state.features)
    bonus = state.next_logits.argmax(-1, keepdim=True)
    emitted = bonus[0].tolist()
    before = runner.calls
    for _ in range(6):
        candidates, _ = fast.propose(state.ids, bonus)
        n = state.ids.shape[1]
        inputs = torch.cat((bonus, candidates), -1)
        out, features = runner.forward(inputs, state.cache, all_logits=True)
        expected = out.logits[:, :-1].argmax(-1)
        bad = (expected != candidates)[0].nonzero()
        accepted = int(bad[0]) if bad.numel() else 15
        keep = accepted + 1
        bonus = out.logits[:, accepted].argmax(-1, keepdim=True)
        out.past_key_values.crop(n + keep)
        fast.cache.update("x", n, features[:, :keep])
        state.ids = torch.cat((state.ids, inputs[:, :keep]), -1)
        state.cache = out.past_key_values
        emitted += candidates[0, :accepted].tolist() + bonus[0].tolist()
    assert runner.calls - before == 6
    ar = runner.prefill(prompt)
    reference = []
    for _ in emitted:
        token = ar.next_logits.argmax(-1, keepdim=True)
        reference += token[0].tolist()
        ar = runner.append(ar, token)
    assert emitted == reference


def test_checkpoint_contract_and_portable_reload(components, tmp_path):
    target, old, _ = components
    with pytest.raises(ValueError):
        BonusInFourDraft(old, 128)
    new = new_draft(components)
    new.save(tmp_path)
    restored = FourAnchorDFlash.load(tmp_path, target)
    assert restored.cfg == new.cfg
    assert ServingSettings(str(tmp_path)).inference_version() == "bonus_in_four_cached_v2"
    with pytest.raises(ValueError):
        KnownBonusDraft(restored, 128)
    metadata = json.loads((tmp_path / "four_anchor_config.json").read_text())
    metadata["draft_config"]["format_version"] = 1
    (tmp_path / "four_anchor_config.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError):
        ServingSettings(str(tmp_path)).metadata()


def test_filter_requires_a_prefix_before_the_bonus(tmp_path):
    from datasets import Dataset
    from four_anchor.packed_data import prepare_records
    data = tmp_path / "data"
    Dataset.from_dict({"input_ids": [[1, 2], [1, 2, 3], [1, 2, 3]],
                       "loss_mask": [[0, 1], [0, 1, 0], [0, 0, 1]]}).save_to_disk(data)
    cfg = replace(tiny_config(num_proposals=15, anchor_layout="bonus_in_four"),
                  data=str(data), output=str(tmp_path / "out"))
    rows, lengths, summary = prepare_records(cfg)
    assert len(rows) == 1 and lengths.tolist() == [3] and summary["dropped_rows"] == 2
