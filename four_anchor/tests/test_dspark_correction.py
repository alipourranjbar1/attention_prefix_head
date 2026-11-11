from dataclasses import replace
import random

import pytest
import torch
from torch.nn import functional as F

from four_anchor.tests.test_dspark_multiround import setup
from four_anchor.tests.test_packed_training import batch_for
from four_anchor.packed_objective import PackedTask
from four_anchor.dspark_train import DSparkTrainConfig


@pytest.mark.parametrize("amp", [False, True])
@pytest.mark.parametrize("hidden_bfloat16", [False, True])
def test_proposal_optimization_matches_original_and_updates_are_not_cached(components, amp, hidden_bfloat16):
    _, draft, _, _ = setup(components)
    if hidden_bfloat16 and not amp:
        draft.bfloat16()
    hidden = torch.randn(3, 15, draft.hidden_size)
    if hidden_bfloat16:
        hidden = hidden.bfloat16()
    seeds = torch.tensor([2, 7, 9])

    def reference():
        tokens, probs, confidence = [], [], []
        previous = seeds
        with torch.no_grad():
            for pos in range(15):
                logits, c = draft.project(hidden[:, pos], previous)
                previous = logits.argmax(-1)
                tokens.append(previous)
                probs.append(F.log_softmax(logits.float(), -1).amax(-1))
                confidence.append(F.logsigmoid(c.float()))
        return tuple(torch.stack(x, 1) for x in (tokens, probs, confidence))

    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=amp):
        expected = reference()
        actual = draft.greedy_proposals(hidden, seeds)
        reduced = draft.greedy_proposals(hidden, seeds, need_token_probabilities=False)
        for a, b in zip(expected, actual):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        assert reduced[1] is None
        torch.testing.assert_close(actual[0], reduced[0], rtol=0, atol=0)
        torch.testing.assert_close(actual[2], reduced[2], rtol=0, atol=0)
        with torch.no_grad():
            draft.confidence_head.proj.bias.add_(1.)
            draft.markov_head.markov_w2.weight.add_(.4)
        updated = draft.greedy_proposals(hidden, seeds)
        for a, b in zip(reference(), updated):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        assert not torch.equal(actual[2], updated[2])


def test_confidence_proposals_do_not_normalize_vocabulary(components, monkeypatch):
    _, draft, _, _ = setup(components)
    def forbidden(*args, **kwargs):
        raise AssertionError("confidence-based proposal selection must not compute vocabulary log-softmax")
    monkeypatch.setattr(F, "log_softmax", forbidden)
    tokens, probs, confidence = draft.greedy_proposals(
        torch.randn(2, 15, draft.hidden_size), torch.tensor([3, 5]), need_token_probabilities=False)
    assert probs is None and tokens.shape == confidence.shape == (2, 15)


@pytest.mark.parametrize("selector", ["confidence_head", "draft_probability"])
def test_objective_uses_selected_scores_and_cap_with_draft_anchors(components, monkeypatch, selector):
    import four_anchor.dspark_objective as obj
    from four_anchor.packed_data import TeacherRecords, collate_documents
    from datasets import Dataset
    target, draft, runner, cfg = setup(components)
    cfg = replace(cfg, max_anchors=1, max_sequence_length=64,
                  boundary_confidence=selector, boundary_exploration=0)
    records = Dataset.from_dict({"input_ids": [list(range(1, 41))], "loss_mask": [[1] * 40]})
    batch = collate_documents([TeacherRecords(records, cfg, target.config, runner)[0]])
    class Fixed(random.Random):
        def sample(self, population, k):
            return [5]
    q = torch.full((1, 15), -.8)
    head = torch.full((1, 15), -.1)
    # The first five tokens match gold, then the chain differs. A predicted
    # seven-token prefix must therefore be capped at five.
    tokens = batch["ids"][5:20][None].clone()
    tokens[:, 5] = 0
    def proposals(h, seed, *, need_token_probabilities=True):
        assert need_token_probabilities == (selector == "draft_probability")
        return tokens, q if need_token_probabilities else None, head
    def boundary(logc):
        torch.testing.assert_close(logc, head if selector == "confidence_head" else q)
        return torch.tensor([7])
    original, anchors = obj.redraft_anchors, []
    def collect(*args):
        value = original(*args)
        anchors.append(value)
        return value
    monkeypatch.setattr(draft, "greedy_proposals", proposals)
    monkeypatch.setattr(obj, "rejection_boundary", boundary)
    monkeypatch.setattr(obj, "redraft_anchors", collect)
    _, _, metrics = PackedTask(draft, target.model.norm, cfg, Fixed())(batch)
    assert metrics[10] == 5 and metrics[11] == 7
    assert len(anchors) == 1 and (anchors[0].source == 2).all()
    assert not anchors[0].target_features.any()
    assert not anchors[0].draft_features.requires_grad


def test_default_recipe_uses_head_and_graph_off():
    cfg = DSparkTrainConfig(target="model", data="data", output="output",
                           mask_token_id=63, target_layer_ids=[0, 1])
    assert cfg.boundary_confidence == "confidence_head" and cfg.loss_graph == "off"
