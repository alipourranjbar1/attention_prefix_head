"""CPU checks for the native vLLM worker's position and fixed-window contract."""
from dataclasses import replace
import json

import pytest
import torch

from four_anchor.model import DraftOutput
from four_anchor.vllm_support import (
    FeatureHistory, ServingSettings, fixed_window_proposal, retained_forward_tokens,
)


@pytest.mark.parametrize("accepted", range(16))
def test_verification_keeps_previous_bonus_and_only_accepted_inputs(accepted):
    # 16 inputs = prior bonus + 15 proposals. a+1 sampled outputs include a
    # NEW bonus, whose hidden state is not present in these 16 input states.
    assert retained_forward_tokens(16, 15, accepted + 1) == accepted + 1


def test_prefill_keeps_whole_prompt_and_invalid_counts_fail():
    assert retained_forward_tokens(127, 0, 1) == 127
    with pytest.raises(ValueError):
        retained_forward_tokens(16, 15, 17)


def test_history_rejection_reprocessing_and_new_request():
    history = FeatureHistory(32, 2, "cpu", torch.float32)
    ids = torch.arange(8)
    features = ids[:, None].expand(-1, 2).float()
    history.update("a", 0, ids, features)
    # A separately materialized bonus is reprocessed in the next target call.
    history.update("a", 8, torch.tensor([40]), torch.tensor([[40., 40.]]))
    history.update("a", 8, torch.tensor([40, 41]), torch.tensor([[40., 40.], [41., 41.]]))
    # Rewind after a rejected suffix; no rejected state remains in prefix().
    history.update("a", 9, torch.tensor([50]), torch.tensor([[50., 50.]]))
    assert history.length == 10
    assert history.prefix()[0].tolist() == [list(range(8)) + [40, 50]]
    with pytest.raises(ValueError, match="Missing target features"):
        history.update("a", 11, torch.tensor([1]), torch.zeros(1, 2))
    history.update("b", 0, torch.tensor([7]), torch.tensor([[7., 7.]]))
    assert history.length == 1
    assert history.prefix()[1].tolist() == [[[7., 7.]]]


@pytest.mark.parametrize("boundary", [0, 1, 3, 4, 14])
def test_fixed_window_stays_fifteen_with_correct_second_anchors(components, monkeypatch, boundary):
    _target, drafter, runner = components
    drafter.cfg = replace(drafter.cfg, num_proposals=15)
    state = runner.prefill(torch.tensor([[2, 3, 4, 5, 6, 7]]))
    calls = []
    first_hidden = torch.arange(15 * drafter.hidden_size).reshape(1, 15, -1).float()

    def forward(features, anchors, start):
        calls.append((features.clone(), anchors, start))
        logits = torch.full((1, 15, 64), -100.)
        tokens = torch.arange(15) + (10 if len(calls) == 1 else 30)
        logits[0, torch.arange(15), tokens] = 100.
        return DraftOutput(logits, first_hidden if len(calls) == 1 else torch.zeros_like(first_hidden))

    monkeypatch.setattr(drafter, "forward", forward)
    monkeypatch.setattr("four_anchor.vllm_support.select_boundary", lambda _: boundary)
    result, chosen = fixed_window_proposal(drafter, state.ids, state.features)
    assert result.tolist() == [list(range(10, 10 + boundary)) + list(range(30, 45 - boundary))]
    assert result.shape == (1, 15)
    assert chosen == boundary and len(calls) == 2
    _, anchors, start = calls[1]
    assert start == 6 + boundary
    assert anchors.positions.tolist() == [list(range(start - 4, start))]
    for j, pos in enumerate(range(start - 4, start)):
        if pos < 6:
            assert anchors.source[0, j] == 1
            torch.testing.assert_close(anchors.target_features[0, j], state.features[0, pos])
        else:
            assert anchors.source[0, j] == 2
            torch.testing.assert_close(anchors.draft_features[0, j], first_hidden[0, pos - 6])
    # Second-pass conditioning never receives target states for proposed tokens.
    torch.testing.assert_close(calls[1][0], state.features)


def test_trained_checkpoint_remains_sixteen_when_serving_fifteen(components, tmp_path):
    _target, drafter, _runner = components
    drafter.save(tmp_path)
    config_path = tmp_path / "four_anchor_config.json"
    before = config_path.read_bytes()
    ServingSettings(str(tmp_path), num_proposals=15).apply(drafter)
    assert drafter.cfg.num_proposals == 15
    assert config_path.read_bytes() == before
    assert json.loads(before)["draft_config"]["num_proposals"] == 16
    with pytest.raises(ValueError, match="exceed"):
        ServingSettings(str(tmp_path), num_proposals=17).metadata()


@torch.inference_mode()
def test_real_tiny_model_fixed_window_verification_matches_target(components):
    _target, drafter, runner = components
    drafter.cfg = replace(drafter.cfg, num_proposals=15)
    ids = torch.tensor([[2, 3, 4, 5]])
    state = runner.prefill(ids)
    output = []
    for _ in range(5):
        candidates, _ = fixed_window_proposal(drafter, state.ids, state.features)
        assert candidates.shape == (1, 15)
        accepted, bonus, state = runner.verify(state, candidates)
        output.extend(candidates[0, :accepted].tolist() + bonus[0].tolist())
        assert accepted <= 15
    baseline = runner.prefill(ids)
    expected = []
    for _ in output:
        token = baseline.next_logits.argmax(-1, keepdim=True)
        expected.extend(token[0].tolist())
        baseline = runner.append(baseline, token)
    assert output == expected
