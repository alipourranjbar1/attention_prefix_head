import pytest
import torch

from four_anchor.runtime import Decoder
from four_anchor.model import DraftOutput


@pytest.mark.parametrize("accepted", [0, 1, 4, 15, 16])
def test_verification_crops_cache_and_materializes_correct_hidden_states(components, accepted):
    _, draft, runner = components
    prompt = torch.tensor([[1, 7, 3, 9, 2]])
    gold = Decoder(runner, draft).generate(prompt, 17, "ar", eos_ids=[]).token_ids
    candidate = torch.tensor([gold[:16]])
    if accepted < 16:
        candidate[0, accepted] = (candidate[0, accepted] + 1) % 64
    state = runner.prefill(prompt)
    actual, correction, state = runner.verify(state, candidate)
    assert actual == accepted
    assert correction.item() == gold[accepted]
    assert state.cache.get_seq_length() == prompt.shape[1] + accepted + 1
    expected_ids = torch.cat([prompt, torch.tensor([gold[:accepted + 1]])], 1)
    assert torch.equal(state.ids, expected_ids)
    fresh = runner.prefill(expected_ids)
    torch.testing.assert_close(state.next_logits, fresh.next_logits, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(state.features, fresh.features, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("prompt_length", [1, 3, 4, 9])
@pytest.mark.parametrize("limit", [0, 1, 17, 39])
def test_two_pass_and_one_pass_match_target_greedy(components, prompt_length, limit):
    target, draft, runner = components
    prompt = torch.arange(1, prompt_length + 1)[None]
    decoder = Decoder(runner, draft)
    reference = decoder.generate(prompt, limit, "ar", eos_ids=[])
    for mode in ["one_pass", "two_pass"]:
        result = decoder.generate(prompt, limit, mode, eos_ids=[])
        assert result.token_ids == reference.token_ids
        assert len(result.token_ids) == limit
        assert all(c.draft_calls == (2 if mode == "two_pass" else 1) for c in result.cycles)
        assert sum(c.committed for c in result.cycles) == limit
    if limit:
        generated = target.generate(prompt, max_new_tokens=limit, do_sample=False, eos_token_id=None,
                                    pad_token_id=0, use_cache=True)[0, prompt_length:].tolist()
        assert generated == reference.token_ids


def test_eos_stops_even_inside_verified_block(components):
    _, draft, runner = components
    prompt = torch.tensor([[5, 6, 7, 8]])
    decoder = Decoder(runner, draft)
    gold = decoder.generate(prompt, 16, "ar", eos_ids=[]).token_ids
    eos = gold[3]
    expected = gold[:gold.index(eos) + 1]
    for mode in ["ar", "one_pass", "two_pass"]:
        assert decoder.generate(prompt, 30, mode, eos_ids=[eos]).token_ids == expected


def test_no_target_call_between_draft_passes(components):
    _, draft, runner = components
    calls = []
    handle = draft.register_forward_pre_hook(lambda _m, _inputs: calls.append(runner.calls))
    result = Decoder(runner, draft).generate(torch.tensor([[1, 2, 3, 4]]), 7)
    handle.remove()
    assert len(calls) == 2 * len(result.cycles)
    assert all(calls[i] == calls[i + 1] for i in range(0, len(calls), 2))


def test_nonzero_restart_preserves_prefix_and_extends_by_sixteen(components, monkeypatch):
    _, draft, runner = components
    prompt = torch.tensor([[1, 2, 3, 4, 5, 6]])
    gold = Decoder(runner, draft).generate(prompt, 21, "ar", eos_ids=[]).token_ids
    original = draft.forward
    starts, anchors_seen = [], []
    def scripted(context, anchors, start):
        actual = original(context, anchors, start)
        starts.append(start)
        anchors_seen.append(anchors)
        offset = start - prompt.shape[1]
        tokens = torch.tensor([gold[offset:offset + 16]])
        # First five confidence=.99, sixth=.1: D-Loop chooses l=5.
        confidence = torch.full((1, 16), .99)
        if offset == 0:
            confidence[:, 5:] = .1
        probability = ((1 - confidence) / 63)[..., None].expand(-1, -1, 64).clone()
        probability.scatter_(-1, tokens[..., None], confidence[..., None])
        return DraftOutput(probability.log(), actual.hidden)
    monkeypatch.setattr(draft, "forward", scripted)
    out = Decoder(runner, draft).generate(prompt, 21, eos_ids=[])
    assert out.token_ids == gold
    assert starts == [6, 11]
    assert anchors_seen[1].positions.tolist() == [[7, 8, 9, 10]]
    assert anchors_seen[1].source.tolist() == [[2, 2, 2, 2]]
    assert anchors_seen[1].ids.tolist() == [gold[1:5]]
    assert out.cycles[0].boundary == 5
    assert out.cycles[0].proposed == 21
    assert out.cycles[0].accepted_draft == 21
    # No unused correction call is needed at the output budget.
    assert out.target_calls == 2  # prefill + verification
