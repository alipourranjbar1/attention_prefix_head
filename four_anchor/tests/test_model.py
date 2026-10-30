import json
import random

import pytest
import torch

from four_anchor.model import DraftOutput, FourAnchorDFlash, boundary_scores, make_anchors, select_boundary
from four_anchor.train import TrainConfig, TrainingTask


def test_boundary_matches_direct_dloop_formula():
    torch.manual_seed(8)
    logits = torch.randn(1, 16, 64) * 4
    confidence = logits.softmax(-1).amax(-1)
    direct = torch.cat([torch.ones_like(confidence[:, :1]), confidence.cumprod(-1)[:, :-1]], 1) * (1 - confidence)
    assert torch.allclose(boundary_scores(logits).exp(), direct, atol=1e-7)
    assert select_boundary(logits) == int(direct.argmax())


@pytest.mark.parametrize("boundary", [0, 1, 3, 4, 9, 15])
@pytest.mark.parametrize("length", [1, 3, 4, 9])
def test_exact_preceding_positions_and_state_sources(boundary, length):
    prefix = torch.arange(1, length + 1)[None]
    target = torch.arange(length * 6).view(1, length, 6).float()
    logits = torch.full((1, 16, 64), -100.0)
    tokens = torch.arange(20, 36)[None]
    logits.scatter_(-1, tokens[..., None], 100.0)
    hidden = torch.arange(16 * 4).view(1, 16, 4).float().requires_grad_()
    first = DraftOutput(logits, hidden)
    a = make_anchors(prefix, target, 4, 4, first, boundary)
    for i, position in enumerate(range(length + boundary - 4, length + boundary)):
        assert a.positions[0, i] == max(0, position)
        if position < 0:
            assert a.source[0, i] == 0
        elif position < length:
            assert a.source[0, i] == 1
            assert a.ids[0, i] == prefix[0, position]
            assert torch.equal(a.target_features[0, i], target[0, position])
        else:
            assert a.source[0, i] == 2
            assert a.ids[0, i] == tokens[0, position - length]
            assert torch.equal(a.draft_features[0, i], hidden[0, position - length])
    assert not a.draft_features.requires_grad


def test_twenty_input_positions_sixteen_outputs_and_no_future_features(components):
    _, draft, runner = components
    state = runner.prefill(torch.tensor([[1, 2, 3, 4, 5]]))
    anchors = make_anchors(state.ids, state.features, 4, draft.hidden_size)
    seen = []
    handle = draft.layers[0].register_forward_pre_hook(
        lambda _m, _a, kw: seen.append(kw["hidden_states"].shape[1]), with_kwargs=True)
    out = draft(state.features, anchors, 5)
    handle.remove()
    assert seen == [20]
    assert out.logits.shape == (1, 16, 64)
    assert out.hidden.shape == (1, 16, 32)
    with pytest.raises(ValueError, match="leak"):
        draft(state.features, anchors, 4)
    assert type(draft.layers[0]).__name__ == "Qwen3DFlashDecoderLayer"
    assert not any("prefix" in name.lower() or "mtp" in name.lower() for name, _ in draft.named_modules())


def test_checkpoint_roundtrip_and_strict_validation(components, tmp_path):
    target, draft, runner = components
    state = runner.prefill(torch.tensor([[1, 2, 3]]))
    a = make_anchors(state.ids, state.features, 4, draft.hidden_size)
    before = draft(state.features, a, 3).logits
    draft.save(tmp_path)
    restored = FourAnchorDFlash.load(tmp_path, target)
    after = restored(state.features, a, 3).logits
    torch.testing.assert_close(before, after, atol=0, rtol=0)
    metadata = json.loads((tmp_path / "four_anchor_config.json").read_text())
    metadata["target_signature"]["vocab_size"] = 999
    (tmp_path / "four_anchor_config.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="architectures"):
        FourAnchorDFlash.load(tmp_path, target)


def test_training_updates_shared_backbone_and_state_bridge_without_target_grad(components):
    target, draft, runner = components
    cfg = TrainConfig("tiny", "unused", "unused", 63, [0, 1], num_draft_layers=2,
                      device="cpu", dtype="float32", boundary_exploration=1)
    class FixedBoundary(random.Random):
        def choices(self, *args, **kwargs):
            return [5]
    task = TrainingTask(draft, runner, cfg, FixedBoundary(1))
    draft.gradient_checkpointing = True
    draft.train()
    before = draft.draft_anchor_proj.weight.detach().clone()
    optimizer = torch.optim.AdamW([p for p in draft.parameters() if p.requires_grad], lr=.001)
    prefix = torch.tensor([[2, 3, 4, 5]])
    state = runner.prefill(prefix)
    labels, valid = runner.rollout(state, 16)
    contexts = []
    hook = draft.register_forward_pre_hook(lambda _m, inputs: contexts.append(inputs[0].shape[1]))
    loss, metrics = task(prefix, labels, valid)
    hook.remove()
    loss.backward()
    assert torch.isfinite(loss)
    assert metrics["boundary"] == 5
    assert contexts == [4, 4]  # second teacher's extra states NEVER go to the drafter
    assert draft.layers[0].self_attn.q_proj.weight.grad.abs().sum() > 0
    assert draft.draft_anchor_proj.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in target.parameters())
    optimizer.step()
    assert not torch.equal(before, draft.draft_anchor_proj.weight)


def test_reject_prefix_checkpoint(components, tmp_path):
    _, draft, _ = components
    (tmp_path / "config.json").write_text(json.dumps({"speculators_model_type": "dflash_prefix"}))
    with pytest.raises(ValueError, match="regular"):
        draft.initialize_regular_dflash(tmp_path)


def test_confidence_extremes():
    logits = torch.zeros(1, 16, 4)
    logits[:, :, 0] = 1000
    assert select_boundary(logits) == 0
    assert not torch.isnan(boundary_scores(logits)).any()


def test_regular_upstream_checkpoint_warm_start(components, tmp_path):
    from copy import deepcopy
    from speculators import DFlashDraftModel, DFlashSpeculatorConfig
    target, draft, _ = components
    config = deepcopy(target.config)
    config._attn_implementation = "sdpa"
    upstream = DFlashDraftModel(DFlashSpeculatorConfig(
        transformer_layer_config=config, draft_vocab_size=64, block_size=17,
        aux_hidden_state_layer_ids=[0, 1], mask_token_id=63,
        speculators_config={"algorithm": "dflash", "default_proposal_method": "greedy",
                            "proposal_methods": [{"proposal_type": "greedy", "speculative_tokens": 16}],
                            "verifier": {"name_or_path": None, "architectures": ["Qwen3ForCausalLM"]}}))
    upstream.save_pretrained(tmp_path)
    bridge_before = draft.draft_anchor_proj.weight.detach().clone()
    draft.initialize_regular_dflash(tmp_path)
    for name, value in upstream.state_dict().items():
        if name.startswith(("layers.", "norm.", "fc.", "hidden_norm.")):
            torch.testing.assert_close(draft.state_dict()[name], value, atol=0, rtol=0)
    assert torch.equal(bridge_before, draft.draft_anchor_proj.weight)
