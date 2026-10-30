from dataclasses import replace
import importlib
import json
import random

import numpy as np
import pytest
import torch
from datasets import Dataset
from safetensors.torch import load_file

from four_anchor.model import DraftOutput, make_anchors
from four_anchor.packed_data import TeacherRecords, collate_documents, epoch_batches
from four_anchor.packed_objective import PackedTask, distribution_terms, packed_anchors, rejection_boundary
from four_anchor.train import TrainConfig


def tiny_config(**kw):
    return TrainConfig("tiny", "data", "output", 63, [0, 1], device="cpu", dtype="float32",
                       num_draft_layers=2, training_mode="packed", epochs=2,
                       train_data_ratio=1, num_workers=0, max_anchors=5,
                       blocks_per_chunk=2, head_chunk_size=7, max_sequence_length=32,
                       loss_fn={"ce": .1, "tv": .9}, **kw)


def batch_for(components):
    target, draft, runner = components
    cfg = tiny_config()
    rows = Dataset.from_list([{"input_ids": list(range(1, 13)), "loss_mask": [0] + [1] * 11},
                              {"input_ids": list(range(18, 27)), "loss_mask": [0, 0] + [1] * 7}])
    dataset = TeacherRecords(rows, cfg, target.config, runner)
    return collate_documents([dataset[0], dataset[1]])


@pytest.mark.parametrize("checkpointing", [False, True])
def test_packed_forward_and_gradient_match_independent_blocks(components, checkpointing):
    _, draft, _ = components
    batch = batch_for(components)
    draft.gradient_checkpointing = checkpointing
    starts, docs = torch.tensor([1, 7, 4]), torch.tensor([0, 0, 1])
    anchors = packed_anchors(batch, starts, starts, docs, draft.hidden_size, 4)
    packed = draft.packed_hidden(batch["features"][None], batch["positions"], batch["documents"],
                                 anchors, starts, starts, docs)
    packed.square().sum().backward()
    expected_grad = draft.layers[0].self_attn.q_proj.weight.grad.clone()
    draft.zero_grad(set_to_none=True)
    independent = []
    for start, doc in zip(starts.tolist(), docs.tolist()):
        offset = batch["offsets"][doc]
        ids = batch["ids"][offset:offset + start][None]
        features = batch["features"][offset:offset + start][None]
        a = make_anchors(ids, features, 4, draft.hidden_size)
        independent.append(draft(features, a, start).hidden[0])
    independent = torch.stack(independent)
    independent.square().sum().backward()
    torch.testing.assert_close(packed, independent, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(expected_grad, draft.layers[0].self_attn.q_proj.weight.grad, atol=5e-5, rtol=1e-3)


@pytest.mark.parametrize("cached", [False, True])
def test_second_pass_uses_draft_states_and_cannot_read_future_or_other_documents(components, cached):
    _, draft, _ = components
    batch = batch_for(components)
    starts, docs, keep = torch.tensor([3]), torch.tensor([0]), 5
    a = packed_anchors(batch, starts, starts, docs, draft.hidden_size, 4)
    first = draft.packed_hidden(batch["features"][None], batch["positions"], batch["documents"], a, starts, starts, docs)
    tokens = draft.lm_head(first).argmax(-1)
    a2 = packed_anchors(batch, starts, starts + keep, docs, draft.hidden_size, 4, first, tokens)
    assert a2.source.tolist() == [[2, 2, 2, 2]]
    assert a2.positions.tolist() == [[4, 5, 6, 7]]
    assert not a2.draft_features.requires_grad
    def run(features):
        memory = draft.packed_memory(features[None], batch["positions"]) if cached else None
        return draft.packed_hidden(features[None], batch["positions"], batch["documents"], a2, starts + keep, starts, docs, memory=memory)
    expected = run(batch["features"])
    modified = batch["features"].clone()
    modified[3:] = torch.randn_like(modified[3:]) * 100
    torch.testing.assert_close(run(modified), expected, atol=0, rtol=0)
    expected.square().sum().backward()
    assert draft.draft_anchor_proj.weight.grad.abs().sum() > 0
    # Compare with the unmodified single-block inference path.
    first_output = DraftOutput(draft.lm_head(first), first)
    single = make_anchors(batch["ids"][:3][None], batch["features"][:3][None], 4, draft.hidden_size, first_output, keep)
    torch.testing.assert_close(a2.draft_features, single.draft_features)
    torch.testing.assert_close(expected, draft(batch["features"][:3][None], single, 8).hidden, atol=2e-6, rtol=2e-6)


def test_dspark_loss_values_and_gradients_match_upstream():
    from speculators.losses.eager import ce_loss, tv_loss, kl_div_loss
    torch.manual_seed(32)
    logits = torch.randn(2, 5, 64, requires_grad=True)
    target = torch.randn_like(logits)
    actual = distribution_terms(logits.flatten(0, 1), target.flatten(0, 1)).reshape(2, 5, 3)
    expected = torch.stack([ce_loss(logits, target), tv_loss(logits, target), kl_div_loss(logits, target)], -1)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
    mix = torch.tensor([.1, .9, 0.])
    a = torch.autograd.grad((actual * mix).sum(), logits, retain_graph=True)[0]
    b = torch.autograd.grad((expected * mix).sum(), logits)[0]
    torch.testing.assert_close(a, b, atol=2e-7, rtol=2e-6)


def test_teacher_target_alignment_and_weighted_chunk_loss(components):
    target, draft, _ = components
    batch = batch_for(components)
    with torch.no_grad():
        direct = target(input_ids=batch["ids"][:12][None]).logits[0, :-1]
        recovered = target.lm_head(target.model.norm(batch["final"][:11]))
    torch.testing.assert_close(direct, recovered, atol=0, rtol=0)
    cfg = tiny_config()
    task = PackedTask(draft, target.model.norm, cfg, random.Random(2))
    hidden = torch.randn(2, 16, draft.hidden_size, requires_grad=True)
    teacher = torch.randn_like(hidden)
    weights = torch.rand(2, 16)
    value, stats, tokens, confidence = task.loss_and_predictions(hidden, teacher, weights)
    logits = draft.lm_head(hidden)
    gold = target.lm_head(target.model.norm(teacher))
    terms = distribution_terms(logits.flatten(0, 1), gold.flatten(0, 1)).reshape(2, 16, 3)
    expected = ((.1 * terms[..., 0] + .9 * terms[..., 1]) * weights).sum()
    torch.testing.assert_close(value, expected)
    torch.testing.assert_close(torch.autograd.grad(value, hidden, retain_graph=True)[0],
                               torch.autograd.grad(expected, hidden)[0])
    assert torch.equal(tokens, logits.argmax(-1))
    from four_anchor.model import boundary_scores
    assert torch.equal(rejection_boundary(confidence), boundary_scores(logits).argmax(-1))


def test_packing_covers_rows_once_and_distributes_equal_step_counts(tmp_path):
    cfg = replace(tiny_config(), data_cache_dir=str(tmp_path))
    lengths = np.array([5, 7, 9, 3, 16, 6, 2, 12, 8, 4, 4, 6])
    plans = [epoch_batches(cfg, lengths, "example", 2, rank, 0) for rank in range(2)]
    assert len(plans[0]) == len(plans[1]) < len(lengths) // 2
    visited = [int(i) for plan in plans for batch in plan for i in batch]
    assert len(visited) == len(set(visited))
    assert len(lengths) - len(visited) < 2
    assert all(lengths[b].sum() <= 32 for plan in plans for b in plan)


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize("execution", ["reference", "cached"])
def test_actual_packed_two_epoch_training_and_exact_resume(components, tmp_path, monkeypatch, dtype, execution):
    target, _, _ = components
    if dtype == "bfloat16":
        target.to(torch.bfloat16)
    module = importlib.import_module("four_anchor.packed_train")
    data_module = importlib.import_module("four_anchor.packed_data")
    rows = Dataset.from_list([{"input_ids": [i + 1, 3, 4, 5, 6, 7, 8, 9], "loss_mask": [0, 0] + [1] * 6}
                              for i in range(9)])
    monkeypatch.setattr(data_module, "load_records", lambda *a: rows)
    monkeypatch.setattr(module, "load_target", lambda *a: (target, None))
    cfg = replace(tiny_config(), dtype=dtype, output=str(tmp_path / "full"), num_proposals=4,
                  max_anchors=4, gradient_accumulation=2, checkpoint_each_epoch=True,
                  warmup_steps=0, progress=False, data_cache_dir=str(tmp_path / "cache"), packed_execution=execution,
                  profile_steps=1)
    full = module.train_packed(cfg)
    state = torch.load(full + "/trainer.pt", weights_only=True)
    assert state["steps_per_epoch"] == [2, 2]
    early = module.train_packed(replace(cfg, output=str(tmp_path / "early"), max_steps=1))
    resumed = module.train_packed(replace(cfg, output=str(tmp_path / "resume"), resume=early))
    a, b = load_file(full + "/model.safetensors"), load_file(resumed + "/model.safetensors")
    assert all(torch.equal(a[k], b[k]) for k in a)
    from four_anchor.model import FourAnchorDFlash
    from four_anchor.runtime import Decoder, TargetRunner
    restored = FourAnchorDFlash.load(resumed, target).eval()
    runner = TargetRunner(target, [0, 1])
    try:
        decoder = Decoder(runner, restored)
        prefix = torch.tensor([[1, 2, 3]])
        assert decoder.generate(prefix, 8, "ar").token_ids == decoder.generate(prefix, 8, "two_pass").token_ids
    finally:
        runner.close()


def test_packed_training_loss_and_no_target_gradients(components):
    target, draft, _ = components
    task = PackedTask(draft, target.model.norm, tiny_config(), random.Random(13))
    draft.gradient_checkpointing = True
    loss, denom, metrics = task(batch_for(components))
    (loss / denom).backward()
    assert torch.isfinite(loss) and metrics[-1] == 5
    assert draft.layers[0].mlp.down_proj.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in target.parameters())


def test_nonzero_redraft_training_uses_branch_correct_teacher_and_updates_bridge(components):
    target, draft, runner = components
    prefix = torch.tensor([[2, 3, 4]])
    state = runner.prefill(prefix)
    first = draft(state.features, make_anchors(prefix, state.features, 4, draft.hidden_size), 3)
    ids = torch.cat([prefix, first.tokens.detach()], -1)[0].tolist()
    cfg = replace(tiny_config(), max_anchors=1, boundary_exploration=1.)
    rows = Dataset.from_list([{"input_ids": ids, "loss_mask": [0] * 3 + [1] * 16}])
    batch = collate_documents([TeacherRecords(rows, cfg, target.config, runner)[0]])
    class Fixed(random.Random):
        def sample(self, population, k):
            return [3]
        def choices(self, *a, **kw):
            return [5]
    task = PackedTask(draft, target.model.norm, cfg, Fixed(4))
    loss, denom, stats = task(batch)
    assert stats[10] == 5
    (loss / denom).backward()
    assert draft.draft_anchor_proj.weight.grad.abs().sum() > 0
    original = loss.detach().clone()
    draft.zero_grad(set_to_none=True)
    # Target loss labels stay intact; only unavailable future *input* states change.
    batch["features"][3:] = torch.randn_like(batch["features"][3:]) * 20
    changed, _, _ = task(batch)
    torch.testing.assert_close(changed, original, atol=0, rtol=0)
