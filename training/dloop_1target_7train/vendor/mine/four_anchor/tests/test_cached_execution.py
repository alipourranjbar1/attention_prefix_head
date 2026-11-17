from dataclasses import replace
import random

import pytest
import torch
from torch.nn import functional as F

from four_anchor.cached_loss import projected_loss, teacher_cache
from four_anchor.packed_objective import PackedTask, distribution_terms, packed_anchors
from four_anchor.tests.test_packed_training import batch_for, tiny_config


@pytest.mark.parametrize("mix", [{"ce": .1, "tv": .9}, {"ce": 1.}, {"tv": 1.},
                                  {"kl_div": 1.}, {"ce": .3, "tv": .2, "kl_div": .5}])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_frozen_head_first_order_loss_and_gradients_match_autograd(components, mix, dtype):
    target, draft, _ = components
    h = torch.randn(3, 4, 32, requires_grad=True)
    teacher = torch.randn(9, 32)
    indices = torch.randint(0, 9, (3, 4))
    weights = torch.rand(3, 4)
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=dtype == torch.bfloat16):
        cache = teacher_cache(teacher, target.model.norm, draft.lm_head, 3)
        actual, stats, tokens, logc = projected_loss(h, draft.lm_head, cache, indices, weights, mix, 5)
        logits = draft.lm_head(h)
        with torch.no_grad():
            teacher_logits = draft.lm_head(target.model.norm(teacher))[indices]
        terms = distribution_terms(logits.flatten(0, 1), teacher_logits.flatten(0, 1)).reshape(3, 4, 3)
        coefficients = torch.tensor([mix.get(k, 0.) for k in ("ce", "tv", "kl_div")])
        expected = ((terms * coefficients).sum(-1) * weights).sum(-1)
    torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-6)
    got_grad = torch.autograd.grad(actual.sum(), h, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected.sum(), h)[0]
    torch.testing.assert_close(got_grad, expected_grad, atol=2e-6 if dtype == torch.bfloat16 else 2e-8,
                               rtol=.02 if dtype == torch.bfloat16 else 1e-5)
    assert torch.equal(tokens, logits.argmax(-1))
    torch.testing.assert_close(logc, logits.float().log_softmax(-1).amax(-1))
    assert all(p.grad is None for p in target.parameters())


@pytest.mark.parametrize("checkpointing", [False, True])
def test_shared_context_cache_preserves_forward_and_all_parameter_gradients(components, checkpointing):
    _, draft, _ = components
    batch = batch_for(components)
    draft.gradient_checkpointing = checkpointing
    starts, docs = torch.tensor([1, 4, 6]), torch.tensor([0, 1, 0])
    anchors = packed_anchors(batch, starts, starts, docs, draft.hidden_size, 4)
    def run(memory=None):
        return draft.packed_hidden(batch["features"][None], batch["positions"], batch["documents"],
                                    anchors, starts, starts, docs, memory=memory)
    expected = run()
    (expected ** 2).sum().backward()
    gradients = {k: p.grad.clone() for k, p in draft.named_parameters() if p.grad is not None}
    draft.zero_grad(set_to_none=True)
    memory = draft.packed_memory(batch["features"][None], batch["positions"])
    actual = run(memory)
    (actual ** 2).sum().backward()
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
    for key, p in draft.named_parameters():
        if key in gradients:
            torch.testing.assert_close(p.grad, gradients[key], atol=1e-4, rtol=.01, msg=key)


def test_cached_objective_preserves_loss_and_gradients_and_projects_context_once(components):
    target, draft, _ = components
    cfg = tiny_config()
    batch = batch_for(components)
    task = PackedTask(draft, target.model.norm, cfg, random.Random(5))
    expected, denominator, metrics = task(batch)
    expected.backward()
    gradients = {k: p.grad.clone() for k, p in draft.named_parameters() if p.grad is not None}
    draft.zero_grad(set_to_none=True)
    task = PackedTask(draft, target.model.norm, replace(cfg, packed_execution="cached"), random.Random(5))
    calls = []
    handle = draft.fc.register_forward_pre_hook(lambda m, args: calls.append(tuple(args[0].shape)))
    actual, den2, metrics2 = task(batch)
    actual.backward()
    handle.remove()
    assert calls.count(tuple(batch["features"][None].shape)) == 1
    torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-6)
    torch.testing.assert_close(den2, denominator)
    for key, p in draft.named_parameters():
        if key in gradients:
            torch.testing.assert_close(p.grad, gradients[key], atol=3e-6, rtol=.001, msg=key)
    assert metrics2[12] == metrics[12] == 5


def test_only_positive_boundary_blocks_run_second_backbone(components):
    target, draft, runner = components
    prefix = torch.tensor([[2, 3, 4]])
    state = runner.prefill(prefix)
    from four_anchor.model import make_anchors
    first = draft(state.features, make_anchors(prefix, state.features, 4, draft.hidden_size), 3)
    from four_anchor.packed_data import TeacherRecords, collate_documents
    from datasets import Dataset
    ids = torch.cat([prefix, first.tokens.detach()], -1)[0].tolist()
    rows = Dataset.from_list([{"input_ids": ids, "loss_mask": [0] * 3 + [1] * 16}])
    cfg = replace(tiny_config(), max_anchors=3, blocks_per_chunk=3, boundary_exploration=1.)
    batch = collate_documents([TeacherRecords(rows, cfg, target.config, runner)[0]])
    class Fixed(random.Random):
        def sample(self, population, k): return [3, 4, 5]
        def choices(self, *args, **kw):
            self.counter = getattr(self, "counter", 0) + 1
            return [5 if self.counter == 1 else 0]
    reference = PackedTask(draft, target.model.norm, cfg, Fixed(5))
    a, _, _ = reference(batch)
    cached = PackedTask(draft, target.model.norm, replace(cfg, packed_execution="cached"), Fixed(5))
    b, _, stats = cached(batch)
    assert reference.last_redraft_blocks == 3
    assert cached.last_redraft_blocks == 1
    assert stats[12] == 3  # all three blocks still contribute both views
    torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)


def test_tv_equal_distributions_has_zero_gradient(components):
    _, draft, _ = components
    h = torch.zeros(1, 2, 32, requires_grad=True)
    p = torch.full((1, 64), 1 / 64)
    loss, _, _, _ = projected_loss(h, draft.lm_head, (p, torch.zeros(1, dtype=torch.long)),
                                   torch.zeros(1, 2, dtype=torch.long), torch.ones(1, 2), {"tv": 1.}, 1)
    loss.sum().backward()
    torch.testing.assert_close(h.grad, torch.zeros_like(h), atol=1e-7, rtol=0)
