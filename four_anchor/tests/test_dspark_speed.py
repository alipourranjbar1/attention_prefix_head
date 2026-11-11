from copy import deepcopy
from dataclasses import replace
import random

import pytest
import torch

from four_anchor.dspark_loss import projected_dspark_loss
from four_anchor.packed_objective import PackedTask
from four_anchor.tests.test_dspark_multiround import setup
from four_anchor.tests.test_packed_training import batch_for


@pytest.mark.parametrize("amp", [False, True])
@pytest.mark.parametrize("chunk", [1, 7, 64])
@pytest.mark.parametrize("kl", [0., .2])
def test_cached_loss_matches_checkpoint_values_and_all_gradients(components, amp, chunk, kl):
    _, draft, _, _ = setup(components)
    other = deepcopy(draft)
    draft.head_loss_backend = "cached_vjp"
    h = torch.randn(2, 5, draft.hidden_size, requires_grad=True)
    h2 = h.detach().clone().requires_grad_(True)
    prev = torch.tensor([[1, 2, 1, 7, 8], [2, 2, 1, 4, 7]])
    p = torch.softmax(torch.randn(11, 64), -1)
    cache = p, p.argmax(-1)
    indices = torch.tensor([[1, 2, 4, 7, 8], [0, 2, 9, 4, 6]])
    weights = torch.tensor([[1., .8, 0., .4, .2], [.7, .6, .5, 0., 0.]])
    kwargs = (prev, cache, indices, weights, {"ce": .1, "tv": .9, "kl_div": kl}, .6, chunk)
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=amp):
        actual = projected_dspark_loss(draft, h, *kwargs)
        expected = projected_dspark_loss(other, h2, *kwargs)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, rtol=3e-5, atol=2e-6)
    # Exercise non-unit upstream scaling too (global supervised-weight normalization).
    (actual[0] * .37).backward()
    (expected[0] * .37).backward()
    # Cached VJPs move upstream scaling after BF16 rounding. Bound both
    # element error and whole-gradient relative error; FP32 remains tight.
    tol = dict(rtol=.025, atol=8e-4) if amp else dict(rtol=5e-5, atol=1e-7)
    torch.testing.assert_close(h.grad, h2.grad, **tol)
    for (name, a), (_, b) in zip(draft.named_parameters(), other.named_parameters()):
        assert (a.grad is None) == (b.grad is None), name
        if a.grad is not None:
            torch.testing.assert_close(a.grad, b.grad, msg=name, **tol)
            if amp:
                assert (a.grad - b.grad).norm() <= .01 * b.grad.norm() + 1e-7, name


@pytest.mark.parametrize("checkpointing", [False, True])
def test_full_two_round_objective_and_gradients(components, checkpointing):
    target, draft, _, cfg = setup(components)
    other = deepcopy(draft)
    draft.head_loss_backend = "cached_vjp"
    draft.gradient_checkpointing = checkpointing
    other.gradient_checkpointing = True
    batch = batch_for(components)
    results = []
    for model in (draft, other):
        task = PackedTask(model, target.model.norm,
                          replace(cfg, packed_execution="cached"), random.Random(43))
        out = task(batch)
        (out[0] / out[1]).backward()
        results.append(out)
    for a, b in zip(*results):
        torch.testing.assert_close(a, b, rtol=5e-5, atol=2e-6)
    for (name, a), (_, b) in zip(draft.named_parameters(), other.named_parameters()):
        if a.grad is not None:
            torch.testing.assert_close(a.grad, b.grad, rtol=5e-4, atol=3e-7, msg=name)


def test_zero_weights_and_kernel_overlap_ties(components):
    _, draft, _, _ = setup(components)
    other = deepcopy(draft)
    draft.head_loss_backend = "cached_vjp"
    h = torch.randn(3, draft.hidden_size, requires_grad=True)
    h2 = h.detach().clone().requires_grad_(True)
    prev = torch.tensor([1, 2, 3])
    with torch.no_grad():
        p = other.project(h2, prev)[0].float().log_softmax(-1).exp()
    cache = p, p.argmax(-1)
    args = prev, cache, torch.arange(3), torch.tensor([0., 1., 0.]), {"tv": 1.}, .7, 1
    x = projected_dspark_loss(draft, h, *args)
    y = projected_dspark_loss(other, h2, *args)
    x[0].backward()
    y[0].backward()
    torch.testing.assert_close(h.grad, h2.grad, rtol=1e-4, atol=2e-7)
    assert torch.count_nonzero(h.grad[[0, 2]]) == 0


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_cached_vjp_real_fsdp_api_checkpoint_continuation(tmp_path, dtype):
    import torch.distributed as dist
    import torch.testing._internal.distributed.fake_pg  # noqa: F401
    from four_anchor.fsdp_check import run_check
    torch.set_num_threads(1)
    dist.init_process_group("fake", rank=0, world_size=1)
    try:
        run_check(torch.device("cpu"), tmp_path, dtype, draft_type="dspark", head_loss_backend="cached_vjp")
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("amp", [False, True])
def test_batched_base_preserves_sequential_markov_proposals(components, amp):
    _, draft, _, _ = setup(components)
    hidden = torch.randn(7, 15, draft.hidden_size)
    seeds = torch.tensor([2, 7, 9, 3, 1, 8, 10])
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=amp):
        old = draft.greedy_proposals(hidden, seeds)
        draft.proposal_base_batched = True
        new = draft.greedy_proposals(hidden, seeds)
    assert torch.equal(new[0], old[0])
    for a, b in zip(new[1:], old[1:]):
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)


def test_benchmark_suppresses_training_checkpoints(components, tmp_path, monkeypatch):
    import json
    from datasets import Dataset
    import four_anchor.packed_train as pt
    target, _, _, cfg = setup(components)
    data = Dataset.from_dict({"input_ids": [[1, 2, 3, 4, 5, 6, 7, 8]] * 2,
                              "loss_mask": [[1] * 8] * 2})
    data_path = tmp_path / "data"
    data.save_to_disk(data_path)
    monkeypatch.setattr(pt, "load_target", lambda *a, **k: (target, None))
    cfg = replace(cfg, data=str(data_path), output=str(tmp_path / "trial"), max_steps=1,
                  benchmark_only=True, head_loss_backend="cached_vjp", proposal_base_batched=True,
                  packed_execution="cached", profile_steps=1, gradient_checkpointing=False)
    assert pt.train_packed(cfg) is None
    assert not list((tmp_path / "trial").glob("checkpoint-*"))
    row = json.loads((tmp_path / "trial/metrics.jsonl").read_text().splitlines()[0])
    assert row["max_rank_step_seconds"] > 0

