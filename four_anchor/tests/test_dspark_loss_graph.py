from copy import deepcopy
import pytest
import torch

from four_anchor.check_dspark_graph import eager, check
from four_anchor.dspark_loss_graph import distribution_loss, graph_status
from four_anchor.dspark_loss import projected_dspark_loss
from four_anchor.tests.test_dspark_multiround import setup


@pytest.mark.parametrize("count", [1, 5, 8])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("mix", [{"ce": .1, "tv": .9}, {"ce": .2, "kl_div": .8}, {"tv": 1.}])
def test_exact_first_order_loss_gradients(count, dtype, mix):
    torch.manual_seed(77)
    logits = torch.randn(count, 37, dtype=dtype).requires_grad_()
    c = torch.randn(count, dtype=dtype).requires_grad_()
    p = torch.randn(count, 37).softmax(-1)
    labels, w = p.argmax(-1), torch.rand(count)
    w[::3] = 0
    x, d = logits.detach().clone().requires_grad_(), c.detach().clone().requires_grad_()
    loss, stats, conf = distribution_loss(logits, c, p, labels, w, mix, .7, 8, False)
    expected = eager(x, d, p, labels, w, mix, .7)
    torch.testing.assert_close(loss, expected)
    assert not stats.requires_grad and not conf.requires_grad
    (loss * 2.3).backward()
    (expected * 2.3).backward()
    tol = dict(rtol=.015, atol=.004) if dtype == torch.bfloat16 else dict(rtol=1e-5, atol=1e-7)
    torch.testing.assert_close(logits.grad, x.grad, **tol)
    torch.testing.assert_close(c.grad, d.grad, **tol)


def test_equal_distributions_tie_gradient_and_saturated_confidence():
    x = torch.zeros(5, 8, requires_grad=True)
    c = torch.tensor([-100., -1., 0., 1., 100.], requires_grad=True)
    p, labels, w = torch.full((5, 8), .125), torch.zeros(5, dtype=torch.long), torch.ones(5)
    loss = distribution_loss(x, c, p, labels, w, {"tv": 1.}, 1., 8, False)[0]
    loss.backward()
    torch.testing.assert_close(x.grad, torch.zeros_like(x))
    torch.testing.assert_close(c.grad, c.detach().sigmoid() - 1.)


def test_multiple_outstanding_nodes_and_partial_padding():
    assert check("cpu", 8, 64, graph=False)["status"] == "passed"
    assert graph_status()["loss_graph_captures"] == 0


def test_graph_flag_does_not_silently_fall_back_on_cpu():
    with pytest.raises(ValueError, match="NPU tensors"):
        distribution_loss(torch.zeros(2, 7), torch.zeros(2), torch.full((2, 7), 1 / 7),
                          torch.zeros(2, dtype=torch.long), torch.ones(2), {"ce": 1.}, 1., 8, True)


@pytest.mark.parametrize("chunk", [1, 7, 64])
def test_custom_gradient_path_with_checkpointed_trainable_heads(components, monkeypatch, chunk):
    _, draft, _, _ = setup(components)
    other = deepcopy(draft)
    draft.loss_graph = "npu"
    # Exercise exactly the graph loss autograd bridge on CPU. Real capture is
    # separately checked by the on-NPU launcher; this is not an NPU test.
    def cpu_bridge(*args, **kw):
        kw["graph"] = False
        return distribution_loss(*args, **kw)
    monkeypatch.setattr("four_anchor.dspark_loss_graph.distribution_loss", cpu_bridge)
    h = torch.randn(2, 5, draft.hidden_size, requires_grad=True)
    h2 = h.detach().clone().requires_grad_()
    prev = torch.randint(0, 64, (2, 5))
    probabilities = torch.randn(15, 64).softmax(-1)
    cache = probabilities, probabilities.argmax(-1)
    indices = torch.randint(0, 15, (2, 5))
    weights = torch.rand(2, 5)
    mix = {"ce": .1, "tv": .9}
    actual = projected_dspark_loss(draft, h, prev, cache, indices, weights, mix, 1., chunk)
    expected = projected_dspark_loss(other, h2, prev, cache, indices, weights, mix, 1., chunk)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b)
    actual[0].backward()
    expected[0].backward()
    torch.testing.assert_close(h.grad, h2.grad, atol=1e-7, rtol=1e-5)
    for (name, p), (_, q) in zip(draft.named_parameters(), other.named_parameters()):
        if p.grad is not None:
            torch.testing.assert_close(p.grad, q.grad, atol=1e-7, rtol=1e-5, msg=name)
