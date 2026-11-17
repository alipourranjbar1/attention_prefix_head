import copy

import pytest
import torch

from ascend_support.chunked_ce import enable_chunked_ce
from test_native import wrapper


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("offset", [1, 46])
def test_weighted_loss_stats_and_gradients_match_dense(dtype, offset):
    torch.manual_seed(91)
    dense = wrapper().to(dtype=dtype)
    chunked = enable_chunked_ce(copy.deepcopy(dense), 7)
    ids = torch.randint(0, 66, (2, 100))
    mask = torch.ones_like(ids).float()
    mask[:, 17:25] = 0
    mask[:, 30:38] = 0.25
    anchors = torch.tensor([[4, 26, 70], [8, 33, 80]])
    keep = torch.tensor([[True, False, True], [True, True, True]])
    x = torch.randn(2, 3, 15, 32, dtype=dtype, requires_grad=True)
    y = x.detach().clone().requires_grad_(True)
    a = dense._block_loss(x, ids, mask, anchors, keep, offset)
    b = chunked._block_loss(y, ids, mask, anchors, keep, offset)
    for av,bv in zip(a,b):torch.testing.assert_close(av,bv,rtol=1e-5,atol=2e-5)
    (a[0] / (a[1]+1e-6)).backward()
    (b[0] / (b[1]+1e-6)).backward()
    torch.testing.assert_close(x.grad,y.grad,rtol=1e-5,atol=2e-6)


@pytest.mark.parametrize("depth", [1,2,3])
def test_entire_loop_and_carrier_gradients_with_layer_recompute(depth):
    torch.manual_seed(93)
    dense=wrapper()
    chunked=enable_chunked_ce(copy.deepcopy(dense),7)
    chunked.draft_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
    dense.train();chunked.train()
    ids=torch.randint(0,66,(1,110))
    mask=torch.ones_like(ids);mask[:,14:19]=0
    hidden=torch.randn(1,110,64)
    anchors=torch.tensor([[8,17,31]])
    keep=torch.ones_like(anchors,dtype=torch.bool)
    for model in (dense,chunked):
        model._sample_anchor_positions=lambda *args:(anchors,keep)
        model._sample_loop_depth=lambda:depth
    a=dense(ids,hidden,mask,position_offsets=torch.tensor([37]))
    b=chunked(ids,hidden,mask,position_offsets=torch.tensor([37]))
    torch.testing.assert_close(a[0],b[0]);torch.testing.assert_close(a[1],b[1])
    for key in a[2]:torch.testing.assert_close(a[2][key],b[2][key])
    a[0].backward();b[0].backward()
    for ap,bp in zip(dense.draft_model.parameters(),chunked.draft_model.parameters()):
        torch.testing.assert_close(ap.grad,bp.grad,rtol=1e-4,atol=2e-6)


def test_loss_does_not_retain_vocabulary_sized_tensors_for_backward():
    model=enable_chunked_ce(wrapper(),7)
    ids=torch.randint(0,66,(1,100));mask=torch.ones_like(ids)
    anchors=torch.tensor([[1,20,40]]);keep=torch.ones_like(anchors,dtype=torch.bool)
    rows=torch.randn(1,3,15,32,requires_grad=True)
    saved=[]
    def pack(t):saved.append(tuple(t.shape));return t
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t:t):
        loss=model._block_loss(rows,ids,mask,anchors,keep,1)[0]
    assert not any(len(shape)==2 and shape[-1]==67 for shape in saved),saved
    loss.backward()
    assert rows.grad is not None and torch.isfinite(rows.grad).all()


def test_zero_weight_loss_retains_zero_gradient_and_eval_works():
    model=enable_chunked_ce(wrapper(),7)
    ids=torch.randint(0,66,(1,100));mask=torch.zeros_like(ids)
    anchors=torch.tensor([[1,20,40]]);keep=torch.ones_like(anchors,dtype=torch.bool)
    rows=torch.randn(1,3,15,32,requires_grad=True)
    stats=model._block_loss(rows,ids,mask,anchors,keep,1)
    assert all(v.item()==0 for v in stats)
    stats[0].backward()
    assert torch.count_nonzero(rows.grad)==0
    with torch.no_grad():
        again=model._block_loss(rows,ids,mask,anchors,keep,1)
        assert all(v.item()==0 for v in again)
