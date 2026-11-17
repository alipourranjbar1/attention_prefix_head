import copy
import os
from pathlib import Path
import pytest
import torch
from torch.nn import functional as F
from transformers import Qwen3Config, Qwen3ForCausalLM
from pair.fast_attention import make_mask, block_sdpa
from pair.common import quarter_plan
from pair.evaluate import answer_number,generate_mine,greedy_reference
from specforge.core.dflash import create_dflash_sdpa_mask,OnlineDFlashModel
from specforge.modeling.draft.dflash import DFlashDraftModel


@pytest.mark.parametrize('carriers',[0,14,29,44])
@pytest.mark.parametrize('chunk',[1,3])
def test_block_attention_output_and_all_gradients(carriers,chunk,monkeypatch):
    monkeypatch.setenv('ATTN_BLOCK_CHUNK',str(chunk))
    torch.manual_seed(5)
    b,n,s,r,h,hk,d=2,3,9,4,4,2,8
    anchors=torch.tensor([[1,4,8],[2,5,0]])
    keep=torch.tensor([[True,True,True],[True,True,False]])
    q=torch.randn(b,h,n*r,d,requires_grad=True)
    k=torch.randn(b,hk,s+n*carriers+n*r,d,requires_grad=True)
    v=torch.randn_like(k,requires_grad=True)
    qq,kk,vv=[x.detach().clone().requires_grad_() for x in (q,k,v)]
    mask=create_dflash_sdpa_mask(anchors,keep,s,r,'cpu',carriers)
    ref=F.scaled_dot_product_attention(q,k.repeat_interleave(h//hk,1),v.repeat_interleave(h//hk,1),
                                      attn_mask=mask,scale=d**-0.5).transpose(1,2).contiguous()
    actual,_=block_sdpa(None,qq,kk,vv,make_mask(anchors,keep,s,r,'cpu',carriers),scaling=d**-0.5)
    torch.testing.assert_close(ref,actual,atol=2e-6,rtol=2e-5)
    upstream=torch.randn_like(ref)
    (ref*upstream).sum().backward();(actual*upstream).sum().backward()
    for x,y in zip((q,k,v),(qq,kk,vv)):
        torch.testing.assert_close(x.grad,y.grad,atol=3e-6,rtol=3e-5)


def tiny_target():
    return Qwen3ForCausalLM(Qwen3Config(vocab_size=48,hidden_size=32,intermediate_size=48,
        num_hidden_layers=3,num_attention_heads=4,num_key_value_heads=2,head_dim=8,
        max_position_embeddings=512,layer_types=['full_attention']*3,
        attention_dropout=0.,pad_token_id=0,eos_token_id=None,_attn_implementation='sdpa')).eval()


def test_bonus_addon_greedy_parity():
    from four_anchor.model import FourAnchorDFlash,DraftConfig
    torch.manual_seed(42)
    target=tiny_target()
    cfg=DraftConfig('tiny',[0,1],47,2,4,15,'sdpa',format_version=4,anchor_layout='bonus_addon')
    draft=FourAnchorDFlash(cfg,target).eval()
    ids=torch.tensor([[1,2,3,4,5]])
    actual,lengths,extra=generate_mine(target,draft,ids,40,[])
    expected=greedy_reference(target,ids,40,[])
    assert actual==expected and len(actual)==40
    assert all(1<=x<=16 for x in lengths)
    assert len(extra['redraft_boundaries'])==len(lengths)


def test_row_quarters_are_not_step_quarters():
    plan=quarter_plan([1,1,1,9])
    assert [p['step'] for p in plan]==[3,4]
    assert plan[-1]['scheduled_fractions']==[.5,.75,1.]
    assert plan[-1]['actual_fraction']==1


def test_literal_quarter_percent():
    assert len(quarter_plan([1]*400,.0025))==400


def test_answer_scoring():
    assert answer_number('work 999 final \\boxed{1,200.00}')==answer_number('calculation\n#### 1200',True)
    assert answer_number('work 999 with no final box') is None


@pytest.mark.parametrize('gate',[-0.5,-1000.0])
def test_dloop_evaluator_greedy_parity_and_looping(gate,monkeypatch):
    from eval import linear_decode
    import time
    monkeypatch.setattr(linear_decode,'_cuda_time',time.perf_counter)
    torch.manual_seed(88)
    target=tiny_target()
    cfg=copy.deepcopy(target.config)
    cfg.num_hidden_layers=2;cfg.layer_types=['full_attention']*2
    cfg.num_target_layers=3;cfg.block_size=16
    cfg.dflash_config={'target_layer_ids':[0,1],'mask_token_id':47}
    draft=DFlashDraftModel(cfg).eval()
    ids=torch.tensor([[1,2,3,4,5]])
    result=linear_decode.dflash_generate(draft,target,ids,40,[],loop_gate=gate,stage_cap=3,return_stats=True)
    assert result.output_ids[0,ids.shape[1]:].tolist()==greedy_reference(target,ids,40,[])
    if gate==-1000.0:
        assert all(x>0 for x in result.stages_fired)


def test_native_dloop_full_objective_and_gradients(monkeypatch):
    """Compare all 1..3 carrier stages and both supervised losses through full model."""
    import specforge.core.dflash as core
    import specforge.modeling.draft.dflash as model
    torch.manual_seed(7)
    target=tiny_target()
    config=copy.deepcopy(target.config)
    config.num_hidden_layers=2
    config.layer_types=['full_attention']*2
    config.num_target_layers=3
    config.block_size=4
    config.dflash_config={'target_layer_ids':[0,1],'mask_token_id':47}
    config._attn_implementation='sdpa'
    first=OnlineDFlashModel(DFlashDraftModel(config),target.lm_head,target.model.embed_tokens,47,
             block_size=4,attention_backend='sdpa',num_anchors=3,loss_decay_gamma=7,loop_max_stage=3)
    second=copy.deepcopy(first)
    ids=torch.randint(0,47,(2,28))
    features=torch.randn(2,28,64)
    loss_mask=torch.ones_like(ids,dtype=torch.float32)
    for depth in (1,2,3):
        first._sample_loop_depth=lambda:depth
        second._sample_loop_depth=lambda:depth
        torch.manual_seed(55)
        native=first(ids,features,loss_mask)[0]
        native.backward()
        with monkeypatch.context() as m:
            from pair.fast_attention import install
            # Record original attributes BEFORE install, so each depth compares
            # against the real native dense implementation again.
            m.setattr(model.Qwen3DFlashAttention,'forward',model.Qwen3DFlashAttention.forward)
            m.setattr(core,'create_dflash_sdpa_mask',core.create_dflash_sdpa_mask)
            install()
            torch.manual_seed(55)
            actual=second(ids,features,loss_mask)[0]
            actual.backward()
        torch.testing.assert_close(native,actual,atol=1e-6,rtol=1e-5)
        for (name,p),(_,q) in zip(first.named_parameters(),second.named_parameters()):
            if p.grad is not None:
                torch.testing.assert_close(p.grad,q.grad,atol=4e-6,rtol=2e-4,msg=lambda msg:name+': '+msg)
        first.zero_grad();second.zero_grad()
