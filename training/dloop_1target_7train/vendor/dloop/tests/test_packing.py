"""Document isolation, coverage, native objective/gradient parity and resume."""
import copy
import json

import numpy as np
import pytest
import torch
from datasets import Dataset
from transformers import Qwen3ForCausalLM

from ascend_support.packed_data import (build_index, load_index, PackCollator,
                                       TokenBudgetBatchSampler)
from ascend_support.packed_objective import (allocate_anchors, enable_packed_sampling,
                                            set_packed_step)
from ascend_support.packed_target import enable_hidden_only_target
from ascend_support.tokenized_data import SavedTokenDataset
from specforge.inference.target_engine.dflash_target_model import HFDFlashTargetEngine
from test_native import tiny_config, wrapper


def test_index_keeps_upstream_replacements_and_checks_fingerprint(tmp_path):
    lengths = [8, 63, 64, 80, 95, 128, 256, 310]
    path = tmp_path / "data"
    Dataset.from_dict({"input_ids":[[i+1]*n for i,n in enumerate(lengths)],
                       "loss_mask":[[1]*n for n in lengths]}).save_to_disk(path)
    ds = SavedTokenDataset(path, 300, 64)
    index = build_index(ds, tmp_path/"cache")
    packed_lengths, replaced = load_index(ds, index)
    assert replaced == 2
    assert packed_lengths.tolist() == [ds[i]["input_ids"].shape[-1] for i in range(len(ds))]
    before = index.stat().st_mtime_ns
    assert build_index(ds, tmp_path/"cache") == index
    assert index.stat().st_mtime_ns == before
    with pytest.raises(ValueError, match="does not match"):
        load_index(SavedTokenDataset(path, 299, 64), index)


def test_packing_distributed_coverage_budget_and_epoch_resume():
    lengths = np.array([65, 80, 127, 128, 129, 200, 255, 300, 301, 512]*17)
    samplers = [TokenBudgetBatchSampler(lengths, 1536, 8, r, bucket_width=32) for r in range(8)]
    original_rng = torch.get_rng_state().clone()
    for epoch in (0, 1, 19):
        for sampler in samplers:
            sampler.set_epoch(epoch)
        global_packs = samplers[0].global_packs()
        originals = global_packs[:samplers[0].pack_count]
        assert sorted(i for pack in originals for i in pack) == list(range(len(lengths)))
        assert len(global_packs) % 8 == 0
        for r, sampler in enumerate(samplers):
            local = list(sampler)
            assert local == global_packs[r::8]
            assert len(local) == len(sampler) == len(samplers[0])
            assert local[2:] == list(sampler)[2:]  # deterministic mid-epoch skip
            for pack in local:
                assert len(pack)*max(lengths[pack]) <= 1536
        assert len(global_packs)-len(originals) < 8
    torch.testing.assert_close(torch.get_rng_state(), original_rng, rtol=0, atol=0)
    samplers[0].set_epoch(0); a=list(samplers[0])
    samplers[0].set_epoch(1); b=list(samplers[0])
    assert a != b


def test_collator_preserves_documents_and_padding_is_not_supervised():
    records = [{"input_ids":torch.full((1,n), token), "attention_mask":torch.ones(1,n,dtype=torch.long),
                "loss_mask":torch.ones(1,n)} for n,token in [(70,3),(100,4)]]
    batch = PackCollator(200)(records)
    assert batch["input_ids"].shape == (2,100)
    assert batch["input_ids"][0,:70].eq(3).all()
    assert batch["input_ids"][1].eq(4).all()
    assert batch["attention_mask"][0,70:].sum() == 0
    assert batch["loss_mask"][0,70:].sum() == 0
    with pytest.raises(ValueError,match="exceed"):
        PackCollator(199)(records)


def test_batch_anchor_cap_true_document_ends_and_bounded_padded_work():
    model = enable_packed_sampling(wrapper())
    model.num_anchors = 17
    lengths = torch.tensor([70,96,100])
    set_packed_step(model, lengths, 123)
    mask = torch.zeros(3,100)
    for i,n in enumerate(lengths):
        mask[i,:n] = 1
    anchors, keep = model._sample_anchor_positions(100,mask,torch.device("cpu"))
    assert int(keep.sum()) == 17
    assert torch.all(anchors[keep] <= (lengths-32).unsqueeze(1).expand_as(anchors)[keep])
    assert model._packed_allocated_anchor_slots <= 17+len(lengths)-1
    for row, valid in zip(anchors,keep):
        assert row[valid].unique().numel() == int(valid.sum())
    # A response-rich document cannot inflate every other row's block count.
    quotas = allocate_anchors([1,1,4000],768,42)
    assert quotas.tolist() == [1,1,256]
    assert quotas.sum() <= 768


def test_loop_depth_identical_across_ranks_despite_different_rng_consumption():
    a, b = enable_packed_sampling(wrapper()), enable_packed_sampling(wrapper())
    depths=[]
    for step in range(1,301):
        set_packed_step(a,torch.tensor([70,80]),step)
        set_packed_step(b,torch.tensor([100]),step)
        first=a._sample_loop_depth()
        torch.rand(step % 31 + 1)
        assert b._sample_loop_depth() == first
        depths.append(first)
    assert set(depths) == {1,2,3}


@pytest.mark.parametrize("depth",[1,2,3])
def test_padded_native_dloop_equals_separate_documents_loss_and_gradients(depth):
    torch.manual_seed(13)
    packed=wrapper()
    separate=copy.deepcopy(packed)
    lengths=[76,103]
    ids=torch.randint(1,66,(2,103))
    masks=torch.ones_like(ids)
    masks[0,76:]=0
    masks[:, :8]=0
    hidden=torch.randn(2,103,64)
    # Include a short document's boundary block whose later stages fall off
    # the document, and a padded dummy block in the second row.
    anchors=torch.tensor([[10,33,43],[11,51,0]])
    keep=torch.tensor([[True,True,True],[True,True,False]])
    packed._sample_anchor_positions=lambda *a:(anchors,keep)
    packed._sample_loop_depth=lambda:depth
    offsets=torch.tensor([31,79])
    actual=packed(ids,hidden,masks,position_offsets=offsets)[0]
    actual.backward()
    sums=[];weights=[]
    for i,length in enumerate(lengths):
        sampled=(anchors[i:i+1,keep[i]],torch.ones(1,int(keep[i].sum()),dtype=torch.bool))
        x,h,m=ids[i:i+1,:length],hidden[i:i+1,:length],masks[i:i+1,:length]
        _,_,out=separate._forward_draft_blocks(x,h,m,sampled=sampled,position_offsets=offsets[i:i+1])
        main=out.view(1,sampled[0].shape[1],16,-1)[:,:,1:]
        final=separate._forward_loop_stages(x,h,m,sampled,main,depth,1,offsets[i:i+1])
        for states,offset in ((main,1),(final,depth*15+1)):
            ce,w,_,_=separate._block_loss(states,x,m,*sampled,offset)
            sums.append(ce);weights.append(w)
    expected=sum(sums)/(sum(weights)+1e-6)
    expected.backward()
    torch.testing.assert_close(actual,expected,rtol=2e-5,atol=2e-6)
    for a,b in zip(packed.draft_model.parameters(),separate.draft_model.parameters()):
        torch.testing.assert_close(a.grad,b.grad,rtol=2e-4,atol=2e-5)


def test_teacher_hidden_only_parity_and_cross_document_isolation():
    torch.manual_seed(17)
    cfg=tiny_config();cfg.num_hidden_layers=4;cfg.layer_types=["full_attention"]*4
    engine=HFDFlashTargetEngine(Qwen3ForCausalLM(cfg).eval())
    engine.set_capture_layers([0,2])
    ids=torch.randint(1,66,(2,90))
    attention=torch.ones_like(ids);attention[0,70:]=0
    loss_mask=attention.clone();offsets=torch.tensor([35,71])
    before=engine.generate_dflash_data(ids,attention,loss_mask,offsets).hidden_states
    enable_hidden_only_target(engine)
    # Fail if the otherwise-unused vocabulary head is invoked.
    def forbidden(*a,**k):raise AssertionError("Teacher LM head should not run")
    engine.model.lm_head.forward=forbidden
    after=engine.generate_dflash_data(ids,attention,loss_mask,offsets).hidden_states
    torch.testing.assert_close(before,after,rtol=0,atol=0)
    single=engine.generate_dflash_data(ids[:1,:70],attention[:1,:70],loss_mask[:1,:70],offsets[:1]).hidden_states
    torch.testing.assert_close(after[:1,:70],single,rtol=1e-5,atol=1e-6)
    changed=ids.clone();changed[1]=torch.randint(1,66,(90,));changed[0,70:]=17
    perturb=engine.generate_dflash_data(changed,attention,loss_mask,offsets).hidden_states
    torch.testing.assert_close(after[0,:70],perturb[0,:70],rtol=0,atol=0)
