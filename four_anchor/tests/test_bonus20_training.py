from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import random
from types import SimpleNamespace as NS

import pytest
import torch
from datasets import Dataset

from four_anchor.bonus_addon_objective import first_anchors, redraft_anchors
from four_anchor.batch_inference import BatchedBonusAddonDraft
from four_anchor.model import FourAnchorDFlash
from four_anchor.packed_objective import PackedTask
from four_anchor.tests.test_packed_training import batch_for, tiny_config
from four_anchor.vllm_support import ServingSettings


def addon(components):
    _, draft, _ = components
    draft.cfg = replace(draft.cfg, anchor_layout='bonus_addon', format_version=4, num_proposals=15)
    return draft


def packed(draft, batch, a, starts, limits, docs, cached, **kw):
    memory = draft.packed_memory(batch['features'][None], batch['positions']) if cached else None
    return draft.packed_hidden(batch['features'][None], batch['positions'], batch['documents'],
                               a, starts, limits, docs, memory=memory, **kw)


@pytest.mark.parametrize('cached', [False, True])
@pytest.mark.parametrize('boundary', range(15))
def test_both_training_passes_match_batched_inference_all_boundaries(components, cached, boundary):
    draft = addon(components)
    batch = batch_for(components)
    starts, docs = torch.tensor([2, 5, 9, 4]), torch.tensor([0, 0, 0, 1])
    limits = starts - 1
    a1, bonus = first_anchors(batch, starts, docs, draft.hidden_size)
    h1 = packed(draft, batch, a1, starts, limits, docs, cached, bonus_ids=bonus, return_all=True)
    assert h1.shape == (4, 20, draft.hidden_size)
    tokens = draft.lm_head(h1[:, 5:]).argmax(-1)
    keep = torch.full_like(starts, boundary)
    a2 = redraft_anchors(a1, bonus, starts, keep, h1, tokens)
    assert not a2.target_features.any() and not a2.draft_features.requires_grad
    pos = starts[:, None] + boundary + torch.arange(-4, 0)
    assert torch.equal(a2.source, torch.where(pos >= 0, 2, 0))
    h2 = packed(draft, batch, a2, starts + keep, limits, docs, cached)
    assert h2.shape == (4, 15, draft.hidden_size)
    with torch.inference_mode():
        fast = BatchedBonusAddonDraft(draft, 128, 4)
        updates=[]
        for i, (n, d) in enumerate(zip(limits.tolist(), docs.tolist())):
            o = int(batch['offsets'][d])
            updates.append((str(i), 0, batch['ids'][o:o+n], batch['features'][o:o+n]))
        fast.cache.update(updates, [str(i) for i in range(4)])
        memory=[(k[:, :, :int(limits.max())], v[:, :, :int(limits.max())]) for k,v in fast.cache.memory]
        rows=torch.arange(4)[:, None]
        feature=fast.cache.projected[rows, a1.positions] * (a1.source != 0)[..., None]
        masks=torch.full((4,15), draft.cfg.mask_token_id)
        out1=fast.query_hidden(memory, limits, a1.ids, a1.positions, feature, a1.source != 0,
                               torch.cat((bonus,masks),-1), limits)
        out2=fast.query_hidden(memory, limits, a2.ids, a2.positions,
                               draft.draft_anchor_norm(draft.draft_anchor_proj(a2.draft_features))*(a2.source != 0)[...,None],
                               a2.source != 0, masks, starts+keep)[:,4:]
    torch.testing.assert_close(h1,out1,atol=3e-6,rtol=3e-5)
    torch.testing.assert_close(h2,out2,atol=3e-6,rtol=3e-5)


@pytest.mark.parametrize('cached', [False, True])
def test_bonus_has_embedding_only_and_no_future_or_other_document_features_leak(components, cached):
    draft=addon(components); batch=batch_for(components)
    starts, docs=torch.tensor([5]),torch.tensor([0])
    a, bonus=first_anchors(batch, starts, docs, draft.hidden_size)
    assert a.ids.tolist()==[batch['ids'][:4].tolist()]
    assert bonus.item()==batch['ids'][4]
    torch.testing.assert_close(a.target_features[0],batch['features'][:4])
    h1=packed(draft,batch,a,starts,starts-1,docs,cached,bonus_ids=bonus,return_all=True)
    changed={**batch,'features':batch['features'].clone()}
    changed['features'][4:]=torch.randn_like(changed['features'][4:])*100
    other=packed(draft,changed,a,starts,starts-1,docs,cached,bonus_ids=bonus,return_all=True)
    torch.testing.assert_close(h1,other,atol=0,rtol=0)
    different=packed(draft,batch,a,starts,starts-1,docs,cached,bonus_ids=(bonus+1)%63,return_all=True)
    assert not torch.allclose(h1[:,5:],different[:,5:])
    a2=redraft_anchors(a,bonus,starts,torch.tensor([0]),h1,draft.lm_head(h1[:,5:]).argmax(-1))
    assert a2.ids.tolist()==[batch['ids'][1:5].tolist()]
    torch.testing.assert_close(a2.draft_features,h1[:,1:5].detach())
    torch.testing.assert_close(packed(draft,batch,a2,starts,starts-1,docs,cached),
                               packed(draft,changed,a2,starts,starts-1,docs,cached),atol=0,rtol=0)


@pytest.mark.parametrize('checkpointing',[False,True])
def test_losses_and_gradients_cached_reference_and_zero_boundary_redraft(components, monkeypatch, checkpointing):
    target,_,_=components; draft=addon(components); batch=batch_for(components)
    draft.gradient_checkpointing=checkpointing; clone=deepcopy(draft)
    cfg=tiny_config(num_proposals=15,anchor_layout='bonus_addon',boundary_exploration=0.)
    monkeypatch.setattr('four_anchor.bonus_addon_objective.rejection_boundary',lambda x:torch.zeros(len(x),dtype=torch.long))
    values=[]
    for model,mode in ((draft,'reference'),(clone,'cached')):
        task=PackedTask(model,target.model.norm,replace(cfg,packed_execution=mode),random.Random(7))
        num,den,stats=task(batch)
        assert task.last_redraft_blocks==cfg.max_anchors
        assert stats[10]==0 and stats[5]>0 and stats[9]>0
        (num/den).backward(); values.append((num,den,stats))
        assert model.draft_anchor_proj.weight.grad.abs().sum()>0
    for x,y in zip(*values): torch.testing.assert_close(x,y,atol=2e-5,rtol=3e-5)
    for (name,p),(_,q) in zip(draft.named_parameters(),clone.named_parameters()):
        if p.grad is not None:
            torch.testing.assert_close(p.grad,q.grad,atol=3e-5,rtol=5e-4,msg=name)


def test_loss_supervises_only_fifteen_candidates_and_the_replaced_suffix(components,monkeypatch):
    import four_anchor.bonus_addon_objective as objective
    target,_,_=components; draft=addon(components); batch=batch_for(components)
    seen=[]; real=objective.projected_loss
    def capture(h,head,targets,indices,weights,*args):
        seen.append((h.shape[1],indices.clone(),weights.clone()))
        return real(h,head,targets,indices,weights,*args)
    monkeypatch.setattr(objective,'projected_loss',capture)
    cfg=replace(tiny_config(num_proposals=15,anchor_layout='bonus_addon'),max_anchors=1)
    task=PackedTask(draft,target.model.norm,cfg,random.Random(2))
    task(batch)
    assert [n for n,_,_ in seen]==[15,15]
    assert (seen[0][1]>=1).all()  # teacher at s-1, never bonus label at s-2
    assert seen[1][2].sum()<=seen[0][2].sum()+1e-6


def test_format4_roundtrip_and_refuse_wrong_serving_policy(components,tmp_path):
    target,_,_=components; draft=addon(components)
    draft.save(tmp_path)
    loaded=FourAnchorDFlash.load(tmp_path,target)
    assert loaded.cfg==draft.cfg
    ServingSettings(str(tmp_path),anchor_policy='bonus_addon').metadata()
    with pytest.raises(ValueError,match='bonus_addon'):
        ServingSettings(str(tmp_path)).metadata()
    with pytest.raises(ValueError):
        replace(draft.cfg,format_version=3)


def test_new_epoch_worker_routes_to_concurrency32_bonus20_not_materialized_bonus():
    from four_anchor.epoch_benchmarks import suite_command
    args=NS(code_root='code',root='deps',run='run',model='model',npu=7,limit=128,max_new_tokens=2048,
            inference_layout='bonus_addon_batch',concurrency=32,cache_dir='/cache')
    cmd=suite_command(args,'out','checkpoint')
    assert cmd[1].endswith('/run_batched_benchmarks.py')
    assert cmd[cmd.index('--concurrency')+1]=='32'
    assert cmd[cmd.index('--inference-layout')+1]=='bonus_addon_batch'
    assert '--with-baseline' not in cmd


def test_tiny_packed_training_saves_format4_and_queues_complete_epoch(components,tmp_path,monkeypatch):
    from four_anchor import packed_train as pt
    target,_,_=components;data=tmp_path/'data'
    Dataset.from_dict({'input_ids':[list(range(1,24))]*2,'loss_mask':[[0,0]+[1]*21]*2}).save_to_disk(data)
    monkeypatch.setattr(pt,'load_target',lambda *a:(target,None))
    cfg=replace(tiny_config(num_proposals=15,anchor_layout='bonus_addon'),
        data=str(data),output=str(tmp_path/'checkpoints'),epochs=1,
        checkpoint_each_epoch=True,epoch_benchmark_dir=str(tmp_path/'evaluation'),
        max_anchors=2,gradient_accumulation=1,packed_execution='cached')
    saved=pt.train_packed(cfg)
    metadata=json.loads((Path(saved)/'four_anchor_config.json').read_text())['draft_config']
    assert metadata['anchor_layout']=='bonus_addon' and metadata['format_version']==4
    job=json.loads((tmp_path/'evaluation/queue/epoch-001.json').read_text())
    assert job['checkpoint']==str(Path(saved).resolve())
    assert (Path(saved)/'trainer.pt').is_file()
