import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import random

import pytest
import torch
from torch import nn
from torch.nn import functional as F
from transformers import Qwen3Config
from datasets import Dataset

from ascend_support.verify_sources import ROOT, verify
from ascend_support.tokenized_data import SavedTokenDataset
from ascend_support.preflight import require_free, validate_configs, resolve_draft
from ascend_support.checkpoints import capture_rng, restore_rng, get_last_complete_checkpoint
from specforge.core.dflash import OnlineDFlashModel
from specforge.modeling.draft.dflash import DFlashDraftModel
from specforge.data.on_the_fly import OnTheFlyDataset


def tiny_config():
    cfg = Qwen3Config(vocab_size=67, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8, max_position_embeddings=32768, attention_dropout=0.,
        pad_token_id=0, bos_token_id=1, eos_token_id=2, tie_word_embeddings=True)
    cfg.num_target_layers=4
    cfg.block_size=16
    cfg.dflash_config={"target_layer_ids":[0,2],"mask_token_id":66}
    cfg._attn_implementation="sdpa"
    return cfg


def wrapper():
    cfg=tiny_config()
    draft=DFlashDraftModel(cfg)
    head=nn.Linear(32,67,bias=False).requires_grad_(False)
    embedding=nn.Embedding(67,32).requires_grad_(False)
    return OnlineDFlashModel(draft,head,embedding,66,16,"sdpa",3,7.0,loop_max_stage=3)


def test_method_files_identical_to_pinned_naver():
    changed=verify()
    assert "scripts/train_dflash.py" in changed
    originals=json.loads((ROOT/"UPSTREAM_SHA256.json").read_text())
    for name in ["specforge/core/dflash.py", "specforge/modeling/draft/dflash.py",
                 "specforge/optimizer.py", "specforge/lr_scheduler.py",
                 "specforge/inference/target_engine/dflash_target_model.py",
                 "specforge/data/on_the_fly.py"]:
        assert hashlib.sha256((ROOT/"upstream"/name).read_bytes()).hexdigest()==originals[name]


@pytest.mark.parametrize("depth", [1,2,3])
def test_native_loop_ce_and_gradients(depth):
    torch.manual_seed(7)
    native=wrapper()
    manual=copy.deepcopy(native)
    ids=torch.randint(0,66,(1,100))
    mask=torch.ones_like(ids)
    hidden=torch.randn(1,100,64)
    anchors=torch.tensor([[8,17,31]])
    keep=torch.ones_like(anchors,dtype=torch.bool)
    native._sample_anchor_positions=lambda *args:(anchors,keep)
    native._sample_loop_depth=lambda:depth
    actual=native(ids,hidden,mask,position_offsets=torch.tensor([37]))[0]
    actual.backward()

    previous=[]
    sums=[]
    weights=[]
    for stage in range(depth+1):
        offset=15*stage
        carrier=None if not previous else torch.cat(previous,dim=2)[:,:,:offset-1]
        _,_,out=manual._forward_draft_blocks(ids,hidden,mask,sampled=(anchors,keep),
            offset=offset,rows=16,carrier=carrier,position_offsets=torch.tensor([37]))
        produced=out.reshape(1,3,16,32)[:,:,1:]
        previous.append(produced)
        if stage in (0,depth):
            logits=manual.lm_head(produced)
            labels=ids[:,anchors[0,:,None]+torch.arange(offset+1,offset+16)]
            ce=F.cross_entropy(logits.reshape(-1,67),labels.reshape(-1),reduction="none").reshape(1,3,15)
            weight=torch.exp(-torch.arange(15).float()/7).expand_as(ce)
            sums.append((ce*weight).sum());weights.append(weight.sum())
    expected=sum(sums)/(sum(weights)+1e-6)
    expected.backward()
    torch.testing.assert_close(actual,expected)
    for a,b in zip(native.draft_model.parameters(),manual.draft_model.parameters()):
        torch.testing.assert_close(a.grad,b.grad)


def test_sampling_requires_room_for_main_plus_loop():
    model=wrapper()
    mask=torch.ones(1,100)
    for _ in range(10):
        anchors,keep=model._sample_anchor_positions(100,mask,torch.device("cpu"))
        assert torch.all(anchors[keep] <= 68)


def test_saved_tokens_preserve_ids_and_upstream_replacement_policy(tmp_path):
    records={"input_ids":[[3]*10,[4]*80,[5]*96],
             "loss_mask":[[1]*10,[0]*10+[1]*70,[1]*96]}
    path=tmp_path/"data"
    Dataset.from_dict(records).save_to_disk(path)
    ds=SavedTokenDataset(path,max_length=90,min_loss_tokens=64)
    assert type(ds).__getitem__ is OnTheFlyDataset.__getitem__
    first=ds[0]
    assert first["input_ids"].shape[1] >= 64
    assert len(ds)==3
    assert ds[1]["input_ids"].tolist()==[[4]*80]
    assert ds[1]["loss_mask"].tolist()==[[0]*10+[1]*70]
    assert ds[2]["input_ids"].shape==(1,90)
    torch.testing.assert_close(first["input_ids"],ds[0]["input_ids"])


def test_occupied_devices_refused():
    require_free(["7"],"| No running processes found in NPU 7 |")
    with pytest.raises(ValueError,match="occupied"):
        require_free(["7"],"| 7 0 12345 python 4000 |")
    with pytest.raises(ValueError,match="distinct"):
        require_free(["7","7"],"")


def test_no_custom_checkpoint_or_scratch_fallback(tmp_path,monkeypatch):
    monkeypatch.setenv("INIT_MODE", "pretrained")
    monkeypatch.setenv("DRAFT_INIT",str(tmp_path/"missing"))
    with pytest.raises(ValueError,match="config/weights"):
        resolve_draft()
    target=tmp_path/"target";target.mkdir()
    draft=tmp_path/"draft";draft.mkdir()
    cfg=tiny_config().to_dict();cfg["architectures"]=["DFlashDraftModel"]
    (target/"config.json").write_text(json.dumps({**cfg,"num_hidden_layers":4}))
    (draft/"config.json").write_text(json.dumps(cfg))
    validate_configs(target,draft)
    (draft/"four_anchor_config.json").write_text("{}")
    with pytest.raises(ValueError,match="four-anchor"):
        validate_configs(target,draft)


def test_explicit_scratch_uses_config_only_and_rejects_stale_weight_path(monkeypatch):
    monkeypatch.setenv("INIT_MODE", "scratch")
    monkeypatch.delenv("DRAFT_INIT", raising=False)
    path = resolve_draft()
    config = json.loads((path / "config.json").read_text())
    assert not list(path.glob("*.safetensors"))
    assert config["num_hidden_layers"] == 5
    assert config["block_size"] == 16
    assert config["dflash_config"]["target_layer_ids"] == [1,9,17,25,33]
    monkeypatch.setenv("DRAFT_INIT", "/old/pretrained/checkpoint")
    with pytest.raises(ValueError, match="unset DRAFT_INIT"):
        resolve_draft()


def test_invalid_initialization_mode_refused(monkeypatch):
    monkeypatch.setenv("INIT_MODE", "scrath")
    with pytest.raises(ValueError, match="INIT_MODE"):
        resolve_draft()


def test_rng_and_incomplete_checkpoint_detection(tmp_path):
    random.seed(17);torch.manual_seed(17)
    saved=capture_rng()
    first=(random.random(),torch.rand(3))
    restore_rng(saved)
    assert first[0]==random.random()
    torch.testing.assert_close(first[1],torch.rand(3))
    path=tmp_path/"epoch_0_step_5";path.mkdir()
    (path/"COMPLETE.json").write_text('{"world_size":2}')
    (path/"training_state_rank0.pt").write_bytes(b"saved")
    with pytest.raises(ValueError,match="No complete"):
        get_last_complete_checkpoint(tmp_path)
    (path/"training_state_rank1.pt").write_bytes(b"saved")
    selected,position=get_last_complete_checkpoint(tmp_path)
    assert Path(selected)==path and position==(0,5)
