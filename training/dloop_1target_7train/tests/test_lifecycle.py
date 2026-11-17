"""CPU lifecycle tests; single-rank identity FSDP is not accelerator validation."""
from dataclasses import replace
import importlib.util
from pathlib import Path
import sys
import torch
from datasets import Dataset
from safetensors.torch import load_file
from transformers import Qwen3Config,Qwen3ForCausalLM
from pair.common import ROOT


def test_dloop_optimized_save_resume_with_activation_checkpoints(tmp_path,monkeypatch):
    import specforge.core.dflash as core
    import specforge.modeling.draft.dflash as model
    monkeypatch.setattr(model.Qwen3DFlashAttention,'forward',model.Qwen3DFlashAttention.forward)
    monkeypatch.setattr(core,'create_dflash_sdpa_mask',core.create_dflash_sdpa_mask)
    from pair.fast_attention import install
    install()
    tests=ROOT/'vendor/dloop/tests'
    monkeypatch.syspath_prepend(str(tests))
    spec=importlib.util.spec_from_file_location('pair_native_smoke',tests/'test_train_smoke.py')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.test_actual_upstream_trainer_save_and_resume_cpu(tmp_path,monkeypatch,'pretrained',True,True)


def test_own_bonus20_pause_resume_exact(tmp_path,monkeypatch):
    monkeypatch.setenv('SPECULATORS_ROOT',str(ROOT/'vendor/speculators'))
    from four_anchor import packed_train as pt
    from four_anchor.train import TrainConfig
    cfg=Qwen3Config(vocab_size=64,hidden_size=32,intermediate_size=48,num_hidden_layers=2,
        num_attention_heads=4,num_key_value_heads=2,head_dim=8,max_position_embeddings=512,
        eos_token_id=None,pad_token_id=0,attention_dropout=0.,_attn_implementation='sdpa')
    target=Qwen3ForCausalLM(cfg).eval()
    monkeypatch.setattr(pt,'load_target',lambda *a:(target,None))
    data=tmp_path/'data'
    Dataset.from_dict({'input_ids':[list(range(1,32))]*4,'loss_mask':[[0,0]+[1]*29]*4}).save_to_disk(data)
    recipe=TrainConfig(target='tiny',data=str(data),output=str(tmp_path/'full'),mask_token_id=63,
        target_layer_ids=[0,1],device='cpu',dtype='float32',num_draft_layers=2,num_anchors=4,num_proposals=15,
        epochs=1,gradient_accumulation=1,max_sequence_length=32,max_anchors=2,blocks_per_chunk=2,
        head_chunk_size=7,warmup_steps=1,checkpoint_every=1000,log_every=1000,num_workers=0,
        train_data_ratio=1,training_mode='packed',packed_execution='cached',anchor_layout='bonus_addon',
        progress=False,data_cache_dir=str(tmp_path/'cache'),loss_fn={'ce':.1,'tv':.9})
    full=pt.train_packed(recipe)
    early=pt.train_packed(replace(recipe,output=str(tmp_path/'partial'),max_steps=2))
    continued=pt.train_packed(replace(recipe,output=str(tmp_path/'continued'),resume=early))
    a,b=load_file(str(Path(full)/'model.safetensors')),load_file(str(Path(continued)/'model.safetensors'))
    assert a.keys()==b.keys()
    assert all(torch.equal(a[k],b[k]) for k in a)
