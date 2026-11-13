"""Actual trainer lifecycle on tiny CPU models; distributed/FSDP are stubbed.

This is deliberately not evidence of NPU or distributed collective support.
"""
from contextlib import nullcontext
import importlib.util
from pathlib import Path
import sys

import pytest
import torch
import torch.distributed as dist
from datasets import Dataset
from safetensors.torch import load_file
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast, Qwen3ForCausalLM

from ascend_support.verify_sources import ROOT
from specforge.modeling.draft.dflash import DFlashDraftModel
from test_native import tiny_config


@pytest.mark.parametrize("mode", ["pretrained", "scratch"])
@pytest.mark.parametrize("memory_saving", [False, True])
@pytest.mark.parametrize("packed", [False, True])
def test_actual_upstream_trainer_save_and_resume_cpu(tmp_path,monkeypatch,mode,memory_saving,packed):
    spec=importlib.util.spec_from_file_location("native_train_test",ROOT/"upstream/scripts/train_dflash.py")
    training=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(training)
    monkeypatch.setenv("SPECFORGE_DEVICE","cpu")
    monkeypatch.setattr(dist,"get_rank",lambda *a,**k:0)
    monkeypatch.setattr(dist,"get_world_size",lambda *a,**k:1)
    monkeypatch.setattr(dist,"is_initialized",lambda:True)
    monkeypatch.setattr(dist,"barrier",lambda *a,**k:None)
    monkeypatch.setattr(dist,"all_reduce",lambda *a,**k:None)
    monkeypatch.setattr(training,"init_distributed",lambda **k:None)
    monkeypatch.setattr(training,"destroy_distributed",lambda:None)

    class CPUIdentityFSDP:
        def __new__(cls,module,**kwargs):return module
        @staticmethod
        def state_dict_type(*args):return nullcontext()
    monkeypatch.setattr(training,"FSDP",CPUIdentityFSDP)

    torch.manual_seed(13)
    target=tmp_path/"target"
    cfg=tiny_config();cfg.num_hidden_layers=4;cfg.layer_types=["full_attention"]*4
    Qwen3ForCausalLM(cfg).save_pretrained(target)
    tokens=Tokenizer(WordLevel({f"t{i}":i for i in range(67)},unk_token="t0"))
    PreTrainedTokenizerFast(tokenizer_object=tokens,unk_token="t0",pad_token="t0",
        bos_token="t1",eos_token="t2",mask_token="t66").save_pretrained(target)
    draft=tmp_path/"draft"
    if mode == "pretrained":
        DFlashDraftModel(tiny_config()).save_pretrained(draft)
        initial_args=["--draft-init-path",str(draft)]
    else:
        tiny_config().save_pretrained(draft)
        assert not list(draft.glob("*.safetensors"))
        initial_args=["--scratch","--draft-config-path",str(draft)]
        def forbidden_load(*a,**k):
            raise AssertionError("Scratch attempted to load pretrained drafter weights")
        monkeypatch.setattr(training,"load_draft_init",forbidden_load)
    data=tmp_path/"data"
    Dataset.from_dict({"input_ids":[[3+i%60 for i in range(96)]]*4,
                       "loss_mask":[[0]*8+[1]*88]*4}).save_to_disk(data)
    common=["train_dflash.py","--target-model-path",str(target),
        "--target-model-backend","hf",*initial_args,
        "--train-data-path",str(data),"--saved-tokenized-data",
        "--num-epochs","1","--batch-size","1","--num-anchors","2",
        "--max-length","96","--attention-backend","sdpa",
        "--loss-type","dflash","--loss-decay-gamma","7",
        "--dloop-max-stage","3","--position-offset-max","300",
        "--dataloader-num-workers","0","--save-interval","100",
        "--log-interval","1","--report-to","none"]
    if memory_saving:
        common += ["--head-loss-chunk-size", "7", "--activation-checkpointing"]
    if packed:
        from ascend_support.packed_data import build_index
        from ascend_support.tokenized_data import SavedTokenDataset
        index=build_index(SavedTokenDataset(data,96,64),tmp_path/"cache")
        common += ["--packed-token-budget","192","--packing-index",str(index),
                   "--packing-bucket-width","16"]
    end_step=2 if packed else 4
    full=tmp_path/"full";part=tmp_path/"part"
    for out,extra in [(full,[]),(part,["--max-steps",str(end_step//2)]),(part,["--resume"])]:
        out.mkdir(exist_ok=True)
        monkeypatch.setattr(sys,"argv",common+["--output-dir",str(out)]+extra)
        training.main()
    a=load_file(full/f"epoch_1_step_{end_step}/model.safetensors")
    b=load_file(part/f"epoch_1_step_{end_step}/model.safetensors")
    assert a.keys()==b.keys()
    for name in a:
        torch.testing.assert_close(a[name],b[name],rtol=0,atol=0)
    assert (part/f"epoch_0_step_{end_step//2}/training_state_rank0.pt").is_file()
    assert (part/f"epoch_1_step_{end_step}/COMPLETE.json").is_file()


def test_scratch_model_initialization_is_random_seeded_and_does_not_load_weights(tmp_path,monkeypatch):
    spec=importlib.util.spec_from_file_location("scratch_build_test",ROOT/"upstream/scripts/train_dflash.py")
    training=importlib.util.module_from_spec(spec);spec.loader.exec_module(training)
    cfg_dir=tmp_path/"config";tiny_config().save_pretrained(cfg_dir)
    from types import SimpleNamespace
    target=SimpleNamespace(set_capture_layers=lambda layers:None)
    monkeypatch.setattr(training,"get_dflash_target_model",lambda **k:target)
    monkeypatch.setattr(training,"get_local_device",lambda:torch.device("cpu"))
    monkeypatch.setattr(training,"print_on_rank0",lambda *a,**k:None)
    def forbidden(*a,**k):raise AssertionError("Unexpected warm-start weight load")
    monkeypatch.setattr(training,"load_draft_init",forbidden)
    args=SimpleNamespace(target_model_path="unused",target_model_backend="hf",
        draft_init_path=None,draft_config_path=str(cfg_dir),trust_remote_code=False,
        block_size=16,attention_backend="sdpa",activation_checkpointing=False)
    def fresh(seed):
        torch.manual_seed(seed)
        return training.build_models(args)[1].state_dict()
    a,b,c=fresh(42),fresh(42),fresh(43)
    for name in a:torch.testing.assert_close(a[name],b[name],rtol=0,atol=0)
    assert not torch.equal(a["fc.weight"],c["fc.weight"])
    assert not torch.equal(a["layers.0.self_attn.q_proj.weight"],c["layers.0.self_attn.q_proj.weight"])
