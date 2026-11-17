import json
from pathlib import Path
import pytest
import torch
from datasets import Dataset,load_from_disk
from safetensors.torch import save_file,load_file
from transformers import Qwen3Config,Qwen3ForCausalLM
from specforge.modeling.draft.dflash import DFlashDraftModel
from four_anchor.model import DraftConfig,FourAnchorDFlash
from pair.checkpoints import convert
from pair.prepare import prepare_data,prepare_output_directory
from pair.run import dloop_arguments


@pytest.mark.parametrize('source_format',['speculators','hf'])
@pytest.mark.parametrize('source_attention',['full_attention','sliding_attention'])
def test_checkpoint_views_have_identical_weights_and_load_both_methods(tmp_path,source_format,source_attention):
    target_dir=tmp_path/'target'
    source=tmp_path/'source';source.mkdir()
    tc=Qwen3Config(vocab_size=48,hidden_size=16,intermediate_size=24,num_attention_heads=2,
        num_key_value_heads=1,head_dim=8,num_hidden_layers=36,max_position_embeddings=512,
        layer_types=['full_attention']*36,attention_dropout=0.,_attn_implementation='sdpa')
    tc.save_pretrained(target_dir)
    dc=Qwen3Config(**tc.to_dict())
    dc.num_hidden_layers=5;dc.layer_types=[source_attention]*5
    dc.use_sliding_window=source_attention=='sliding_attention'
    dc.sliding_window=2048 if dc.use_sliding_window else None
    dc.num_target_layers=36;dc.block_size=16
    dc.dflash_config={'target_layer_ids':[1,9,17,25,33],'mask_token_id':47}
    dc.architectures=['DFlashDraftModel']
    dc._attn_implementation='sdpa'
    original=DFlashDraftModel(dc)
    state=original.state_dict()
    raw=dc.to_dict()
    if source_format=='speculators':
        raw={'speculators_model_type':'dflash','transformer_layer_config':raw,
             'aux_hidden_state_layer_ids':[1,9,17,25,33],'mask_token_id':47,
             'draft_vocab_size':48,'block_size':16}
    (source/'config.json').write_text(json.dumps(raw))
    save_file(state,str(source/'model.safetensors'))
    source_config_bytes=(source/'config.json').read_bytes()
    source_weight_bytes=(source/'model.safetensors').read_bytes()
    output=tmp_path/'converted'
    convert(source,str(target_dir),output)
    converted=load_file(str(output/'dloop/model.safetensors'))
    for key in state:
        torch.testing.assert_close(converted[key],state[key],rtol=0,atol=0)
    official=DFlashDraftModel.from_pretrained(output/'dloop',attn_implementation='sdpa')
    target=Qwen3ForCausalLM(tc)
    own=FourAnchorDFlash(DraftConfig('tiny',[1,9,17,25,33],47,5,4,15,'sdpa',
                       format_version=4,anchor_layout='bonus_addon'),target)
    own.initialize_regular_dflash(output/'mine')
    assert (source/'config.json').read_bytes()==source_config_bytes
    assert (source/'model.safetensors').read_bytes()==source_weight_bytes
    for name in ('mine','dloop'):
        config=json.loads((output/name/'config.json').read_text())
        config=config.get('transformer_layer_config',config)
        assert config['layer_types']==['full_attention']*5
        assert config['use_sliding_window'] is False and config['sliding_window'] is None
    assert all(layer.self_attn.sliding_window is None for model in (own,official) for layer in model.layers)
    audit=json.loads((output/'conversion.json').read_text())
    assert audit['source_attention']['layer_types']==[source_attention]*5
    assert audit['training_attention']['layer_types']==['full_attention']*5
    for key in state:
        torch.testing.assert_close(own.state_dict()[key],official.state_dict()[key],rtol=0,atol=0)
    torch.testing.assert_close(own.draft_anchor_proj.weight,torch.eye(16))


def test_retry_directory_preserves_completed_gsm8k(tmp_path):
    out=tmp_path/'run'
    prepare_output_directory(out)
    prompts=out/'gsm8k.json'
    prompts.write_text('{"records": []}')
    before=prompts.read_bytes()
    prepare_output_directory(out)
    assert prompts.read_bytes()==before


@pytest.mark.parametrize('entry',['settings.json','initialization','mine','dloop'])
def test_retry_refuses_prepared_or_training_outputs(tmp_path,entry):
    out=tmp_path/'run';out.mkdir()
    (out/entry).mkdir()
    with pytest.raises(FileExistsError): prepare_output_directory(out)
    assert (out/entry).exists()


def test_common_filter_no_replacement_source_unchanged(tmp_path):
    source=tmp_path/'source'
    Dataset.from_dict({'input_ids':[list(range(100))]*3,
                       'loss_mask':[[1.]*100,[0.]*50+[1.]*50,[0.]*2+[1.]*98]}).save_to_disk(source)
    report=prepare_data(source,tmp_path/'filtered')
    assert report['original_rows']==3 and report['training_rows']==2
    assert len(load_from_disk(source))==3
    assert len(load_from_disk(tmp_path/'filtered'))==2


def test_quarter_stop_does_not_change_full_lr_schedule():
    settings={'model':'target','dataset':'train','epochs':2,'dloop_head_chunk':512,
              'anchors_per_rank_batch':1536,
              'dloop_packing_index':'index.npz'}
    a=dloop_arguments(settings,Path('/run'),50,False)
    b=dloop_arguments(settings,Path('/run'),100,True)
    assert a[a.index('--num-epochs')+1]==b[b.index('--num-epochs')+1]=='2'
    assert '--resume' not in a and '--resume' in b
    assert a[a.index('--draft-init-path')+1]=='/run/initialization/dloop'
    assert a[a.index('--num-anchors')+1]==b[b.index('--num-anchors')+1]=='1536'
