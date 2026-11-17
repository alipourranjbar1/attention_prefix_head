"""Lossless regular-DFlash weights, explicitly fine-tuned with full attention."""
from pathlib import Path
import hashlib
import json
import os
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import Qwen3Config


def convert(source, target, destination):
    from specforge.modeling.draft.dflash import DFlashDraftModel
    src, dest = Path(source).resolve(), Path(destination)
    if dest.exists():
        raise FileExistsError(dest)
    raw = json.loads((src/'config.json').read_text())
    tc = Qwen3Config.from_pretrained(target)
    if (src/'four_anchor_config.json').exists():
        raise ValueError('Use the shared regular DFlash base checkpoint, not a four-anchor checkpoint')
    if raw.get('speculators_model_type') == 'dflash':
        if raw.get('sample_from_anchor') or raw.get('sliding_window_non_causal'):
            raise ValueError('Only the regular 15-proposal / block-16 DFlash host is supported')
        dc = Qwen3Config(**raw['transformer_layer_config'])
        taps, mask = raw['aux_hidden_state_layer_ids'], raw['mask_token_id']
        vocab, block = raw['draft_vocab_size'], raw['block_size']
    elif raw.get('architectures') == ['DFlashDraftModel']:
        dc = Qwen3Config(**raw)
        taps, mask = raw['dflash_config']['target_layer_ids'], raw['dflash_config']['mask_token_id']
        vocab, block = dc.vocab_size, raw['block_size']
        if any(raw['dflash_config'].get(k) for k in ('projector_type','shift_label','pure_draft_prefix_len')):
            raise ValueError('Checkpoint is not a regular DFlash host')
    else:
        raise ValueError('Unrecognized base checkpoint config; no random-initialization fallback')
    if block != 16 or dc.num_hidden_layers != 5 or vocab != tc.vocab_size:
        raise ValueError('Expected the five-layer, full-vocabulary, block-16 DFlash base')
    if taps != [1,9,17,25,33] or not 0 <= mask < vocab:
        raise ValueError('Expected Qwen3-4B taps [1,9,17,25,33] and a valid mask token')
    for key in ('hidden_size','intermediate_size','num_attention_heads','num_key_value_heads',
                'head_dim','rms_norm_eps','attention_bias','hidden_act'):
        if getattr(dc,key,None) != getattr(tc,key,None):
            raise ValueError(f'Base/target architecture mismatch: {key}')
    # Own method derives RoPE from target, so require identical RoPE too.
    for key in ('rope_theta','rope_scaling','rope_parameters'):
        if getattr(dc,key,None) != getattr(tc,key,None):
            raise ValueError(f'Base/target positional embedding mismatch: {key}')
    if dc.attention_dropout != 0.0:
        raise ValueError(f'Expected zero dropout, found attention_dropout={dc.attention_dropout!r}')
    if (not dc.layer_types or len(dc.layer_types) < dc.num_hidden_layers
        or any(t not in {'full_attention','sliding_attention'} for t in dc.layer_types)):
        raise ValueError(f'Unsupported source attention layout: {dc.layer_types!r}')
    source_attention = {
        'layer_types': list(dc.layer_types),
        'use_sliding_window': getattr(dc,'use_sliding_window',False),
        'sliding_window': getattr(dc,'sliding_window',None),
        'attention_dropout': dc.attention_dropout,
    }
    # User-selected fine-tuning recipe: keep every pretrained tensor, but remove
    # the source checkpoint's sliding-window restriction in BOTH draft models.
    # The original checkpoint/config and the frozen target are never rewritten.
    dc.layer_types = ['full_attention'] * dc.num_hidden_layers
    dc.use_sliding_window = False
    dc.sliding_window = None
    training_attention = {
        'layer_types': list(dc.layer_types), 'use_sliding_window': False,
        'sliding_window': None, 'attention_dropout': dc.attention_dropout,
    }
    dc.num_target_layers = tc.num_hidden_layers
    dc.block_size = block
    dc.dflash_config = {'target_layer_ids':taps,'mask_token_id':mask}
    dc.architectures = ['DFlashDraftModel']
    dc._attn_implementation = 'sdpa'
    with torch.device('meta'):
        expected = {k:tuple(v.shape) for k,v in DFlashDraftModel(dc).state_dict().items()}
    files = sorted(src.glob('*.safetensors'))
    index = src/'model.safetensors.index.json'
    if index.exists():
        files = [src/x for x in sorted(set(json.loads(index.read_text())['weight_map'].values()))]
    if not files:
        raise ValueError(f'No safetensors weights in {src}')
    state, ignored, tensor_sha = {}, [], {}
    allowed_extra = {'embed_tokens.weight','lm_head.weight','verifier_norm.weight','verifier_lm_head.weight'}
    for file in files:
        with safe_open(str(file),framework='pt',device='cpu') as saved:
            for key in saved.keys():
                if key not in expected:
                    if key not in allowed_extra:
                        raise ValueError(f'Unexpected checkpoint tensor {key}')
                    ignored.append(key)
                    continue
                if key in state or tuple(saved.get_slice(key).get_shape()) != expected[key]:
                    raise ValueError(f'Duplicate or incompatible checkpoint tensor {key}')
                value = saved.get_tensor(key).contiguous()
                if not torch.isfinite(value).all():
                    raise ValueError(f'Nonfinite checkpoint tensor {key}')
                state[key] = value
                tensor_sha[key] = hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()
    if state.keys() != expected.keys():
        raise ValueError(f'Missing pretrained tensors: {set(expected)-set(state)}')
    for name in ('mine','dloop'):
        (dest/name).mkdir(parents=True)
    save_file(state,str(dest/'dloop/model.safetensors'))
    # Two config views, physically one weights file; never modify the original.
    os.link(dest/'dloop/model.safetensors',dest/'mine/model.safetensors')
    official = dc.to_dict()
    official['architectures'] = ['DFlashDraftModel']
    official['rope_theta'] = getattr(tc,'rope_theta',None) or tc.rope_parameters['rope_theta']
    (dest/'dloop/config.json').write_text(json.dumps(official,indent=2)+'\n')
    own = {'speculators_model_type':'dflash','architectures':['DFlashSpeculator'],
           'transformer_layer_config':official,'draft_vocab_size':vocab,'block_size':16,
           'aux_hidden_state_layer_ids':taps,'mask_token_id':mask,'sample_from_anchor':False}
    (dest/'mine/config.json').write_text(json.dumps(own,indent=2)+'\n')
    report = {'source':str(src),'target':str(Path(target).resolve()),'tensors':tensor_sha,
              'ignored_frozen_target_tensors':ignored,'initialization':'identical pretrained regular DFlash weights',
              'source_attention':source_attention,'training_attention':training_attention,
              'attention_note':'User-requested full-attention fine-tuning; weights preserved exactly, connectivity may change',
              'new_mine_parameters':['draft_anchor_proj.weight (identity)','draft_anchor_norm.weight (ones)']}
    (dest/'conversion.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'event':'draft_attention_configuration',
                      'source':source_attention,'both_training_drafters':training_attention}),flush=True)
    return own

