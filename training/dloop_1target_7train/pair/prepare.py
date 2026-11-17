"""CPU preparation: common eligible data, pretrained weights, frozen GSM8K, schedules."""
from dataclasses import asdict
from pathlib import Path
import argparse
import json
import math
import os
import shutil
import numpy as np
from datasets import DatasetDict, load_dataset, load_from_disk
from tqdm import tqdm
from .common import ROOT, write_json, sha, quarter_plan
from .checkpoints import convert
from .gsm8k_source import prepare_gsm8k


def prepare_data(source, dest):
    data = load_from_disk(str(source))
    if isinstance(data, DatasetDict):
        data = data['train']
    if not {'input_ids','loss_mask'} <= set(data.column_names):
        raise ValueError('Expected tokenized input_ids and loss_mask columns')
    indices, wrapped, position = [], False, 0
    for batch in tqdm(data.with_format('numpy').iter(batch_size=1024),
                      total=math.ceil(len(data)/1024),desc='Checking common training records'):
        for i in range(len(batch['input_ids'])):
            ids, loss = np.asarray(batch['input_ids'][i]), np.asarray(batch['loss_mask'][i])
            if any(x.ndim not in (1,2) or (x.ndim==2 and x.shape[0]!=1) for x in (ids,loss)):
                raise ValueError(f'Row {position}: expected one sequence, not multiple documents')
            wrapped |= ids.ndim == 2
            ids, loss = ids.reshape(-1), loss.reshape(-1)
            if ids.shape != loss.shape or not len(ids) or len(ids)>3072:
                raise ValueError(f'Row {position}: expected matching, nonempty sequences of at most 3072 tokens')
            if not np.all(np.isfinite(loss)) or np.any(loss<0):
                raise ValueError(f'Invalid loss_mask in row {position}')
            if 'attention_mask' in batch and not np.all(np.asarray(batch['attention_mask'][i])==1):
                raise ValueError('Expected unpadded source records')
            # Avoid upstream DLoop's replacement of short records: both jobs see
            # the same filtered subset. Also guarantee >1 usable DLoop starts.
            eligible = loss.sum()>=64 and (loss[:max(len(ids)-31,0)]>0.5).sum()>1 and np.any(loss[2:]>0)
            if eligible:
                indices.append(position)
            position += 1
    if not indices:
        raise ValueError('No common eligible records')
    selected = data.select(indices)
    if wrapped:
        def flatten(row):
            return {k:np.asarray(row[k]).reshape(-1).tolist() for k in ('input_ids','loss_mask','attention_mask') if k in row}
        selected = selected.map(flatten,desc='Normalizing singleton batch dimensions')
    selected.save_to_disk(str(dest))
    report = {'source':str(source),'source_fingerprint':data._fingerprint,'original_rows':len(data),
              'training_rows':len(selected),'excluded_rows':len(data)-len(selected),
              'policy':'loss sum >=64, at least two DLoop starts, supervised own-method start; no replacement',
              'original_data_unchanged':True}
    write_json(dest/'selection_report.json',report)
    np.save(dest/'selected_source_indices.npy',np.asarray(indices,dtype=np.int64))
    return report

def prepare_output_directory(out):
    """Retry a failed checkpoint check without discarding saved evaluation prompts."""
    out=Path(out)
    if out.exists():
        if not out.is_dir():
            raise FileExistsError(f'Output path is not a directory: {out}')
        entries={p.name for p in out.iterdir()}
        if entries-{'gsm8k.json'} or ('gsm8k.json' in entries and not (out/'gsm8k.json').is_file()):
            raise FileExistsError(f'Use a new RUN_DIR; output already contains preparation/training data: {out}')
    out.mkdir(parents=True,exist_ok=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',required=True)
    p.add_argument('--eval-fraction',type=float,default=0.25)
    p.add_argument('--epochs',type=int,default=1)
    p.add_argument('--anchors-per-sequence',type=int,default=512,
                   help='Anchor budget per 3072-token unit; rank budget is three times this value')
    a=p.parse_args()
    rank_anchors=3*a.anchors_per_sequence
    if a.anchors_per_sequence<1 or rank_anchors<128:
        raise ValueError('Anchor budget must support all 128 possible packed document rows')
    base=Path(os.environ.get('BASE','/home/n84449292/a84384644'))
    model=os.environ.get('MODEL',str(base/'models/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c'))
    source=os.environ.get('DRAFT_SOURCE',str(base/'dfalsh_check/checkpoints/0'))
    data_source=os.environ.get('DATASET',str(base/'datasets/open_perfectblend.qwen3-4b-rollout.qwen3.seq3072.dedup'))
    out=Path(a.run_dir).resolve()
    if a.epochs<1 or not 0<a.eval_fraction<=1: raise ValueError('Invalid epochs/eval fraction')
    prepare_output_directory(out)
    # Prepare validation prompts first, so unavailable cache fails before a long dataset copy.
    prepare_gsm8k(model,out/'gsm8k.json',os.environ.get('GSM8K_SOURCE'),int(os.environ.get('EVAL_LIMIT','128')))
    own_base=convert(source,model,out/'initialization')
    report=prepare_data(data_source,out/'training_data')
    from .shared_packing import install
    install()
    from four_anchor.train import TrainConfig
    from four_anchor.packed_data import prepare_records,epoch_batches,schedule
    cfg=TrainConfig(target=model,data=str(out/'training_data'),output=str(out/'mine/checkpoints'),
        device='npu',dtype='bfloat16',attention='sdpa',mask_token_id=own_base['mask_token_id'],
        target_layer_ids=own_base['aux_hidden_state_layer_ids'],num_draft_layers=5,num_anchors=4,num_proposals=15,
        max_sequence_length=9216,epochs=a.epochs,gradient_accumulation=1,learning_rate=6e-4,weight_decay=0.0,
        initialize_dflash=str(out/'initialization/mine'),checkpoint_each_epoch=False,checkpoint_every=10**9,
        log_every=10,tensorboard=True,training_mode='packed',max_anchors=rank_anchors,blocks_per_chunk=64,head_chunk_size=256,
        train_data_ratio=1.0,num_workers=0,loss_fn={'ce':0.1,'tv':0.9},distributed_strategy='fsdp',
        packed_execution='cached',profile_steps=3,anchor_layout='bonus_addon',data_cache_dir=str(out/'cache/mine'))
    records,lengths,summary=prepare_records(cfg)
    counts=schedule(cfg,records,lengths,4)
    cfg.warmup_steps=int(0.04*sum(counts))
    write_json(out/'mine_config.json',asdict(cfg))
    mine_targets, previous=[],0
    for epoch in range(a.epochs):
        plans=[epoch_batches(cfg,lengths,records._fingerprint,4,r,epoch) for r in range(4)]
        docs=[sum(len(plans[r][step]) for r in range(4)) for step in range(counts[epoch])]
        for point in quarter_plan(docs,a.eval_fraction):
            mine_targets.append(dict(point,step=point['step']+previous,epoch=epoch+1))
        previous+=counts[epoch]
    from ascend_support.packed_data import build_index,load_index,TokenBudgetBatchSampler
    from ascend_support.tokenized_data import SavedTokenDataset
    ds=SavedTokenDataset(str(out/'training_data'),3072,64)
    index=build_index(ds,str(out/'cache/dloop'))
    dl_lengths,replaced=load_index(ds,index)
    if replaced: raise AssertionError('Common filtering must prevent short-row replacement')
    if not np.array_equal(lengths,dl_lengths):
        raise AssertionError('The two methods must use exactly the same documents and lengths')
    sampler=TokenBudgetBatchSampler(dl_lengths,9216,4,seed=42,max_records=128)
    dloop_targets=[]
    for epoch in range(a.epochs):
        sampler.set_epoch(epoch)
        packs=sampler.global_packs()
        docs=[sum(map(len,packs[i:i+4])) for i in range(0,len(packs),4)]
        for point in quarter_plan(docs,a.eval_fraction):
            dloop_targets.append(dict(point,step=point['step']+epoch*len(sampler),epoch=epoch+1))
    if counts!=[len(sampler)]*a.epochs or mine_targets!=dloop_targets:
        raise AssertionError('Shared training/evaluation schedules disagree')
    settings={'model':model,'source_checkpoint':source,'dataset':str(out/'training_data'),
              'dataset_report':report,'epochs':a.epochs,'eval_fraction':a.eval_fraction,
              'eval_limit':len(json.loads((out/'gsm8k.json').read_text())['records']),
              'eval_max_new_tokens':int(os.environ.get('EVAL_MAX_NEW_TOKENS','2048')),
              'loop_gate':-0.5,'stage_cap':16,'world_size_per_job':4,'seed':42,'lr':6e-4,
              'warmup_ratio':0.04,'anchors_per_rank_batch':rank_anchors,'token_budget_per_rank':9216,
              'anchors_per_sequence':a.anchors_per_sequence,'anchor_sequence_length':3072,
              'packing_alignment':'shared_length_buckets_v1',
              'draft_attention':'full_attention','draft_sliding_window':None,
              'mine_steps_per_epoch':counts,'mine_targets':mine_targets,
              'dloop_steps_per_epoch':len(sampler),'dloop_targets':dloop_targets,
              'dloop_packing_index':str(index),'dloop_packing':sampler.describe(),
              'mine_data_summary':summary,'gsm8k_sha256':sha(out/'gsm8k.json'),
              'fast_dloop':os.environ.get('FAST_DLOOP','1')=='1',
              'attention_block_chunk':int(os.environ.get('ATTN_BLOCK_CHUNK','8')),
              'dloop_head_chunk':int(os.environ.get('HEAD_LOSS_CHUNK_SIZE','512'))}
    write_json(out/'settings.json',settings)
    write_json(out/'PREPARED_SHA256.json',
               {name:sha(out/name) for name in ('settings.json','mine_config.json','gsm8k.json')})
    shutil.copy2(ROOT/'SOURCE_COMMITS.json',out/'SOURCE_COMMITS.json')
    if (ROOT/'PACKAGE_SHA256.json').exists(): shutil.copy2(ROOT/'PACKAGE_SHA256.json',out/'PACKAGE_SHA256.json')
    print(json.dumps({'prepared':str(out),'training_rows':report['training_rows'],
                     'anchors_per_sequence':a.anchors_per_sequence,'anchors_per_rank_batch':rank_anchors,
                     'mine_eval_steps':[x['step'] for x in mine_targets],
                     'dloop_eval_steps':[x['step'] for x in dloop_targets]},indent=2),flush=True)

if __name__=='__main__': main()
