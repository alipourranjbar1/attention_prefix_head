"""Four independent HF evaluators, same frozen GSM8K questions and greedy target.

Own inference uses the trained bonus-add-on 20/19 layout, not the legacy Decoder.
DLoop uses its pinned native linear decoder with loop_gate=-0.5.
"""
from pathlib import Path
from decimal import Decimal, InvalidOperation
import argparse
import json
import os
import re
import time
import torch
from .common import write_json, sha


def sync(device):
    if device.type in ('npu','cuda'):
        getattr(torch,device.type).synchronize(device)


def answer_number(text, gold=False):
    if text is None:
        return None
    text=str(text)
    if gold:
        values=text.split('####')
        if len(values)>=2:
            value=values[-1].strip()
        else:
            boxes=re.findall(r'\\boxed\{([^{}]+)\}',text)
            value=boxes[-1].strip() if boxes else text.strip()
    else:
        boxes=re.findall(r'\\boxed\{([^{}]+)\}',text)
        if boxes: value=boxes[-1].strip()
        elif '####' in text: value=text.split('####')[-1].strip()
        else: return None
    value=value.replace(',','').replace('$','').strip()
    try:
        n=Decimal(value)
        return str(n.normalize()) if n.is_finite() else None
    except InvalidOperation:
        return None


@torch.inference_mode()
def generate_mine(target, draft, input_ids, max_new_tokens, eos_ids):
    from four_anchor.runtime import TargetRunner
    from four_anchor.batch_inference import BatchedBonusAddonDraft
    runner=TargetRunner(target,draft.cfg.target_layer_ids)
    n=input_ids.shape[1]
    capacity=n+max_new_tokens+64
    inference=BatchedBonusAddonDraft(draft,capacity,max_requests=1)
    try:
        out,features=runner.forward(input_ids)
        cache=out.past_key_values
        bonus=out.logits[:,-1].argmax(-1,keepdim=True)
        inference.cache.update([(0,0,input_ids[0],features[0])],[0])
        result,acceptances,boundaries=[],[],[]
        while len(result)<max_new_tokens:
            proposed,boundary=inference.propose([0],bonus.reshape(-1),mode='two_pass')
            block=torch.cat((bonus,proposed),dim=1)
            out,features=runner.forward(block,cache,all_logits=True)
            posterior=out.logits.argmax(-1)
            accepted=int((proposed==posterior[:,:-1]).long().cumprod(-1).sum().item())
            committed=accepted+1
            extension=block[0,:committed].tolist()[:max_new_tokens-len(result)]
            eos_at=next((i for i,t in enumerate(extension) if t in eos_ids),None)
            if eos_at is not None: extension=extension[:eos_at+1]
            result.extend(extension)
            acceptances.append(committed)
            boundaries.append(int(boundary.item()))
            if eos_at is not None or len(result)>=max_new_tokens: break
            bonus=posterior[:,accepted:accepted+1]
            cache=out.past_key_values
            cache.crop(n+committed)
            inference.cache.update([(0,n,block[0,:committed],features[0,:committed])],[0])
            n+=committed
        return result,acceptances,{'redraft_boundaries':boundaries}
    finally:
        runner.close()


@torch.inference_mode()
def greedy_reference(target, ids, limit, eos_ids):
    out=target(ids,use_cache=True,logits_to_keep=1)
    result=[]
    for _ in range(limit):
        token=out.logits[:,-1:].argmax(-1)
        result.append(int(token.item()))
        if result[-1] in eos_ids: break
        out=target(token,past_key_values=out.past_key_values,use_cache=True,logits_to_keep=1)
    return result


def aggregate(directory, world, expected_count):
    directory=Path(directory)
    rows=[]
    for rank in range(world):
        rows.extend(json.loads((directory/f'rank-{rank}.json').read_text())['records'])
    rows.sort(key=lambda x:x['eval_index'])
    if [x['eval_index'] for x in rows] != list(range(expected_count)):
        raise ValueError('Missing/duplicate GSM8K evaluation records')
    cycles=sum(len(x['acceptance_lengths']) for x in rows)
    accepted=sum(sum(x['acceptance_lengths']) for x in rows)
    per_rank=[sum(x['seconds'] for x in rows if x['rank']==r) for r in range(world)]
    scored=[x for x in rows if x.get('correct') is not None]
    summary={'prompts':len(rows),'cycles':cycles,'acceptance_length_including_bonus':accepted/max(cycles,1),
             'accepted_draft_per_cycle':accepted/max(cycles,1)-1,
             'gsm8k_numeric_exact_match':sum(x['correct'] for x in scored)/len(scored) if scored else None,
             'accuracy_scored_prompts':len(scored),
             'accuracy_unscored_prompts':len(rows)-len(scored),
             'accuracy_status':'available' if len(scored)==len(rows) else 'partial' if scored else 'unavailable_no_parseable_references',
             'answer_format':'last boxed number or #### number; strict numeric match',
             'no_parseable_answer':sum(x['predicted_answer'] is None for x in rows),
             'truncated_generations':sum(x['truncated'] for x in rows),
             'output_tokens':sum(len(x['output_ids']) for x in rows),
             'aggregate_tokens_per_second':sum(len(x['output_ids']) for x in rows)/max(max(per_rank),1e-9),
             'concurrency_per_device':1,'devices':world,'graphs':False,
             'timing':'generation including prefill; excludes model load and parity checks',
             'greedy_parity_checks':sum(x['greedy_parity_checked'] for x in rows),
             'acceptance_note':'pooled over verifier rounds; includes full final round before EOS/length truncation'}
    write_json(directory/'summary.json',summary)
    write_json(directory/'all_records.json',rows)
    print(json.dumps(summary,indent=2),flush=True)
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method',choices=['mine','dloop'],required=True)
    p.add_argument('--run-dir',required=True)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--limit',type=int,default=None,help='Use only the first N frozen prompts')
    p.add_argument('--max-new-tokens',type=int,default=None)
    p.add_argument('--parity-tokens',type=int,default=128)
    a=p.parse_args()
    if any(x is not None and x < 1 for x in (a.limit,a.max_new_tokens,a.parity_tokens)):
        p.error('Evaluation limits must be positive')
    import torch_npu
    from transformers import AutoModelForCausalLM,AutoTokenizer
    run=Path(a.run_dir)
    settings=json.loads((run/'settings.json').read_text())
    max_new_tokens=a.max_new_tokens or settings['eval_max_new_tokens']
    if sha(run/'gsm8k.json') != settings['gsm8k_sha256']:
        raise ValueError('Frozen evaluation prompts were changed')
    rank=int(os.environ.get('RANK','0'))
    world=int(os.environ.get('WORLD_SIZE','1'))
    local=int(os.environ.get('LOCAL_RANK','0'))
    device=torch.device(f'npu:{local}')
    torch.npu.set_device(device)
    torch.manual_seed(42)
    target=AutoModelForCausalLM.from_pretrained(settings['model'],dtype=torch.bfloat16,
                attn_implementation='sdpa',trust_remote_code=False).to(device).eval().requires_grad_(False)
    tokenizer=AutoTokenizer.from_pretrained(settings['model'],trust_remote_code=False)
    eos=target.generation_config.eos_token_id
    eos=[] if eos is None else eos if isinstance(eos,list) else [eos]
    if a.method=='mine':
        from four_anchor.model import FourAnchorDFlash
        draft=FourAnchorDFlash.load(a.checkpoint,target).eval()
        def generate(ids,limit): return generate_mine(target,draft,ids,limit,eos)
    else:
        from specforge.modeling.draft.dflash import DFlashDraftModel
        from . import linear_decode_ascend as linear_decode
        draft=DFlashDraftModel.from_pretrained(a.checkpoint,dtype=torch.bfloat16,
                    attn_implementation='sdpa').to(device).eval().requires_grad_(False)
        def clock():
            sync(device)
            return time.perf_counter()
        linear_decode._cuda_time=clock
        def generate(ids,limit):
            out=linear_decode.dflash_generate(draft,target,ids,limit,eos,loop_gate=settings['loop_gate'],
                                             stage_cap=settings['stage_cap'],return_stats=True)
            return out.output_ids[0,ids.shape[1]:].tolist(),out.acceptance_lengths,{'stages_fired':out.stages_fired}
    data=json.loads((run/'gsm8k.json').read_text())['records']
    if a.limit is not None:
        data=data[:a.limit]
    rows=[]
    for index in range(rank,len(data),world):
        item=data[index]
        ids=torch.tensor([item['input_ids']],device=device,dtype=torch.long)
        parity=index<2
        if parity:
            short=min(a.parity_tokens,max_new_tokens)
            actual,_,_=generate(ids,short)
            expected=greedy_reference(target,ids,short,eos)
            if actual!=expected: raise AssertionError(f'{a.method}: greedy parity failed for GSM8K row {index}')
        sync(device)
        start=time.perf_counter()
        tokens,lengths,extra=generate(ids,max_new_tokens)
        sync(device)
        seconds=time.perf_counter()-start
        text=tokenizer.decode(tokens,skip_special_tokens=True)
        prediction=answer_number(text)
        gold=answer_number(item.get('answer'),gold=True)
        correct=prediction==gold if gold is not None else None
        rows.append({'eval_index':index,'source_index':item['source_index'],'rank':rank,
            'output_ids':tokens,'text':text,'acceptance_lengths':lengths,'seconds':seconds,
            'predicted_answer':prediction,'gold_answer':gold,'correct':correct,
            'truncated':len(tokens)>=max_new_tokens and tokens[-1] not in eos,
            'greedy_parity_checked':parity,**extra})
        print(json.dumps({'method':a.method,'rank':rank,'gsm8k_index':index,
                          'acceptance':sum(lengths)/max(len(lengths),1),'correct':correct}),flush=True)
    write_json(Path(a.output)/f'rank-{rank}.json',{'records':rows,
        'eval_count':len(data),'max_new_tokens':max_new_tokens,'parity_tokens':a.parity_tokens})

if __name__=='__main__': main()
