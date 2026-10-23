#!/usr/bin/env python3
"""Generate local benchmark responses and save ArrowDataset-compatible rows."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
import urllib.request

DEFAULT_CACHE = ('/home/n84449292/.cache/huggingface/datasets/'
                 'openai___openai_humaneval/openai_humaneval/0.0.0/'
                 '7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544/openai_humaneval-test.arrow')


def extract_ids(choice):
    entries = (choice.get('logprobs') or {}).get('tokens')
    if not isinstance(entries,list) or not entries:
        raise ValueError('Missing output-token logprobs; cannot preserve exact generated IDs')
    ids=[]
    for token in entries:
        match=re.fullmatch(r'token_id:\s*(\d+)',str(token))
        if not match:
            raise ValueError('Server did not honor return_tokens_as_token_ids; received '+repr(token))
        ids.append(int(match.group(1)))
    return ids


def make_record(prompt_ids, response_ids):
    if not prompt_ids or not response_ids:
        raise ValueError('Empty prompt or response')
    ids=list(prompt_ids)+list(response_ids)
    return dict(input_ids=ids, loss_mask=[False]*len(prompt_ids)+[True]*len(response_ids), seq_len=len(ids))


def main():
    from datasets import Dataset, Features, Sequence, Value, load_from_disk
    from transformers import AutoTokenizer, AutoConfig
    import torch
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--endpoint',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--manifest',required=True)
    p.add_argument('--benchmark',required=True)
    p.add_argument('--max-new-tokens',type=int,default=1024)
    p.add_argument('--concurrency',type=int,default=4)
    args=p.parse_args()
    if args.max_new_tokens<1 or args.concurrency<1: p.error('Invalid limit or concurrency')
    checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    settings=checkpoint['settings']
    model=settings['target']; seq_len=int(settings['seq_len'])
    block=int(checkpoint['architecture']['num_proposals'])+1
    if args.max_new_tokens<block: p.error('max-new-tokens must be at least the draft block size')
    out=Path(args.output).resolve()
    if out.exists(): raise ValueError('Choose a new output directory: '+str(out))
    from benchmark_sources import read_source, make_prompt
    manifest=json.loads(Path(args.manifest).read_text())
    entry=manifest['benchmarks'][args.benchmark]
    source=read_source(entry['source'])
    count=entry['selected_rows']
    tokenizer=AutoTokenizer.from_pretrained(model,local_files_only=True)
    cfg=AutoConfig.from_pretrained(model,local_files_only=True)
    cfg=getattr(cfg,'text_config',cfg)
    eos=getattr(cfg,'eos_token_id',None)
    eos=set([] if eos is None else [eos] if isinstance(eos,int) else eos)
    endpoint=args.endpoint.rstrip('/')
    def request(path, payload=None):
        opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
        data=None if payload is None else json.dumps(payload).encode()
        req=urllib.request.Request(endpoint+path,data=data,headers={'Content-Type':'application/json'})
        with opener.open(req,timeout=600) as response: return json.load(response)
    models=request('/models')['data']
    if model not in [item['id'] for item in models]:
        raise ValueError('Server model ID does not match checkpoint target')
    jobs=[]
    for index,row in enumerate(source.select(range(count))):
        task,message=make_prompt(args.benchmark,row,index)
        prompt_ids=tokenizer.apply_chat_template([{'role':'user','content':message}],
            tokenize=True,add_generation_prompt=True,enable_thinking=False)
        from collections.abc import Mapping
        if isinstance(prompt_ids, Mapping):
            prompt_ids = prompt_ids['input_ids']
        if hasattr(prompt_ids, 'tolist'):
            prompt_ids = prompt_ids.tolist()
        if prompt_ids and isinstance(prompt_ids[0], (list, tuple)):
            if len(prompt_ids) != 1:
                raise ValueError('Expected exactly one tokenized prompt')
            prompt_ids = prompt_ids[0]
        prompt_ids = [int(token) for token in prompt_ids]
        budget=min(args.max_new_tokens,seq_len-len(prompt_ids))
        if budget<block:
            raise ValueError(f"Prompt {task} leaves fewer than {block} response tokens; no truncation performed")
        jobs.append((task,prompt_ids,budget))
    out.mkdir(parents=True)
    config=dict(vars(args),target=model,sequence_budget=seq_len,source_rows=len(source),selected_tasks=count,
                decoding='greedy',thinking=False,
                dataset_overlap_with_training='not_checked',
                note='Response-context acceptance evaluation, not answer correctness', source=entry)
    (out/'preparation.json').write_text(json.dumps(config,indent=2))
    def generate(job):
        task,prompt_ids,budget=job
        payload=dict(model=model,prompt=prompt_ids,temperature=0.,top_p=1.,max_tokens=budget,
                     n=1,stream=False,echo=False,logprobs=1,return_tokens_as_token_ids=True,
                     frequency_penalty=0.,presence_penalty=0.,repetition_penalty=1.,seed=42)
        response=request('/completions',payload)
        choice=response['choices'][0]
        if choice.get('finish_reason') not in ('stop','length'):
            raise RuntimeError(f"Unexpected finish reason for {task}: {choice.get('finish_reason')}")
        ids=extract_ids(choice)
        # EOS is not a response context for subsequent blocks.
        if any(t in eos for t in ids): ids=ids[:next(i for i,t in enumerate(ids) if t in eos)]
        record=make_record(prompt_ids,ids)
        if record['seq_len']>seq_len: raise RuntimeError('Server exceeded requested token budget')
        detail=dict(task_id=task,prompt_tokens=len(prompt_ids),response_tokens=len(ids),
                    finish_reason=choice['finish_reason'],eligible_for_full_blocks=len(ids)>=block,
                    input_ids=record['input_ids'],response_text=choice.get('text',''))
        return record,detail
    rows=[]; details=[]
    with (out/'responses.jsonl').open('w') as log, ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        for i,(record,detail) in enumerate(pool.map(generate,jobs),1):
            rows.append(record); details.append(detail)
            log.write(json.dumps(detail)+'\n'); log.flush()
            print(f"Generated {i}/{count}: {detail['task_id']} ({detail['response_tokens']} response tokens)",flush=True)
    if not any(d['eligible_for_full_blocks'] for d in details):
        raise RuntimeError('No response is long enough for a full block')
    schema=Features({'input_ids':Sequence(Value('int64')),'loss_mask':Sequence(Value('bool')),'seq_len':Value('int64')})
    destination=out/'dataset'
    Dataset.from_list(rows,features=schema).save_to_disk(str(destination))
    # Verify the exact loader contract without starting hidden-state extraction.
    restored=load_from_disk(str(destination)).with_format('torch')
    assert len(restored)==count
    for row in restored:
        assert row['input_ids'].numel()==int(row['seq_len'])==row['loss_mask'].numel()
        assert row['loss_mask'].dtype==torch.bool
    summary=dict(tasks=count,short_responses=sum(not d['eligible_for_full_blocks'] for d in details),
                 length_limited_responses=sum(d['finish_reason']=='length' for d in details),
                 total_response_tokens=sum(d['response_tokens'] for d in details),dataset=str(destination))
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__': main()
