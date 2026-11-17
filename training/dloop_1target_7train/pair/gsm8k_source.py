"""Load local frozen GSM8K benchmark inputs without requiring Hub access."""
from pathlib import Path
import json
from numbers import Integral
import numpy as np
from datasets import DatasetDict, load_dataset, load_from_disk
from transformers import AutoConfig, AutoTokenizer
from .common import write_json, sha

MATH_SUFFIX = '\nPlease reason step by step, and put your final answer within \\boxed{}.'


def token_ids(row, vocabulary_size):
    key = next((k for k in ('prompt_token_ids','input_ids') if row.get(k) is not None),None)
    if key is None:
        return None
    if 'loss_mask' in row:
        raise ValueError('Evaluation requires prompt-only token IDs, not training records with loss_mask')
    ids = row[key]
    if isinstance(ids,list) and len(ids)==1 and isinstance(ids[0],list):
        ids=ids[0]
    if not isinstance(ids,list) or not ids or any(isinstance(v,bool) or not isinstance(v,Integral) for v in ids):
        raise ValueError(f'{key} must be a nonempty integer sequence')
    if min(ids)<0 or max(ids)>=vocabulary_size:
        raise ValueError('Saved benchmark token ID is outside the target vocabulary')
    if row.get('attention_mask') is not None:
        mask=np.asarray(row['attention_mask']).reshape(-1)
        if len(mask)!=len(ids) or not np.all(mask==1):
            raise ValueError('Saved benchmark token IDs must be unpadded')
    return [int(v) for v in ids]


def text_tokens(row, tokenizer):
    if row.get('messages') is not None:
        messages=row['messages']
        if (not isinstance(messages,list) or not messages
            or any(not isinstance(m,dict) or m.get('role') not in ('system','user')
                   or not isinstance(m.get('content'),str) for m in messages)
            or messages[-1]['role']!='user'):
            raise ValueError('GSM8K messages must contain only system/user prompts ending in a user turn')
    else:
        text=None
        for key in ('question','turns','prompt','text'):
            value=row.get(key)
            if isinstance(value,list) and len(value)==1:
                value=value[0]
            if isinstance(value,str) and value.strip():
                text=value
                if key=='question' and MATH_SUFFIX not in text:
                    text+=MATH_SUFFIX
                break
        if text is None:
            raise ValueError('Need prompt_token_ids, input_ids, question, prompt, text, turns, or messages')
        # Preserve already-rendered Qwen chat prompts; never double-wrap them.
        if text.lstrip().startswith('<|im_start|>'):
            return tokenizer.encode(text,add_special_tokens=False)
        messages=[{'role':'user','content':text}]
    return tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,
                                         enable_thinking=False,return_dict=False)


def load_rows(source):
    if not source:
        data=load_dataset('openai/gsm8k','main',split='test')
        return list(data),None
    path=Path(source).expanduser().resolve()
    if path.is_dir():
        data=load_from_disk(str(path))
        if isinstance(data,DatasetDict): data=data['test']
        return list(data),None
    if not path.is_file():
        raise FileNotFoundError(path)
    # Direct JSON parsing does not construct an HF builder or consult the Hub.
    if path.suffix.lower()=='.jsonl':
        with path.open() as stream:
            rows=[json.loads(line) for line in stream if line.strip()]
    else:
        value=json.loads(path.read_text())
        rows=value.get('records') if isinstance(value,dict) else value
    if not isinstance(rows,list) or not all(isinstance(row,dict) for row in rows):
        raise ValueError('Expected JSONL objects, a JSON row list, or a records array')
    return rows,sha(path)


def prepare_gsm8k(model,dest,source=None,limit=128):
    if limit<1: raise ValueError('EVAL_LIMIT must be positive')
    rows,source_sha=load_rows(source)
    if not rows: raise ValueError('GSM8K source has no rows')
    # A supplied benchmark file already defines the comparison order/subset.
    indices=list(range(min(limit,len(rows)))) if source else np.random.default_rng(0).permutation(len(rows))[:limit]
    cfg=AutoConfig.from_pretrained(model,trust_remote_code=False,local_files_only=True)
    tokenizer=None
    records=[]
    modes=set()
    for index in indices:
        row=rows[int(index)]
        tokens=token_ids(row,cfg.vocab_size)
        mode='saved_token_ids'
        if tokens is None:
            if tokenizer is None:
                tokenizer=AutoTokenizer.from_pretrained(model,trust_remote_code=False,local_files_only=True)
            tokens=text_tokens(row,tokenizer)
            tokens=token_ids({'input_ids':tokens},cfg.vocab_size)
            mode='tokenized_text'
        modes.add(mode)
        answer=row.get('answer',row.get('gold_answer',row.get('reference_answer')))
        if answer is not None and not isinstance(answer,(str,int,float)):
            raise ValueError('Reference answer must be text, a number, or null')
        records.append({'source_index':row.get('source_index',row.get('source_row',int(index))),
                        'input_ids':tokens,'question':row.get('question'),'answer':answer})
    result={'source':str(Path(source).expanduser().resolve()) if source else 'openai/gsm8k/main/test',
            'source_sha256':source_sha,'selection':'local_file_order' if source else 'seed_0_shuffle',
            'seed':None if source else 0,'thinking':'preserved_in_saved_ids' if modes=={'saved_token_ids'} else False,
            'input_modes':sorted(modes),'source_rows':len(rows),'records':records,
            'reference_answers_present':sum(r['answer'] is not None for r in records)}
    dest=Path(dest)
    reused=dest.exists()
    if reused:
        if json.loads(dest.read_text())!=result:
            raise ValueError('Saved GSM8K prompts differ from the requested source/limit; use a new RUN_DIR')
    else:
        write_json(dest,result)
    print(json.dumps({'gsm8k_source':result['source'],'prompts':len(records),
                      'input_modes':result['input_modes'],'reference_answers_present':result['reference_answers_present'],
                      'reused_prepared_prompts':reused}),flush=True)
    return result
