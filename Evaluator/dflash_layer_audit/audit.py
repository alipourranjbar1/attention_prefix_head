"""Compare residual outputs of an official HF DFlash drafter and Qwen3 target."""
import argparse
import csv
import hashlib
import json
import math
import random
from contextlib import ExitStack, contextmanager
from pathlib import Path

import numpy as np
import torch
from datasets import DatasetDict, load_from_disk
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

from metrics import report


@contextmanager
def capture(layers, positions):
    outputs = []
    def record(module, inputs, output):
        state = output[0] if isinstance(output, tuple) else output
        outputs.append(state[0, positions, :].detach())
    with ExitStack() as stack:
        for layer in layers:
            stack.callback(layer.register_forward_hook(record).remove)
        yield outputs
    if len(outputs) != len(layers):
        raise RuntimeError('Unexpected number of decoder block outputs')


def prompt_ids(row, tokenizer, marker, limit):
    if 'input_ids' in row and row['input_ids'] is not None:
        ids = np.asarray(row['input_ids']).reshape(-1)
        if row.get('attention_mask') is not None:
            ids = ids[np.asarray(row['attention_mask']).reshape(-1).astype(bool)]
        ids = ids.tolist()
        for i in range(len(ids)-len(marker)+1):
            if ids[i:i+len(marker)] == marker:
                return ids[max(0, i+len(marker)-limit):i+len(marker)]
        return ids[:limit]
    prompt = row.get('prompt') or row.get('text')
    if isinstance(prompt, str):
        return tokenizer.encode(prompt, add_special_tokens=False)[-limit:]
    raise ValueError(f'Expected input_ids, prompt or text. Found {list(row)}')


def prompts(path, tokenizer, args):
    if path.is_file():
        with path.open() as f:
            rows = [json.loads(line) for line in f if line.strip()]
    else:
        rows = load_from_disk(str(path))
        if isinstance(rows, DatasetDict):
            rows = rows[args.split]
    n = len(rows)
    if not n:
        raise ValueError('Empty dataset')
    rng = random.Random(args.seed)
    start = rng.randrange(n)
    stride = rng.randrange(1, n) if n > 1 else 1
    while math.gcd(stride, n) != 1:
        stride += 1
    marker = tokenizer.encode('<|im_start|>assistant\n', add_special_tokens=False)
    seen = set()
    for k in range(n):
        idx = (start + stride*k) % n
        ids = prompt_ids(rows[idx], tokenizer, marker, args.prefix_tokens)
        fingerprint = hashlib.sha256(np.asarray(ids, dtype='<i8').tobytes()).hexdigest()
        if not ids or fingerprint in seen:
            continue
        seen.add(fingerprint)
        yield idx, fingerprint, ids


@torch.inference_mode()
def block_pair(target, draft, ids, device):
    prefix = torch.tensor([ids], device=device)
    prefix_len = len(ids)
    size = draft.block_size
    # Ask target only for the prefix before constructing the draft input.
    prefill = target.model(prefix, output_hidden_states=True, use_cache=False)
    anchor = target.lm_head(prefill.last_hidden_state[:, -1:]).argmax(-1)
    context = torch.cat([prefill.hidden_states[i+1] for i in draft.target_layer_ids], dim=-1)
    del prefill
    noise = anchor.new_full((1, size), draft.mask_token_id)
    noise[:, :1] = anchor
    positions = torch.arange(prefix_len+size, device=device)[None, :]
    with capture(draft.layers, slice(1, size)) as draft_states:
        hidden = draft(target_hidden=context, noise_embedding=target.model.embed_tokens(noise),
                       position_ids=positions, use_cache=False, is_causal=False)
    proposed = target.lm_head(hidden[:, 1:]).argmax(-1)
    # Target position prefix_len+s-1 predicts the token in draft slot s.
    # Replay the actual proposals. No ground-truth continuation enters the draft.
    teacher_input = torch.cat([prefix, anchor, proposed[:, :-1]], dim=1)
    with capture(target.model.layers, slice(prefix_len, prefix_len+size-1)) as target_states:
        output = target.model(teacher_input, use_cache=False)
    decisions = target.lm_head(output.last_hidden_state[:, prefix_len:]).argmax(-1)
    accepted = (decisions == proposed).long().cumprod(-1).sum().item()
    def array(states):
        return torch.stack(states).float().cpu().numpy()
    return array(draft_states), array(target_states), accepted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ('target','draft','dataset','out'):
        parser.add_argument('--'+arg, type=Path, required=True)
    parser.add_argument('--device', default='npu:0')
    parser.add_argument('--fit', type=int, default=64)
    parser.add_argument('--holdout', type=int, default=32)
    parser.add_argument('--prefix-tokens', type=int, default=256)
    parser.add_argument('--ridge', type=float, default=.01)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--split', default='train')
    args = parser.parse_args()
    if args.fit < 2 or args.holdout < 4 or args.prefix_tokens < 1 or args.ridge <= 0:
        parser.error('Require fit >= 2, holdout >= 4, positive prefix length and ridge')
    for path in (args.target, args.draft, args.dataset):
        if not path.exists():
            parser.error(f'Missing local path {path}')
    if args.out.exists() and any(args.out.iterdir()):
        parser.error('Choose an empty output directory')
    if args.device.startswith('npu'):
        import torch_npu  # registers the NPU backend
    dtype = torch.float32 if args.device == 'cpu' else torch.bfloat16
    opts = dict(local_files_only=True, dtype=dtype, attn_implementation='eager')
    tokenizer = AutoTokenizer.from_pretrained(args.target, local_files_only=True)
    def load(cls, path, **kw):
        model, info = cls.from_pretrained(path, output_loading_info=True, **opts, **kw)
        if any(info.get(k) for k in ('missing_keys','mismatched_keys','error_msgs')):
            raise RuntimeError(f'Incomplete model weights for {path}: {info}')
        return model.to(args.device).eval()
    target = load(AutoModelForCausalLM, args.target)
    draft = load(AutoModel, args.draft, trust_remote_code=True)
    if (draft.config.num_target_layers != len(target.model.layers)
            or draft.config.hidden_size != target.config.hidden_size):
        raise ValueError('Drafter dimensions do not match target')
    args.out.mkdir(parents=True, exist_ok=True)
    count = args.fit + args.holdout
    width = target.config.hidden_size
    def allocate(layers):
        return np.empty((count, layers, draft.block_size-1, width), dtype=np.float32)
    d, t = allocate(len(draft.layers)), allocate(len(target.model.layers))
    with (args.out/'prompts.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['split','dataset_row','prefix_sha256','prefix_tokens','accepted_draft_tokens'])
        source = iter(prompts(args.dataset, tokenizer, args))
        for i in range(count):
            entry = next(source, None)
            if entry is None:
                raise ValueError(f'Only {i} distinct usable prefixes for {count} requested')
            idx, key, ids = entry
            d[i], t[i], accepted = block_pair(target, draft, ids, args.device)
            writer.writerow(['fit' if i < args.fit else 'holdout',idx,key,len(ids),accepted])
            print(f'[{i+1}/{count}] row {idx} | accepted {accepted}/{draft.block_size-1}', flush=True)
    report(d, t, args.fit, args.ridge, args.out)
    metadata = {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()}
    metadata.update(draft_shape=d.shape, target_shape=t.shape, conditioning_blocks=draft.target_layer_ids,
                    penultimate_target_block=len(target.model.layers)-2, torch_version=torch.__version__)
    (args.out/'run.json').write_text(json.dumps(metadata, indent=2))
    print(f'Complete: {args.out}/best_layers.csv', flush=True)


if __name__ == '__main__':
    main()
