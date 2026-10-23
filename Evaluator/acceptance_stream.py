"""One dataset traversal; frozen inference; each saved block trained once."""
import argparse
import importlib.util
import json
import os
import time
import urllib.request
import urllib.error
from pathlib import Path
from datetime import timedelta

import torch

ROOT = Path(os.environ['SPEC_MAIN'])
OUT = Path(os.environ['EXPERIMENT'])
TRACES = OUT / 'traces'
DONE = OUT / 'collection.done'
FAILED = OUT / 'collection.failed'


def module(name):
    path = ROOT / 'speculators/src/speculators/models/dflash_prefix' / (name + '.py')
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def atomic_json(path, payload):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(payload))
    tmp.replace(path)


def request(url, payload=None):
    req = urllib.request.Request(url, data=None if payload is None else json.dumps(payload).encode(),
                                 headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=1800) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        raise RuntimeError(error.read().decode()) from error


def collect():
    from datasets import load_from_disk, DatasetDict
    from tqdm.auto import tqdm
    dataset = load_from_disk(os.environ['DATASET'])
    if isinstance(dataset, DatasetDict):
        raise ValueError('Expected the single Dataset you inspected, not a DatasetDict')
    dataset = dataset.with_format(None)
    endpoint = 'http://127.0.0.1:' + os.environ['PORT']
    name = os.environ['SERVED_MODEL_NAME']
    models = request(endpoint + '/v1/models')['data']
    if name not in [entry['id'] for entry in models]:
        raise RuntimeError('Wrong model served on the collection port')
    completed, skipped = 0, 0
    total = len(dataset)
    with (OUT / 'samples.jsonl').open('w') as log:
        for index in tqdm(range(total), desc='Dataset pass 1/1', unit='sample', dynamic_ncols=True):
            row = dataset[index]
            ids, mask = row['input_ids'], row['loss_mask']
            seq_len = int(row['seq_len'])
            if not 0 < seq_len <= len(ids) or len(ids) != len(mask):
                raise ValueError(f'Invalid token/mask lengths at row {index}')
            boundary = next((i for i, flag in enumerate(mask[:seq_len]) if flag), None)
            if boundary is None or boundary == 0:
                # No recoverable prompt/response boundary: report, never silently invent one.
                skipped += 1
                log.write(json.dumps({'row': index, 'skipped': 'no usable prompt boundary'}) + '\n')
            else:
                prompt = [int(token) for token in ids[:boundary]]
                # Generate up to the response budget of THIS dataset row, not a 256-token cap.
                budget = seq_len - boundary
                if seq_len > int(os.environ['MAX_MODEL_LEN']):
                    raise ValueError(f'Row {index} exceeds MAX_MODEL_LEN; refusing truncation')
                result = request(endpoint + '/v1/completions', {
                    'model': name, 'prompt': prompt, 'temperature': 0,
                    'max_tokens': budget, 'seed': 42, 'stream': False,
                })
                completed += 1
                log.write(json.dumps({'row': index, 'prompt_tokens': boundary,
                                     'max_tokens': budget, 'usage': result.get('usage')}) + '\n')
            log.flush()
            atomic_json(OUT / 'progress.json', dict(dataset_rows=total, visited=index + 1,
                        completed=completed, skipped=skipped, remaining=total-index-1))
    DONE.touch()


class Queue:
    def __init__(self):
        self.seen = set()
        self.pending = []
        self.identity = None

    def refresh(self):
        if FAILED.exists():
            raise RuntimeError(FAILED.read_text())
        for manifest in sorted(TRACES.rglob('manifest.json')):
            identity = json.loads(manifest.read_text())
            if self.identity is None:
                self.identity = identity
            if identity != self.identity:
                raise RuntimeError('Mixed trace identities')
            for path in sorted(manifest.parent.glob('*.pt')):
                path = str(path)
                if path not in self.seen:
                    self.seen.add(path)
                    self.pending.append(path)

    def take(self, count):
        result = self.pending[:count]
        del self.pending[:count]
        return result


def train():
    import torch_npu
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.tensorboard import SummaryWriter
    rank, local = int(os.environ['RANK']), int(os.environ['LOCAL_RANK'])
    torch.npu.set_device(local)
    device = torch.device(f'npu:{local}')
    dist.init_process_group('hccl', timeout=timedelta(hours=2))
    world = dist.get_world_size()
    torch.manual_seed(42)
    batch_size = int(os.environ['BATCH_SIZE'])
    global_batch = world * batch_size
    queue = Queue() if rank == 0 else None
    plan_path = OUT / 'batch_plan.json'
    head = model = optimizer = writer = None
    step = trained = 0
    loss_fn = module('acceptance_loss').acceptance_boundary_loss
    while True:
        if rank == 0:
            while True:
                # Observe DONE BEFORE scanning, so the last committed records are included.
                finished = DONE.exists()
                queue.refresh()
                if len(queue.pending) >= global_batch or finished:
                    break
                time.sleep(2)
            paths = queue.take(global_batch)
            atomic_json(plan_path, dict(paths=paths, identity=queue.identity, finished=finished))
        dist.barrier()
        plan = json.loads(plan_path.read_text())
        paths = plan['paths']
        if not paths:
            break
        if head is None:
            example = torch.load(paths[0], map_location='cpu', weights_only=True)['features']
            k, dimension = example.shape
            if k != plan['identity']['proposals'] or dimension <= 5:
                raise ValueError('Invalid feature schema')
            architecture = dict(hidden_size=dimension-5, num_proposals=k,
                                projection_size=32, mlp_hidden_size=64)
            head = module('acceptance_head').AcceptedLengthHead(**architecture).to(device)
            model = DDP(head, device_ids=[local], broadcast_buffers=False)
            optimizer = torch.optim.AdamW(model.parameters(), lr=float(os.environ['LR']), weight_decay=0.01)
            if rank == 0:
                writer = SummaryWriter(str(OUT / 'tensorboard'))
                (OUT / 'checkpoints').mkdir()
                print(head, flush=True)
                print('Only the MLP is optimized. Each trace is consumed once.', flush=True)
        # Disjoint rows across ranks; no DistributedSampler padding/replay.
        local_paths = paths[rank::world]
        features, labels = [], []
        for path in local_paths:
            row = torch.load(path, map_location='cpu', weights_only=True)
            x, y = row['features'], int(row['accepted_length'])
            if tuple(x.shape) != (k, dimension) or not torch.isfinite(x.float()).all() or not 0 <= y <= k:
                raise ValueError(f'Invalid record: {path}')
            features.append(x.float())
            labels.append(y)
        optimizer.zero_grad(set_to_none=True)
        if local_paths:
            logits = model(torch.stack(features).to(device))
            target = torch.tensor(labels, dtype=torch.long, device=device)
            loss, _ = loss_fn(logits, target, under_weight=3., over_weight=1., cost_weight=1.)
            error = logits.detach().argmax(-1) - target
            totals = torch.stack((loss.detach()*len(labels), error.eq(0).sum().float(),
                                  (-error).clamp_min(0).sum().float(), error.clamp_min(0).sum().float()))
            objective = loss * (world * len(labels) / len(paths))
        else:
            # Tail batch may contain fewer records than ranks. Dummy has ZERO supervision.
            logits = model(torch.zeros(1, k, dimension, device=device))
            objective = logits.sum() * 0.
            totals = torch.zeros(4, device=device)
        objective.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        dist.all_reduce(totals)
        step += 1
        trained += len(paths)
        if rank == 0:
            values = (totals / len(paths)).cpu().tolist()
            metrics = dict(loss=values[0], exact_accuracy=values[1],
                           mean_under_tokens=values[2], mean_over_tokens=values[3],
                           decision_cost=3*values[2]+values[3])
            for name, value in metrics.items():
                writer.add_scalar('train/' + name, value, step)
            writer.add_scalar('progress/training_blocks', trained, step)
            if (OUT / 'progress.json').exists():
                progress = json.loads((OUT / 'progress.json').read_text())
                for name in ('visited', 'remaining', 'skipped'):
                    writer.add_scalar('dataset/' + name, progress[name], step)
            report = dict(step=step, unique_training_blocks=trained, **metrics)
            print(json.dumps(report), flush=True)
            with (OUT / 'metrics.jsonl').open('a') as log:
                log.write(json.dumps(report) + '\n')
            if step % int(os.environ['SAVE_EVERY']) == 0 or plan['finished']:
                payload = dict(architecture=architecture, identity=plan['identity'],
                               state_dict={name:value.detach().cpu() for name,value in head.state_dict().items()},
                               step=step, training_blocks=trained, metrics=metrics,
                               loss_settings=dict(under_weight=3.,over_weight=1.,cost_weight=1.))
                temp = OUT / 'checkpoints/last.tmp'
                torch.save(payload, temp)
                temp.replace(OUT / 'checkpoints/last.pt')
            writer.flush()
        # Ensure the next plan cannot overwrite the current one before every rank reads it.
        dist.barrier()
    if head is None:
        raise RuntimeError('No verification records were collected')
    if rank == 0:
        # Also save when the final full batch was consumed just before DONE appeared.
        torch.save(dict(architecture=architecture, identity=plan['identity'],
                        state_dict={name:value.detach().cpu() for name,value in head.state_dict().items()},
                        step=step, training_blocks=trained,
                        loss_settings=dict(under_weight=3.,over_weight=1.,cost_weight=1.)),
                   OUT / 'checkpoints/final.pt')
        writer.close()
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['collect', 'train'])
    args = parser.parse_args()
    if args.mode == 'collect':
        try:
            collect()
        except BaseException as error:
            FAILED.write_text(repr(error))
            raise
    else:
        train()
