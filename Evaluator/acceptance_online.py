"""Online head training over atomic verification records from a frozen server."""
import importlib.util
import json
import os
import random
import hashlib
import time
from datetime import timedelta
from pathlib import Path

import torch

OUT = Path(os.environ['EXPERIMENT'])
TRACE = Path(os.environ['VLLM_ASCEND_PREFIX_ACCEPTANCE_TRACE_DIR'])
spec = importlib.util.spec_from_file_location('cached_pipeline', OUT / 'pipeline.py')
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
DONE = OUT / 'online_collection.done'
FAILED = OUT / 'online_collection.failed'


def collect():
    from datasets import load_from_disk, DatasetDict
    data = load_from_disk(os.environ['DATASET'])
    if isinstance(data, DatasetDict):
        data = data['train'] if 'train' in data else data[next(iter(data))]
    data = data.with_format(None)
    seen = set()
    log_path = OUT / 'collection.jsonl'
    if log_path.exists():
        for line in log_path.read_text().splitlines():
            try:
                seen.add(json.loads(line)['prompt_key'])
            except (ValueError, KeyError):
                raise RuntimeError('Invalid collection log; inspect its last line before resuming')
    wanted = int(os.environ['NUM_PROMPTS'])
    endpoint = f"http://127.0.0.1:{os.environ['PORT']}"
    model = os.environ['SERVED_MODEL_NAME']
    available = base.request_json(endpoint + '/v1/models')['data']
    if model not in [entry['id'] for entry in available]:
        raise RuntimeError('Server model name mismatch')
    indices = list(range(len(data)))
    random.Random(int(os.environ['SEED'])).shuffle(indices)
    print(f'Resuming: {len(seen)} completed distinct prompts', flush=True)
    with log_path.open('a') as log:
        for index in indices:
            if len(seen) >= wanted:
                break
            row = data[index]
            ids, mask = row['input_ids'], row['loss_mask']
            if len(ids) != len(mask):
                raise RuntimeError(f'Token/mask mismatch: row {index}')
            boundary = next((i for i, flag in enumerate(mask) if flag), None)
            if boundary is None or not 0 < boundary <= int(os.environ['MAX_PROMPT_TOKENS']):
                continue
            prompt = [int(token) for token in ids[:boundary]]
            key = hashlib.sha256(json.dumps(prompt, separators=(',', ':')).encode()).hexdigest()
            if key in seen:
                continue
            result = base.request_json(endpoint + '/v1/completions', {
                'model': model, 'prompt': prompt, 'temperature': 0,
                'max_tokens': int(os.environ['MAX_NEW_TOKENS']),
                'seed': int(os.environ['SEED']), 'stream': False,
            })
            log.write(json.dumps({'dataset_row': index, 'prompt_key': key,
                                 'prompt_tokens': len(prompt), 'usage': result.get('usage')}) + '\n')
            log.flush()
            seen.add(key)
            print(f'Collected prompt {len(seen)}/{wanted}', flush=True)
    if len(seen) < wanted:
        raise RuntimeError('Not enough eligible distinct prompts')
    DONE.touch()


class Snapshot:
    def __init__(self):
        self.identity = None
        self.shape = None
        self.paths = set()
        self.train, self.valid = [], []

    def refresh(self):
        if FAILED.exists():
            raise RuntimeError(FAILED.read_text())
        for manifest in sorted(TRACE.rglob('manifest.json')):
            identity = json.loads(manifest.read_text())
            if self.identity is None:
                self.identity = identity
            if identity != self.identity:
                raise RuntimeError('Mixed checkpoint/schema identities')
            for path in sorted(manifest.parent.glob('*.pt')):
                if str(path) in self.paths:
                    continue
                row = torch.load(path, map_location='cpu', weights_only=True)
                features = row['features']
                shape = tuple(features.shape)
                if self.shape is None:
                    self.shape = shape
                if (shape != self.shape or len(shape) != 2 or
                    shape[0] != identity['proposals'] or shape[1] <= 5 or
                    not torch.isfinite(features.float()).all()):
                    raise RuntimeError(f'Invalid features: {path}')
                label = int(row['accepted_length'])
                if not 0 <= label <= identity['proposals']:
                    raise RuntimeError(f'Invalid accepted length: {path}')
                valid = int(hashlib.sha256(row['prompt_key'].encode()).hexdigest()[:8], 16) % 10 == 0
                (self.valid if valid else self.train).append(str(path))
                self.paths.add(str(path))
        return bool(self.train and self.valid)

    def save(self):
        payload = dict(identity=self.identity, shape=self.shape,
                       train=self.train, validation=self.valid)
        tmp = OUT / 'online_split.tmp'
        tmp.write_text(json.dumps(payload))
        tmp.replace(OUT / 'online_split.json')


def train():
    import torch_npu
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    from torch.utils.data import DataLoader, DistributedSampler
    from torch.utils.tensorboard import SummaryWriter

    rank, local = int(os.environ['RANK']), int(os.environ['LOCAL_RANK'])
    torch.npu.set_device(local)
    device = torch.device(f'npu:{local}')
    dist.init_process_group('hccl', timeout=timedelta(hours=2))
    torch.manual_seed(int(os.environ['SEED']))
    snapshot = Snapshot() if rank == 0 else None
    # Rank 0 publishes identical immutable file lists for all ranks.
    if rank == 0:
        while not snapshot.refresh() or len(snapshot.train) < 384:
            if DONE.exists():
                raise RuntimeError('Collection finished without enough train/validation records')
            print('Waiting for 384 training records and a validation prompt...', flush=True)
            time.sleep(5)
        snapshot.save()
    dist.barrier()
    info = json.loads((OUT / 'online_split.json').read_text())
    architecture = dict(hidden_size=info['shape'][1] - 5,
                        num_proposals=info['shape'][0], projection_size=32, mlp_hidden_size=64)
    head = base.load_module('acceptance_head').AcceptedLengthHead(**architecture).to(device)
    loss_fn = base.load_module('acceptance_loss').acceptance_boundary_loss
    model = DistributedDataParallel(head, device_ids=[local], broadcast_buffers=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(os.environ['LR']), weight_decay=0.01)
    writer = SummaryWriter(str(OUT / 'tensorboard_online')) if rank == 0 else None
    ckpt = OUT / 'checkpoints_online'
    if rank == 0:
        ckpt.mkdir(exist_ok=False)
        print(head, flush=True)
        print('Training ONLY', sum(p.numel() for p in head.parameters()), 'MLP parameters', flush=True)
    best, step, epoch, final_epochs = (float('inf'), float('inf')), 0, 0, 0
    batch_size = int(os.environ['BATCH_SIZE'])
    while True:
        if rank == 0:
            snapshot.refresh()
            snapshot.save()
        dist.barrier()
        info = json.loads((OUT / 'online_split.json').read_text())
        status = torch.tensor([int(DONE.exists()) if rank == 0 else 0], device=device)
        dist.broadcast(status, src=0)
        final_phase = bool(status.item())
        epoch += 1
        dataset = base.Records(info['train'])
        sampler = DistributedSampler(dataset, shuffle=True, seed=int(os.environ['SEED']))
        sampler.set_epoch(epoch)
        loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=0)
        model.train()
        totals = torch.zeros(3, device=device)
        for features, labels in loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            loss, _ = loss_fn(logits, labels, under_weight=3., over_weight=1., cost_weight=1.)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            batch = torch.stack((loss.detach() * labels.numel(),
                                 (logits.detach().argmax(-1) == labels).sum().float(),
                                 loss.detach().new_tensor(labels.numel())))
            totals += batch
            dist.all_reduce(batch)
            step += 1
            if writer:
                writer.add_scalar('train/step_loss', (batch[0] / batch[2]).item(), step)
                writer.add_scalar('train/step_accuracy', (batch[1] / batch[2]).item(), step)
        dist.all_reduce(totals)
        if rank == 0:
            head.eval()
            sums, count = {}, 0
            validation = DataLoader(base.Records(info['validation']), batch_size=batch_size)
            with torch.no_grad():
                for features, labels in validation:
                    labels = labels.to(device)
                    _, metrics = loss_fn(head(features.to(device)), labels,
                                         under_weight=3., over_weight=1., cost_weight=1.)
                    n = labels.numel()
                    count += n
                    for key, value in metrics.items():
                        sums[key] = sums.get(key, 0.) + float(value.cpu()) * n
            metrics = {k: v / count for k, v in sums.items()}
            writer.add_scalar('train/epoch_loss', (totals[0] / totals[2]).item(), epoch)
            writer.add_scalar('train/epoch_accuracy', (totals[1] / totals[2]).item(), epoch)
            writer.add_scalar('data/train_records', len(info['train']), epoch)
            writer.add_scalar('data/validation_records', len(info['validation']), epoch)
            for key, value in metrics.items():
                writer.add_scalar('validation/' + key, value, epoch)
            writer.flush()
            report = dict(epoch=epoch, final_phase=final_phase,
                          train_records=len(info['train']), validation_records=len(info['validation']),
                          train_loss=(totals[0] / totals[2]).item(), **metrics)
            print(json.dumps(report), flush=True)
            with (OUT / 'metrics_online.jsonl').open('a') as log:
                log.write(json.dumps(report) + '\n')
            payload = dict(architecture=architecture, identity=info['identity'],
                           state_dict={k: v.detach().cpu() for k,v in head.state_dict().items()},
                           epoch=epoch, validation_metrics=metrics,
                           loss_settings=dict(under_weight=3., over_weight=1., cost_weight=1.))
            def save(name):
                tmp = ckpt / (name + '.tmp')
                torch.save(payload, tmp)
                tmp.replace(ckpt / name)
            save('last.pt')
            # Compare best checkpoints only after validation stops growing.
            if final_phase:
                score = (metrics['decision_cost'], metrics['cross_entropy'])
                if score < best:
                    best = score
                    save('best.pt')
        dist.barrier()
        if final_phase:
            final_epochs += 1
            if final_epochs >= int(os.environ['EPOCHS']):
                break
    if writer:
        writer.close()
    dist.destroy_process_group()


if __name__ == '__main__':
    import sys
    if sys.argv[1] == 'collect':
        try:
            collect()
        except BaseException as error:
            FAILED.write_text(repr(error))
            raise
    else:
        train()
