"""Concurrent rollout collection; frozen features; single-device acceptance-head training.
Requires the existing server-side trace writer and acceptance_head/acceptance_loss.
"""
import argparse
import ctypes
import importlib.util
import json
import os
import select
import struct
import time
import urllib.request
from collections import deque
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path

ROOT = Path(os.environ['SPEC_MAIN'])
OUT = Path(os.environ['EXPERIMENT'])
TRACES = OUT / 'traces'
DONE = OUT / 'collection.done'
FAILED = OUT / 'collection.failed'


def atomic_json(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value))
    tmp.replace(path)


def module(name):
    path = ROOT / 'speculators/src/speculators/models/dflash_prefix' / (name + '.py')
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def request(url, payload=None):
    # One opener per request: no shared mutable HTTP state across workers.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    req = urllib.request.Request(url, data=None if payload is None else json.dumps(payload).encode(),
                                 headers={'Content-Type': 'application/json'})
    with opener.open(req, timeout=float(os.environ.get('REQUEST_TIMEOUT', '1800'))) as response:
        return json.load(response)


def collect():
    from datasets import load_from_disk, DatasetDict
    from tqdm.auto import tqdm
    dataset = load_from_disk(os.environ['DATASET'])
    if isinstance(dataset, DatasetDict):
        raise ValueError('Expected one Dataset, not DatasetDict')
    dataset = dataset.with_format(None)
    concurrency = int(os.environ.get('COLLECT_CONCURRENCY', '8'))
    if concurrency < 1:
        raise ValueError('COLLECT_CONCURRENCY must be positive')
    limit = int(os.environ.get('MAX_SAMPLES', '0'))
    total = min(len(dataset), limit) if limit > 0 else len(dataset)
    endpoint = 'http://127.0.0.1:' + os.environ['PORT']
    name = os.environ['SERVED_MODEL_NAME']
    if name not in [x['id'] for x in request(endpoint + '/v1/models')['data']]:
        raise RuntimeError('Wrong model on collection port')
    completed = skipped = visited = generated = submitted = 0
    started = time.monotonic()

    def run_one(index, row):
        ids, mask = row['input_ids'], row['loss_mask']
        seq_len = int(row['seq_len'])
        if not 0 < seq_len <= len(ids) or len(ids) != len(mask):
            raise ValueError(f'Invalid token/mask lengths at row {index}')
        boundary = next((i for i, flag in enumerate(mask[:seq_len]) if flag), None)
        if boundary is None or boundary == 0:
            return {'row': index, 'skipped': 'no usable prompt boundary'}
        if seq_len > int(os.environ['MAX_MODEL_LEN']):
            raise ValueError(f'Row {index} exceeds MAX_MODEL_LEN; refusing truncation')
        begin = time.monotonic()
        result = request(endpoint + '/v1/completions', {
            'model': name, 'prompt': [int(x) for x in ids[:boundary]],
            'temperature': 0, 'max_tokens': seq_len - boundary,
            'seed': int(os.environ.get('SEED', '42')), 'stream': False,
        })
        if 'choices' not in result or 'error' in result:
            raise RuntimeError(f'Invalid completion response for row {index}: {result}')
        return {'row': index, 'prompt_tokens': boundary, 'max_tokens': seq_len-boundary,
                'usage': result.get('usage') or {}, 'request_seconds': time.monotonic()-begin}

    pool = ThreadPoolExecutor(max_workers=concurrency)
    pending = {}
    try:
        with (OUT / 'samples.jsonl').open('w') as log, tqdm(total=total, desc='Completed dataset rows', unit='row') as bar:
            while submitted < total or pending:
                # Bounded queue: never construct millions of futures or load all rows.
                while submitted < total and len(pending) < concurrency:
                    pending[pool.submit(run_one, submitted, dataset[submitted])] = submitted
                    submitted += 1
                ready, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in ready:
                    index = pending.pop(future)
                    try:
                        record = future.result()
                    except Exception as exc:
                        raise RuntimeError(f'Collection failed at row {index}; no automatic retry (avoids duplicate traces)') from exc
                    visited += 1
                    if 'skipped' in record:
                        skipped += 1
                    else:
                        completed += 1
                        generated += int(record['usage'].get('completion_tokens', 0))
                    log.write(json.dumps(record) + '\n')
                    bar.update(1)
                log.flush()
                elapsed = max(time.monotonic()-started, 1e-6)
                progress = dict(dataset_rows=len(dataset), selected_rows=total, visited=visited,
                                submitted=submitted, completed=completed, skipped=skipped,
                                remaining=total-visited, in_flight=len(pending),
                                completion_tokens=generated, elapsed_seconds=elapsed,
                                rows_per_second=visited/elapsed, completion_tokens_per_second=generated/elapsed)
                atomic_json(OUT / 'progress.json', progress)
                bar.set_postfix(tokens_s=round(generated/elapsed, 1), active=len(pending))
        DONE.touch()
    except BaseException as exc:
        FAILED.write_text(repr(exc))
        for future in pending:
            future.cancel()
        raise
    finally:
        pool.shutdown(wait=True, cancel_futures=True)


class TraceQueue:
    """Use Linux close/move notifications to avoid rescanning every past record.
    Falls back to periodic scans if inotify is unavailable. One final scan drains
    records after DONE. The existing writer must publish complete .pt records.
    """
    MASK = 0x8 | 0x80 | 0x100  # CLOSE_WRITE, MOVED_TO, CREATE

    def __init__(self):
        self.pending = deque()
        self.seen = set()
        self.identity = None
        self.dirs = {}
        self.fd = -1
        self.last_scan = 0.
        TRACES.mkdir(parents=True, exist_ok=True)
        try:
            self.libc = ctypes.CDLL(None, use_errno=True)
            self.libc.inotify_init1.argtypes = [ctypes.c_int]
            self.libc.inotify_add_watch.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_uint32]
            self.fd = self.libc.inotify_init1(os.O_NONBLOCK | os.O_CLOEXEC)
        except (AttributeError, OSError):
            self.fd = -1
        self.scan()
        print(f'Trace discovery: {"inotify" if self.fd >= 0 else "periodic scan"}', flush=True)

    def watch(self, directory):
        if self.fd < 0 or directory in self.dirs.values():
            return
        wd = self.libc.inotify_add_watch(self.fd, os.fsencode(directory), self.MASK)
        if wd < 0:
            os.close(self.fd)
            self.fd = -1
            self.dirs.clear()
            print('inotify watch failed; using periodic scans.', flush=True)
        else:
            self.dirs[wd] = directory

    def add(self, path):
        key = str(path)
        if key not in self.seen:
            self.seen.add(key)
            self.pending.append(path)

    def scan(self, root=TRACES):
        for directory, _, files in os.walk(root):
            base = Path(directory)
            self.watch(base)
            # Rescan after installing watch to close the discovery race.
            with os.scandir(base) as entries:
                for entry in entries:
                    if entry.name.endswith('.pt') and entry.is_file():
                        self.add(Path(entry.path))
        self.last_scan = time.monotonic()

    def refresh(self, final=False):
        if FAILED.exists():
            raise RuntimeError(FAILED.read_text())
        if self.fd >= 0:
            while select.select([self.fd], [], [], 0)[0]:
                data = os.read(self.fd, 1024*1024)
                offset = 0
                while offset < len(data):
                    wd, mask, cookie, size = struct.unpack_from('iIII', data, offset)
                    name = os.fsdecode(data[offset+16:offset+16+size].split(b'\0', 1)[0])
                    offset += 16 + size
                    if mask & 0x4000:  # queue overflow
                        self.scan()
                        continue
                    base = self.dirs.get(wd)
                    if base is None:
                        continue
                    path = base / name
                    if mask & 0x40000000 and mask & (0x100 | 0x80):
                        self.scan(path)
                    elif path.suffix == '.pt' and mask & (0x8 | 0x80):
                        self.add(path)
        if final or (self.fd < 0 and time.monotonic()-self.last_scan >= 5):
            self.scan()

    def manifest(self, path):
        manifest = path.parent / 'manifest.json'
        if not manifest.exists():
            return False
        identity = json.loads(manifest.read_text())
        if self.identity is None:
            self.identity = identity
        elif identity != self.identity:
            raise RuntimeError('Mixed trace identities')
        return True

    def take(self, count):
        result = []
        # Allow other directories to progress if a manifest is not yet published.
        for _ in range(len(self.pending)):
            if len(result) >= count:
                break
            path = self.pending.popleft()
            if self.manifest(path):
                result.append(path)
            else:
                self.pending.append(path)
        return result

    def close(self):
        if self.fd >= 0:
            os.close(self.fd)


def train():
    import torch
    import torch_npu
    from torch.utils.tensorboard import SummaryWriter
    torch.npu.set_device(0)
    device = torch.device('npu:0')
    torch.manual_seed(int(os.environ.get('SEED', '42')))
    batch_size = int(os.environ.get('BATCH_SIZE', '384'))
    if batch_size < 1:
        raise ValueError('BATCH_SIZE must be positive')
    save_every = max(1, int(os.environ.get('SAVE_EVERY', '100')))
    log_every = max(1, int(os.environ.get('LOG_EVERY', '10')))
    loss_fn = module('acceptance_loss').acceptance_boundary_loss
    queue = TraceQueue()
    head = optimizer = writer = None
    step = trained = 0
    last_progress = time.monotonic()
    stall_timeout = float(os.environ.get('TRACE_STALL_TIMEOUT', '1800'))
    final_scanned = False
    metrics = {}

    def save(filename):
        payload = dict(architecture=architecture, identity=queue.identity,
                       state_dict={n: v.detach().cpu() for n, v in head.state_dict().items()},
                       step=step, training_blocks=trained, metrics=metrics,
                       loss_settings=dict(under_weight=3., over_weight=1., cost_weight=1.))
        tmp = OUT / 'checkpoints' / (filename + '.tmp')
        torch.save(payload, tmp)
        tmp.replace(OUT / 'checkpoints' / filename)

    try:
        while True:
            wait_start = time.monotonic()
            while True:
                finished = DONE.exists()
                queue.refresh(final=finished and not final_scanned)
                final_scanned = final_scanned or finished
                if finished or len(queue.pending) >= batch_size:
                    break
                if time.monotonic()-last_progress > stall_timeout:
                    raise RuntimeError('No training batch within TRACE_STALL_TIMEOUT. Check server.log and trace writer.')
                time.sleep(0.2)
            wait_seconds = time.monotonic()-wait_start
            paths = queue.take(batch_size)
            if not paths:
                if finished:
                    if queue.pending:
                        raise RuntimeError('Trace records exist without a usable manifest')
                    break
                if time.monotonic()-last_progress > stall_timeout:
                    raise RuntimeError('Traces are waiting for their manifest')
                time.sleep(0.2)
                continue
            load_start = time.monotonic()
            features, labels = [], []
            for path in paths:
                # A brief retry handles a writer publishing a .pt before closing it.
                for attempt in range(20):
                    try:
                        row = torch.load(path, map_location='cpu', weights_only=True)
                        break
                    except (EOFError, RuntimeError, OSError):
                        if attempt == 19:
                            raise
                        time.sleep(0.1)
                x, y = row['features'], int(row['accepted_length'])
                if x.ndim != 2:
                    raise ValueError(f'Expected one [K, H+5] feature record: {path}')
                if head is None:
                    k, dimension = x.shape
                    if k != queue.identity['proposals'] or dimension <= 5:
                        raise ValueError('Invalid feature schema')
                    architecture = dict(hidden_size=dimension-5, num_proposals=k,
                                        projection_size=32, mlp_hidden_size=64)
                    head = module('acceptance_head').AcceptedLengthHead(**architecture).to(device)
                    optimizer = torch.optim.AdamW(head.parameters(), lr=float(os.environ['LR']), weight_decay=0.01)
                    writer = SummaryWriter(str(OUT / 'tensorboard'))
                    (OUT / 'checkpoints').mkdir(exist_ok=True)
                    print(head, flush=True)
                    print(f'Only acceptance head optimized: {sum(p.numel() for p in head.parameters()):,} parameters', flush=True)
                if tuple(x.shape) != (k, dimension) or not torch.isfinite(x.float()).all() or not 0 <= y <= k:
                    raise ValueError(f'Invalid trace record: {path}')
                features.append(x.float())
                labels.append(y)
            load_seconds = time.monotonic()-load_start
            train_start = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            logits = head(torch.stack(features).to(device))
            target = torch.tensor(labels, dtype=torch.long, device=device)
            loss, _ = loss_fn(logits, target, under_weight=3., over_weight=1., cost_weight=1.)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            with torch.no_grad():
                error = logits.argmax(-1)-target
                values = torch.stack((loss.detach(), error.eq(0).float().mean(),
                                      (-error).clamp_min(0).float().mean(), error.clamp_min(0).float().mean())).cpu().tolist()
            train_seconds = time.monotonic()-train_start
            step += 1
            trained += len(paths)
            last_progress = time.monotonic()
            metrics = dict(loss=values[0], exact_accuracy=values[1], mean_under_tokens=values[2],
                           mean_over_tokens=values[3], decision_cost=3*values[2]+values[3])
            report = dict(step=step, unique_training_blocks=trained, batch_blocks=len(paths),
                          wait_seconds=wait_seconds, load_seconds=load_seconds, train_seconds=train_seconds,
                          queued_blocks=len(queue.pending), **metrics)
            for name, value in metrics.items():
                writer.add_scalar('train/'+name, value, step)
            for name in ('wait_seconds', 'load_seconds', 'train_seconds', 'queued_blocks'):
                writer.add_scalar('timing/'+name, report[name], step)
            writer.add_scalar('progress/training_blocks', trained, step)
            if (OUT/'progress.json').exists():
                progress = json.loads((OUT/'progress.json').read_text())
                for name in ('visited', 'remaining', 'skipped', 'rows_per_second', 'completion_tokens_per_second'):
                    writer.add_scalar('dataset/'+name, progress[name], step)
            with (OUT/'metrics.jsonl').open('a') as log:
                log.write(json.dumps(report)+'\n')
            if step % log_every == 0 or step == 1:
                print(json.dumps(report), flush=True)
                writer.flush()
            if step % save_every == 0:
                save('last.pt')
        if head is None:
            raise RuntimeError('No verification records collected; inspect the server trace writer')
        save('last.pt')
        save('final.pt')
        print(f'Finished: {trained} unique blocks, {step} updates', flush=True)
    finally:
        queue.close()
        if writer is not None:
            writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['collect', 'train'])
    args = parser.parse_args()
    try:
        collect() if args.mode == 'collect' else train()
    except BaseException as exc:
        if args.mode == 'collect':
            FAILED.write_text(repr(exc))
        raise
