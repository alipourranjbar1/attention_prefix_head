"""One-batch target lookahead for the pinned packed DLoop training loop.

Only CPU tensors and HTTP/file transfer run in the worker thread. RoPE prediction
runs on the trainer thread under a saved/restored RNG state. It skips the one
known packed-anchor random draw, then predicts the next native offset draw.
The next real offsets are always checked before using prefetched features; a
prediction mismatch falls back to an ordinary request, never different labels.
No future RNG draw is committed and no prefetched batch is stored in checkpoints.
"""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
import json
import os
import time

import torch


def predict_offsets(current, following, device, offset_max):
    from ascend_support.checkpoints import capture_rng, restore_rng
    state = capture_rng()
    try:
        # _sample_packed_anchors consumes exactly this one global RNG draw.
        # Loop depth uses its own generator; the configured attention dropout is 0.
        torch.rand(current['loss_mask'].shape, device=device)
        if offset_max <= 0:
            return None
        hi = (offset_max - following['lengths'].to(device) + 1).clamp(min=1)
        return (torch.rand(following['input_ids'].shape[0], device=device) * hi).long().cpu()
    finally:
        restore_rng(state)


def equal_offsets(a, b):
    return (a is None and b is None) or (
        a is not None and b is not None and torch.equal(a, b.detach().cpu()))


class PrefetchBatches:
    """One next batch, at most one submitted/ready feature result per trainer."""
    def __init__(self, loader, target, device, offset_max, stop_batches,
                 output, rank, log_every=10):
        self.loader, self.target, self.device = loader, target, device
        self.offset_max, self.stop_batches = offset_max, stop_batches
        self.output, self.rank, self.log_every = Path(output), rank, max(1, log_every)
        self.iterator = None
        self.cursor = 0
        self.buffer = None
        self.pending = None
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='dloop-target')
        self.closed = False
        self.batch_wait_seconds = 0.
        self.captures = 0

    def __len__(self):
        # tqdm and the LR scheduler retain the original full-epoch budget.
        return len(self.loader)

    def __iter__(self):
        if self.iterator is None:
            self.iterator = iter(self.loader)
        return self

    def __next__(self):
        if self.cursor >= self.stop_batches:
            raise StopIteration
        if self.iterator is None:
            iter(self)
        started = time.perf_counter()
        if self.buffer is not None:
            data, self.buffer = self.buffer, None
        else:
            data = next(self.iterator)
        self.cursor += 1
        self.batch_wait_seconds = time.perf_counter() - started
        return data

    def peek(self):
        if self.cursor >= self.stop_batches:
            return None
        if self.buffer is None:
            self.buffer = next(self.iterator, None)
        return self.buffer

    def capture(self, data, input_ids, attention_mask, loss_mask, position_offsets, step):
        started = time.perf_counter()
        hit, mismatch = False, False
        if self.pending is not None:
            expected_data, expected_offsets, future = self.pending
            self.pending = None
            # The original tensors are yielded from our one-slot lookahead buffer.
            matches = expected_data is data and equal_offsets(expected_offsets, position_offsets)
            result = future.result()  # Also propagates worker exceptions promptly.
            if matches:
                hit = True
            else:
                mismatch = True
                del result
        if not hit:
            result = self.target.generate_dflash_data(
                input_ids, attention_mask, loss_mask, position_offsets=position_offsets)
        wait_seconds = time.perf_counter() - started
        # Preserve the original target-output interface for this consumed batch.
        result.input_ids, result.attention_mask, result.loss_mask = input_ids, attention_mask, loss_mask
        submit_started = time.perf_counter()
        following = self.peek()
        if following is not None:
            next_offsets = predict_offsets(data, following, self.device, self.offset_max)
            if any(value.device.type != 'cpu' for value in following.values() if torch.is_tensor(value)):
                raise ValueError('Target prefetch workers require CPU batches')
            future = self.pool.submit(self.target.generate_dflash_data,
                following['input_ids'], following['attention_mask'], following['loss_mask'],
                position_offsets=next_offsets)
            self.pending = following, next_offsets, future
        submit_seconds = time.perf_counter() - submit_started
        self.captures += 1
        if self.captures <= 3 or step % self.log_every == 0 or mismatch:
            row = dict(event='target_prefetch', step=step, rank=self.rank,
                prefetch_hit=hit, prediction_mismatch=mismatch,
                target_wait_seconds=wait_seconds,
                token_batch_wait_seconds=self.batch_wait_seconds,
                prefetch_submit_seconds=submit_seconds,
                next_batch_submitted=following is not None,
                **getattr(result, '_pair_timing', {}))
            self.output.mkdir(parents=True, exist_ok=True)
            with (self.output / f'prefetch_rank{self.rank}.jsonl').open('a') as stream:
                stream.write(json.dumps(row) + '\n')
            if self.rank == 0:
                print(json.dumps(row), flush=True)
        return result

    def close(self):
        if not self.closed:
            self.closed = True
            # Finish any in-flight transfer so its transient file is cleaned up.
            self.pool.shutdown(wait=True, cancel_futures=True)
            self.pending = self.buffer = None
            if self.iterator is not None and hasattr(self.iterator, '_shutdown_workers'):
                self.iterator._shutdown_workers()


@contextmanager
def target_prefetch(args, target, loader, device, epoch, draft_config, enabled=None):
    if enabled is None:
        enabled = os.environ.get('DLOOP_PREFETCH', '1') != '0'
    if not enabled or getattr(target, 'backend', '') != 'hf_remote':
        yield None
        return
    if (not args.packed_token_budget or args.accumulation_steps != 1
            or getattr(args, 'is_dspark', False) or args.loss_type != 'dflash'
            or draft_config.attention_dropout != 0.):
        raise ValueError('Prefetch requires packed regular DLoop, accumulation=1, and zero dropout')
    if device.type == 'npu':
        # The checkpoint helpers use the current device; never forecast another rank's RNG.
        if device.index is not None and torch.npu.current_device() != device.index:
            raise ValueError('Current NPU differs from this training rank')
    limit = len(loader)
    if args.max_steps is not None:
        limit = min(limit, max(0, args.max_steps - epoch * len(loader)))
    pipeline = PrefetchBatches(loader, target, device, args.position_offset_max, limit,
                              args.output_dir, int(os.environ.get('RANK', '0')), args.log_interval)
    print(f'DLoop: one-batch target prefetch enabled; epoch {epoch + 1}, stop batch {limit}', flush=True)
    try:
        yield pipeline
    finally:
        pipeline.close()
