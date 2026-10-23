#!/usr/bin/env python3
"""Train only AcceptedLengthHead using frozen DFlash training blocks.

Uses the existing ArrowDataset hidden-state transport and DFlash backbone.
No speculative serving loop, trace hook, or per-block disk cache is involved.
"""
import argparse
import json
import os
import random
import threading
from datetime import timedelta
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from acceptance_block_labels import cached_boundary, verified_boundary, eligible_starts


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--target', required=True)
    p.add_argument('--draft', required=True)
    p.add_argument('--dataset', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--endpoint', required=True)
    p.add_argument('--seq-len', type=int, default=8192)
    p.add_argument('--anchors', type=int, default=256)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--prefetch', type=int, default=2)
    p.add_argument('--verify-workers', type=int, default=4)
    p.add_argument('--selector-chunk', type=int, default=8)
    p.add_argument('--head-batch', type=int, default=384)
    p.add_argument('--epochs', type=int, default=1)
    p.add_argument('--lr', type=float, default=0.001)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--save-every', type=int, default=100)
    p.add_argument('--log-every', type=int, default=10)
    p.add_argument('--max-samples', type=int, default=0)
    p.add_argument('--request-timeout', type=float, default=180)
    p.add_argument('--audit-blocks', type=int, default=4,
                   help='Independently reverify this many cached-label blocks at startup')
    p.add_argument('--resume-head', default='', help='Restore head + optimizer at an epoch boundary')
    args = p.parse_args()
    for key in ('seq_len','anchors','prefetch','verify_workers','selector_chunk',
                'head_batch','epochs','save_every','log_every'):
        if getattr(args, key) < 1:
            p.error(f'--{key.replace("_", "-")} must be positive')
    if min(args.workers, args.max_samples, args.audit_blocks) < 0:
        p.error('Workers, max-samples and audit-blocks must be nonnegative')
    if args.lr <= 0 or args.request_timeout <= 0:
        p.error('Learning rate and timeout must be positive')
    return args


class CheckedRows:
    """Validate actual sequence lengths before the existing packer can truncate."""
    def __init__(self, dataset):
        self.dataset = dataset
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, index):
        row = self.dataset[index]
        if not isinstance(row, dict):
            raise RuntimeError(f'Hidden-state fetch failed for row {index}: {row}')
        actual = row['input_ids'].numel()
        if actual != int(self.dataset.approx_lengths[index]):
            raise ValueError(f'Row {index}: seq_len metadata differs from actual token count')
        return row


# Imports stay in main so spawned dataloader workers do not initialize an NPU.
def main():
    import torch
    import torch_npu
    from torch.utils.data import DataLoader
    from torch.utils.tensorboard import SummaryWriter
    from transformers import AutoConfig
    from hs_connectors import FileTransfer
    from speculators.config import VerifierConfig
    from speculators.models.dflash_prefix.config import DFlashPrefixSpeculatorConfig
    from speculators.models.dflash_prefix.core import DFlashPrefixDraftModel
    from speculators.models.dflash_prefix.selector_v2 import unary_topk
    from speculators.models.dflash_prefix.acceptance_features import build_acceptance_features, FEATURE_SCHEMA
    from speculators.models.dflash_prefix.acceptance_head import AcceptedLengthHead
    from speculators.models.dflash_prefix.acceptance_loss import acceptance_boundary_loss
    from speculators.models.dflash_prefix.acceptance_collector import checkpoint_fingerprint
    from speculators.train.data import ArrowDataset, CollateFn
    from speculators.train.dataloader import _worker_init_fn, _limit_worker_threads
    from speculators.train.distributed_batch_sampler import MultipackDistributedBatchSamplerV2
    from speculators.utils.loading import list_checkpoint_keys

    import torch.distributed as dist
    args = parse_args()
    rank = int(os.environ.get('RANK', '0'))
    world = int(os.environ.get('WORLD_SIZE', '1'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    torch.npu.set_device(local_rank)
    device = torch.device(f'npu:{local_rank}')
    if world > 1:
        dist.init_process_group('hccl', timeout=timedelta(minutes=30))
    if args.head_batch % world:
        raise ValueError('--head-batch is GLOBAL and must be divisible by worker count')
    local_batch = args.head_batch // world

    def reduce_values(values, maximum=False):
        dtype = torch.int64 if all(isinstance(v, int) for v in values) else torch.float32
        value = torch.tensor(values, dtype=dtype, device=device)
        if world > 1:
            dist.all_reduce(value, op=dist.ReduceOp.MAX if maximum else dist.ReduceOp.SUM)
        return value.cpu().tolist()

    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out/'checkpoints').mkdir(exist_ok=True)
    if rank == 0: (out/'settings.json').write_text(json.dumps(dict(vars(args), world_size=world), indent=2))
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    _limit_worker_threads()
    target_cfg = AutoConfig.from_pretrained(args.target, local_files_only=True)
    target_cfg = getattr(target_cfg, 'text_config', target_cfg)
    config = DFlashPrefixSpeculatorConfig.from_pretrained(args.draft, local_files_only=True)
    if config.sample_from_anchor or config.draft_vocab_size != target_cfg.vocab_size:
        raise ValueError('Requires full-vocabulary DFlash-Prefix with anchor excluded')
    if config.prefix_selector_kind not in ('compiled_prefix_v3', 'local_prefix_v2'):
        raise ValueError('This adapter supports compiled_prefix_v3 and local_prefix_v2')
    if config.prefix_inference_gate_scale != 1 or getattr(config, 'prefix_disable_selector', False):
        raise ValueError('Requires enabled selector with inference gate scale 1')
    for key in ('hidden_size', 'vocab_size', 'rms_norm_eps'):
        if getattr(config.transformer_layer_config, key) != getattr(target_cfg, key):
            raise ValueError(f'Target/checkpoint mismatch: {key}')
    config.transformer_layer_config.sliding_window = None
    config.transformer_layer_config.use_sliding_window = False
    config.transformer_layer_config.layer_types = ['full_attention'] * config.transformer_layer_config.num_hidden_layers
    config.transformer_layer_config._attn_implementation = 'sdpa'
    print(f'Rank {rank}/{world}: full attention, device={device}, seq_len={args.seq_len}, anchors={args.anchors}', flush=True)
    config.speculators_config.verifier = VerifierConfig.from_pretrained(args.target, local_files_only=True)
    if any(i < 0 or i > target_cfg.num_hidden_layers for i in config.aux_hidden_state_layer_ids):
        raise ValueError('Invalid target hidden-state layer indices')

    # Explicit anchors; all transformer/attention operations remain inherited.
    class BlockDraft(DFlashPrefixDraftModel):
        def _build_attention_mask(self, loss_mask, max_anchors, document_ids, device):
            anchors = self.acceptance_anchors
            if len(anchors) != max_anchors:
                raise ValueError('Anchor-count mismatch')
            if self.uses_sliding_window_attn:
                raise ValueError('Full-attention checkpoint required; no attention policy is changed')
            full = self._create_attention_mask(document_ids, loss_mask.shape[1], anchors, device)
            return full, None, anchors, torch.ones_like(anchors, dtype=torch.bool)

    print('Loading frozen draft checkpoint and matching verifier-owned weights...', flush=True)
    draft = BlockDraft.from_pretrained(args.draft, config=config, verifier=args.target,
                                      local_files_only=True, dtype='auto')
    allowed_missing = {'embed_tokens.weight', 'lm_head.weight', 'verifier_lm_head.weight',
                       'verifier_norm.weight', 't2d', 'd2t'}
    stored = set(list_checkpoint_keys(Path(args.draft)))
    expected = set(draft.state_dict())
    if expected-stored-allowed_missing or stored-expected:
        raise ValueError(f'Incomplete checkpoint: missing={sorted(expected-stored-allowed_missing)[:8]}, '
                         f'unexpected={sorted(stored-expected)[:8]}')
    draft.eval().requires_grad_(False)
    draft.to(device)
    if draft.uses_sliding_window_attn:
        raise ValueError('This run requires the existing full-attention checkpoint')
    if any(p.requires_grad for p in draft.parameters()):
        raise RuntimeError('Frozen-draft assertion failed')
    block = draft.block_size
    k = block-1
    hidden_size = config.transformer_layer_config.hidden_size
    architecture = dict(hidden_size=hidden_size, num_proposals=k, projection_size=32, mlp_hidden_size=64)
    head = AcceptedLengthHead(**architecture).float().to(device).train()
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=0.01)
    if {id(p) for p in draft.parameters()} & {id(p) for g in optimizer.param_groups for p in g['params']}:
        raise RuntimeError('Optimizer includes frozen-draft parameters')
    eos = getattr(target_cfg, 'eos_token_id', None)
    eos = [] if eos is None else [eos] if isinstance(eos, int) else list(eos)
    print('Fingerprinting checkpoints for compatibility metadata...', flush=True)
    identity = None
    if rank == 0:
        identity = dict(schema=FEATURE_SCHEMA, proposals=k,
                        draft_sha256=checkpoint_fingerprint(args.draft),
                        target_sha256=checkpoint_fingerprint(args.target),
                        decoding='greedy_default_penalties_v1', eos_token_ids=sorted(eos),
                        draft_attention='full_attention')
    if world > 1:
        objects = [identity]
        dist.broadcast_object_list(objects, src=0, device=device)
        identity = objects[0]
    if rank == 0: (out/'identity.json').write_text(json.dumps(identity, indent=2))
    step = packs = trained = visited_rows = 0
    start_epoch = 0
    if args.resume_head:
        saved = torch.load(args.resume_head, map_location='cpu', weights_only=True)
        if saved['identity'] != identity or saved['architecture'] != architecture:
            raise ValueError('Resume head/checkpoint identity mismatch')
        if not saved.get('epoch_complete', False) or 'optimizer' not in saved:
            raise ValueError('Resume only from an epoch-complete checkpoint with optimizer state')
        head.load_state_dict(saved['state_dict'])
        optimizer.load_state_dict(saved['optimizer'])
        step, trained = saved['step'], saved['training_blocks']
        start_epoch = saved['epoch']+1
    if world > 1:
        for param in head.parameters(): dist.broadcast(param.data, src=0)
    print(f'Frozen draft parameters: {sum(p.numel() for p in draft.parameters()):,}; '
          f'only head trainable: {sum(p.numel() for p in head.parameters()):,}', flush=True)

    dataset = ArrowDataset(max_len=args.seq_len, datapath=args.dataset,
                           transfer=FileTransfer(out/f'hidden_state_cache_rank{rank}'),
                           vllm_endpoint=args.endpoint, on_missing='generate', on_generate='delete',
                           train_ratio=1.0, split='train', hidden_states_dtype=torch.bfloat16,
                           model=args.target, request_timeout=args.request_timeout, max_retries=2,
                           generation_validation_retries=0)
    dataset.data = dataset.data.with_format('torch')
    if args.max_samples:
        n = min(args.max_samples, len(dataset))
        dataset.data = dataset.data.select(range(n))
        dataset.approx_lengths = dataset.approx_lengths[:n]
    global_rows = len(dataset)
    # Shard rows first, then pack each shard. No padding, duplicates, or dropped tails.
    if global_rows < world:
        raise ValueError('Need at least one dataset row per training worker')
    shard_indices = range(rank, global_rows, world)
    dataset.data = dataset.data.select(shard_indices)
    dataset.approx_lengths = dataset.approx_lengths[rank::world]
    if any(length > args.seq_len or length < 1 for length in dataset.approx_lengths):
        raise ValueError('Dataset row exceeds token budget or is empty; refusing truncation')
    sampler = MultipackDistributedBatchSamplerV2(args.seq_len, dataset.approx_lengths,
                                                num_replicas=1, rank=0, seed=args.seed,
                                                truncate_long_samples=False)
    loader = DataLoader(CheckedRows(dataset), batch_sampler=sampler,
                        collate_fn=CollateFn(args.seq_len, hidden_size,
                                            num_target_layers=len(draft.target_layer_ids),
                                            dtype=torch.bfloat16),
                        num_workers=args.workers, pin_memory=False,
                        prefetch_factor=args.prefetch if args.workers else None,
                        persistent_workers=bool(args.workers),
                        multiprocessing_context='spawn' if args.workers else None,
                        worker_init_fn=_worker_init_fn if args.workers else None)

    # Fallback verifier shares the exact hidden-state transport used by training.
    # HTTP/CPU work is parallel; all NPU projection work stays on the main thread.
    local = threading.local()
    def fetch_verification(item):
        import openai
        from speculators.data_generation.vllm_client import generate_hidden_states
        from speculators.data_generation.offline import check_hidden_states
        index, tokens = item
        if not hasattr(local, 'client'):
            local.client = openai.OpenAI(base_url=args.endpoint, api_key='EMPTY', max_retries=0)
            local.transfer = FileTransfer(out/f'verification_cache_rank{rank}')
            local.transfer.setup()
        handle = None
        try:
            handle = generate_hidden_states(local.client, args.target, {'input_ids': tokens},
                                            timeout=args.request_timeout, max_retries=2)
            value = local.transfer.get_generated(handle)
            if value is None:
                raise RuntimeError('No hidden states returned by fallback verification')
            check_hidden_states(value, tokens)
            # Prediction for proposal j uses the state BEFORE proposal j.
            last = value['hidden_states'][-k-1:-1, -1].clone()
            if last.shape != (k, hidden_size):
                raise ValueError('Unexpected verification hidden-state shape')
            return index, last
        finally:
            if handle is not None:
                local.transfer.delete(handle)

    @torch.no_grad()
    def verify_items(items):
        result = {}
        # At most args.anchors small responses retained; never retain a dataset.
        for index, states in pool.map(fetch_verification, items):
            with torch.autocast(device_type='npu', dtype=torch.bfloat16):
                logits = draft.verifier_lm_head(draft.verifier_norm(states.to(device)))
            result[index] = logits.argmax(-1).cpu().tolist()
        return result

    class NullWriter:
        def add_scalar(self, *a, **kw): pass
        def flush(self): pass
        def close(self): pass
    writer = SummaryWriter(str(out/'tensorboard')) if rank == 0 else NullWriter()
    metrics = {}
    feature_queue, label_queue = [], []
    queued = 0
    audit_remaining = args.audit_blocks
    run_start = time.monotonic()
    pending_time = dict(fetch=0., draft=0., verification=0., head=0.)
    total_fallback = total_blocks = eos_skipped = no_blocks = 0

    def sync():
        torch.npu.synchronize()

    def save(name, epoch, epoch_complete=False):
        if rank != 0: return
        value = dict(architecture=architecture, identity=identity,
                     state_dict={n:v.detach().cpu() for n,v in head.state_dict().items()},
                     optimizer=optimizer.state_dict(), step=step, training_blocks=trained,
                     epoch=epoch, epoch_complete=epoch_complete, metrics=metrics,
                     loss_settings=dict(under_weight=3.,over_weight=1.,cost_weight=1.),
                     training_mode='frozen_dataset_blocks_with_exact_divergence_verification',
                     settings=vars(args))
        temp = out/'checkpoints'/(name+'.tmp')
        torch.save(value, temp)
        temp.replace(out/'checkpoints'/name)

    def train_queued(final=False):
        nonlocal queued, feature_queue, label_queue, step, trained, metrics
        local_steps = (queued + local_batch - 1)//local_batch if final else queued//local_batch
        rounds = int(reduce_values([local_steps], maximum=True)[0])
        if not rounds:
            return
        x = torch.cat(feature_queue) if queued else None
        y = torch.cat(label_queue) if queued else None
        consumed = 0
        names = ('loss', 'cross_entropy', 'expected_cost_normalized', 'exact_accuracy',
                 'mean_under_tokens', 'mean_over_tokens', 'decision_cost')
        for batch_round in range(rounds):
            count = min(local_batch, queued-consumed) if batch_round < local_steps else 0
            global_count = int(reduce_values([count])[0])
            if global_count == 0: continue
            t = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            stats = {}
            if count:
                logits = head(x[consumed:consumed+count].detach())
                loss, stats = acceptance_boundary_loss(
                    logits, y[consumed:consumed+count], under_weight=3.,
                    over_weight=1., cost_weight=1., validate=True)
                # Summed gradients divided by total participating examples.
                # An exhausted/empty rank contributes zero, but still participates.
                (loss * (count/global_count)).backward()
            flat = torch.cat([
                (p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1)
                for p in head.parameters()])
            if world > 1: dist.all_reduce(flat)
            offset = 0
            for param in head.parameters():
                param.grad = flat[offset:offset+param.numel()].view_as(param)
                offset += param.numel()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            sync()
            pending_time['head'] += time.monotonic()-t
            if any(p.grad is not None or p.requires_grad for p in draft.parameters()):
                raise RuntimeError('Frozen draft unexpectedly received gradients')
            consumed += count
            step += 1
            trained += global_count
            vals = reduce_values([float(stats[n].detach().cpu())*count if count else 0. for n in names])
            metrics = dict(zip(names, [v/global_count for v in vals]))
            if step == 1 or step % args.log_every == 0:
                counters = reduce_values([packs, visited_rows, total_fallback, total_blocks, eos_skipped, no_blocks])
                timings = reduce_values(list(pending_time.values()), maximum=True)
                elapsed = max(reduce_values([time.monotonic()-run_start], maximum=True)[0], 1e-6)
                if rank == 0:
                    report = dict(step=step, packs=int(counters[0]), dataset_rows=int(counters[1]),
                                  training_blocks=trained, blocks_per_second=trained/elapsed,
                                  fallback_blocks=int(counters[2]), candidate_blocks=int(counters[3]),
                                  fallback_fraction=counters[2]/max(counters[3],1),
                                  eos_skipped=int(counters[4]), packs_without_blocks=int(counters[5]),
                                  global_batch=global_count, world_size=world,
                                  seconds_since_last_report_max_rank=dict(zip(pending_time,timings)), **metrics)
                    print(json.dumps(report), flush=True)
                    with (out/'metrics.jsonl').open('a') as f: f.write(json.dumps(report)+'\n')
                    for name,value in metrics.items(): writer.add_scalar('train/'+name,value,step)
                    writer.add_scalar('progress/dataset_rows',counters[1],step)
                    writer.add_scalar('progress/training_blocks',trained,step)
                    writer.flush()
                for name in pending_time: pending_time[name]=0.
            if step % args.save_every == 0: save('last.pt', epoch)
        feature_queue = [x[consumed:].detach().clone()] if consumed < queued else []
        label_queue = [y[consumed:].detach().clone()] if consumed < queued else []
        queued -= consumed

    print(f'Training {len(dataset):,} dataset rows per epoch; {args.anchors} block positions per pack; '
          f'global head batch={args.head_batch}; rank={rank}; no full-response generation.', flush=True)
    try:
        with ThreadPoolExecutor(max_workers=args.verify_workers) as pool:
            for epoch in range(start_epoch, args.epochs):
                sampler.set_epoch(epoch)
                rng = random.Random(args.seed+epoch*world+rank)
                iterator = iter(loader)
                rows_this_epoch = 0
                while True:
                    begin = time.monotonic()
                    try: batch = next(iterator)
                    except StopIteration: batch = None
                    active = int(reduce_values([int(batch is not None)])[0])
                    if active == 0: break
                    if batch is None:
                        train_queued()
                        continue
                    pending_time['fetch'] += time.monotonic()-begin
                    if batch.get('error_records',0):
                        raise RuntimeError('Hidden-state loader dropped a row; refusing silent incomplete training')
                    docs_cpu = batch['document_ids'][0].tolist()
                    mask_cpu = batch['loss_mask'][0].tolist()
                    ids_cpu = batch['input_ids'][0].tolist()
                    n_rows = len(set(d for d in docs_cpu if d >= 0))
                    visited_rows += n_rows
                    rows_this_epoch += n_rows
                    packs += 1
                    starts = eligible_starts(mask_cpu, docs_cpu, block)
                    if not starts:
                        no_blocks += 1
                        print(f'Pack {packs}: no complete assistant blocks; rows visited={visited_rows}', flush=True)
                        train_queued()
                        continue
                    anchors_cpu = sorted(rng.sample(starts,min(args.anchors,len(starts))))
                    tensors = {name:batch[name].to(device) for name in
                               ('hidden_states','input_ids','loss_mask','verifier_last_hidden_states',
                                'document_ids','position_ids')}
                    draft.acceptance_anchors = torch.tensor(anchors_cpu,device=device,dtype=torch.long)
                    begin = time.monotonic()
                    with torch.no_grad(), torch.autocast(device_type='npu',dtype=torch.bfloat16):
                        h,u,target,aligned,indices = draft._backbone_forward(**tensors,max_anchors=len(anchors_cpu))
                        n = len(anchors_cpu)
                        hidden = h.reshape(n,block,-1)[:,1:]
                        unary = u.reshape(n,block,-1)[:,1:]
                        cached_ids = target.reshape(n,block,-1)[:,1:].argmax(-1).cpu().tolist()
                        block_ids = tensors['input_ids'][0,indices].reshape(n,block)
                        reference = block_ids[:,1:].cpu().tolist()
                        candidate_logits,candidates = unary_topk(unary,config.prefix_top_k)
                        del target,u,unary
                        pieces, choices = [],[]
                        for start in range(0,n,args.selector_chunk):
                            sl=slice(start,start+args.selector_chunk)
                            tables = draft.prefix_head.prepare_tables(hidden[sl],candidates[sl],candidate_logits[sl],
                                                                     block_ids[sl,0],**draft._prefix_kwargs())
                            selected,scores = draft.prefix_head.walk(tables,return_scores=True)
                            pieces.append(build_acceptance_features(hidden[sl],candidates[sl],candidate_logits[sl],
                                                                    selected,scores,validate=True))
                            choices.extend(selected.cpu().tolist())
                        features=torch.cat(pieces).detach()
                    del h, hidden, candidates, candidate_logits, tables, scores, selected, pieces, block_ids, tensors
                    sync()
                    pending_time['draft'] += time.monotonic()-begin
                    total_blocks += n
                    labels, statuses = [],[]
                    for chosen,ref,tgt in zip(choices,reference,cached_ids):
                        length,status=cached_boundary(chosen,ref,tgt,eos)
                        labels.append(length); statuses.append(status)
                    fallback=[i for i,s in enumerate(statuses) if s=='verify']
                    total_fallback += len(fallback)
                    audit=[i for i,s in enumerate(statuses) if s=='exact'][:audit_remaining]
                    items=[]
                    for i in fallback+audit:
                        a=anchors_cpu[i]
                        document=docs_cpu[a]
                        first=a
                        while first>0 and docs_cpu[first-1]==document: first-=1
                        tokens=ids_cpu[first:a+1]+choices[i]
                        items.append((i,tokens))
                    begin=time.monotonic()
                    verified=verify_items(items) if items else {}
                    for i in fallback:
                        labels[i],statuses[i]=verified_boundary(choices[i],verified[i],eos)
                    for i in audit:
                        actual,status=verified_boundary(choices[i],verified[i],eos)
                        if (actual,status)!=(labels[i],statuses[i]):
                            raise RuntimeError(f'Cached-label audit failed at pack {packs}, block {i}: '
                                               f'cached={(labels[i],statuses[i])}, verified={(actual,status)}. '
                                               'Investigate target numeric/transport alignment; no approximate labels used.')
                    audit_remaining -= len(audit)
                    sync()
                    pending_time['verification'] += time.monotonic()-begin
                    keep=[i for i,s in enumerate(statuses) if s=='exact']
                    eos_skipped += sum(s=='eos' for s in statuses)
                    if keep:
                        feature_queue.append(features[keep].detach())
                        label_queue.append(torch.tensor([labels[i] for i in keep],dtype=torch.long,device=device))
                        queued += len(keep)
                    train_queued()
                    if packs==1 or packs%args.log_every==0:
                        print(f'rank={rank} epoch={epoch+1} rows={rows_this_epoch}/{len(dataset)} '
                              f'pack={packs} blocks={n} fallback={len(fallback)} '
                              f'head_updates={step} queued={queued}',flush=True)
                if rows_this_epoch!=len(dataset):
                    raise RuntimeError(f'Expected every row once: saw {rows_this_epoch}, expected {len(dataset)}')
                train_queued(final=True)
                if not trained: raise RuntimeError('No trainable full-block examples found')
                save(f'epoch_{epoch+1}.pt',epoch,True)
                save('last.pt',epoch,True)
            if start_epoch>=args.epochs:
                raise ValueError('Resume epoch already reaches --epochs; increase total epochs')
            save('final.pt',args.epochs-1,True)
            print(f'Finished: {out}/checkpoints/final.pt; {trained:,} blocks, {step:,} head updates',flush=True)
    finally:
        writer.close()
        if world > 1: dist.destroy_process_group()


if __name__=='__main__':
    main()
