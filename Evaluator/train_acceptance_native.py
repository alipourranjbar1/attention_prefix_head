"""One full dataset pass with the Speculators Trainer and a frozen drafter."""
import os
from pathlib import Path

import torch


def exact_boundaries(selected, target, reference, valid):
    """Only use cached verifier logits while their reference context is valid."""
    lengths = selected.eq(target).long().cumprod(-1).sum(-1)
    positions = torch.arange(selected.shape[1], device=selected.device)[None]
    # For A<K, verify slots 0..A, whose earlier contexts include slots <A.
    # For A=K, only slots <K-1 enter the last verification context.
    needed = positions < lengths.clamp_max(selected.shape[1] - 1)[:, None]
    known = ((~needed) | selected.eq(reference)).all(-1) & valid.all(-1)
    return lengths, known


def main():
    import torch_npu
    torch.npu.set_device(int(os.environ.get('LOCAL_RANK', '0')))
    from torch.utils.data import DataLoader, DistributedSampler, BatchSampler
    from tqdm.auto import tqdm
    import speculators.train.cli as cli
    import speculators.train.trainer as trainer_module
    from speculators.train.config import TrainConfig
    from speculators.train.data import ArrowDataset, CollateFn
    from speculators.train.distributed import get_dp_rank, get_dp_size
    from speculators.model import SpeculatorModel
    from speculators.models.dflash_prefix.core import DFlashPrefixDraftModel
    from speculators.models.dflash_prefix.selector_v2 import unary_topk
    from speculators.models.dflash_prefix.acceptance_head import AcceptedLengthHead
    from speculators.models.dflash_prefix.acceptance_features import build_acceptance_features
    from speculators.models.dflash_prefix.acceptance_loss import acceptance_boundary_loss

    @SpeculatorModel.register('frozen_prefix_acceptance_training')
    class FrozenAcceptance(DFlashPrefixDraftModel):
        def train(self, mode=True):
            super().train(False)
            self.acceptance_head.train(mode)
            return self

        def forward(self, hidden_states, input_ids, loss_mask,
                    verifier_last_hidden_states, document_ids, position_ids=None,
                    max_anchors=512, **kwargs):
            with torch.no_grad():
                hidden, unary, targets, aligned, indices = self._backbone_forward(
                    hidden_states, input_ids, loss_mask, verifier_last_hidden_states,
                    document_ids, position_ids, max_anchors=max_anchors)
                block = self.block_size
                n = indices.numel() // block
                ids = input_ids[0, indices].reshape(n, block)
                docs = document_ids[0, indices].reshape(n, block)
                valid = aligned.reshape(n, block)[:, 1:].bool()
                valid &= docs[:, 1:].eq(docs[:, :1])
                hidden = hidden.reshape(n, block, -1)[:, 1:]
                unary = unary.reshape(n, block, -1)[:, 1:]
                target = targets.reshape(n, block, -1)[:, 1:].argmax(-1)
                scores, candidates = unary_topk(unary, self.config.prefix_top_k)
                tables = self.prefix_head.prepare_tables(
                    hidden, candidates, scores, ids[:, 0], embedding_weight=self.embed_tokens.weight)
                selected, final = self.prefix_head.walk(tables, return_scores=True)
                features = build_acceptance_features(hidden, candidates, scores, selected, final)
                lengths, known = exact_boundaries(selected, target, ids[:, 1:], valid)
            logits = self.acceptance_head(features)
            count = known.sum().float()
            if bool(known.any()):
                loss, values = acceptance_boundary_loss(
                    logits[known], lengths[known], under_weight=3., over_weight=1., cost_weight=1.)
                # DDP must weight local means by global valid-anchor count.
                global_count = count.detach().clone()
                torch.distributed.all_reduce(global_count)
                loss = loss * count * torch.distributed.get_world_size() / global_count.clamp_min(1)
            else:
                # Every rank participates even when its batch has no known boundaries.
                global_count = count.detach().clone()
                torch.distributed.all_reduce(global_count)
                loss = logits.sum() * 0.
                values = {key: loss.detach() for key in (
                    'loss', 'cross_entropy', 'expected_cost_normalized', 'exact_accuracy',
                    'mean_under_tokens', 'mean_over_tokens', 'decision_cost')}
            metrics = {}
            for key, value in values.items():
                metrics[key + '_sum'] = value.detach() * count
                metrics[key + '_total'] = count.detach().clone()
            metrics['label_coverage_sum'] = count.detach()
            metrics['label_coverage_total'] = count.new_tensor(n)
            return None, loss, metrics

        def save_pretrained(self, save_directory, *, state_dict=None, **kwargs):
            result = super().save_pretrained(save_directory, state_dict=state_dict, **kwargs)
            source = self.state_dict() if state_dict is None else state_dict
            state = {k.removeprefix('acceptance_head.'): v.detach().cpu()
                     for k, v in source.items() if k.startswith('acceptance_head.')}
            torch.save(dict(architecture=self.acceptance_head.architecture, state_dict=state,
                            draft_checkpoint=os.environ['DRAFT_MODEL'],
                            supervision='exact_greedy_boundaries_on_valid_reference_contexts'),
                       Path(save_directory) / 'acceptance_head.pt')
            return result

    original_build = cli.build_draft_model

    def build(args, model_class, t2d, d2t, draft_vocab_size):
        model = original_build(args, DFlashPrefixDraftModel, t2d, d2t, draft_vocab_size)
        if model.config.prefix_selector_kind != 'compiled_prefix_v3':
            raise ValueError('Requires compiled_prefix_v3')
        if model.config.sample_from_anchor or model.use_draft_vocab:
            raise ValueError('Requires ordinary anchor alignment and full vocabulary')
        model.requires_grad_(False)
        model.eval()
        model.acceptance_head = AcceptedLengthHead(
            hidden_size=model.hidden_size, num_proposals=model.block_size - 1,
            projection_size=32, mlp_hidden_size=64)
        model.__class__ = FrozenAcceptance
        assert all(name.startswith('acceptance_head.') for name, p in model.named_parameters() if p.requires_grad)
        if get_dp_rank() == 0:
            print(model.acceptance_head, flush=True)
            print('Only acceptance_head parameters are trainable.', flush=True)
        return model

    def loaders(**kw):
        # Keep the native hidden-state fetcher and collation. Use every dataset row.
        dataset = ArrowDataset(
            datapath=kw['data_path'], max_len=kw['total_seq_len'], train_ratio=1., split='train',
            **{key: kw[key] for key in ('transfer', 'vllm_endpoint', 'on_missing', 'on_generate',
               'hidden_states_dtype', 'request_timeout', 'max_retries',
               'generation_validation_retries', 'max_consecutive_generation_failures')},
            model=kw['verifier_name_or_path'])
        world = get_dp_size()
        if len(dataset) % world:
            raise ValueError('Dataset row count must divide trainer world size to avoid padded/replayed rows')
        sampler = DistributedSampler(dataset, num_replicas=world, rank=get_dp_rank(),
                                     shuffle=True, seed=42, drop_last=False)
        batch_sampler = BatchSampler(sampler, batch_size=1, drop_last=False)
        collate = CollateFn(kw['total_seq_len'], kw['hidden_size'],
                            num_target_layers=kw['num_target_layers'],
                            dtype=dataset.hidden_states_dtype, preprocess=kw['preprocess'])
        loader = DataLoader(dataset, batch_sampler=batch_sampler, collate_fn=collate,
                            num_workers=0, pin_memory=False)
        if get_dp_rank() == 0:
            print(f'Full dataset: {len(dataset):,} rows; one pass; {len(loader):,} steps per rank', flush=True)
        return loader, None

    # Standard tqdm includes elapsed time, remaining time, and n/total steps.
    def progress(iterable, desc=None, **kwargs):
        return tqdm(iterable, total=len(iterable), desc='Dataset pass 1/1',
                    dynamic_ncols=True, unit='step',
                    bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} '
                               '[{elapsed}<{remaining}, {rate_fmt}]')

    cli.build_draft_model = build
    cli.create_train_val_loaders = loaders
    trainer_module.tqdm = progress
    config = TrainConfig.resolve()
    flat = config.flatten()
    if flat['epochs'] != 1 or flat['max_steps'] is not None or flat['fsdp_shard']:
        raise ValueError('Use epochs=1, no max_steps, and ordinary DDP')
    cli.main(config)


if __name__ == '__main__':
    main()
