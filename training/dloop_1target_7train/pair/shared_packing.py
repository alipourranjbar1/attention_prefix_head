"""Identical document batches for both trainers; their objectives remain native."""
import sys


def epoch_batches(cfg, lengths, fingerprint, world, rank, epoch):
    from ascend_support.packed_data import TokenBudgetBatchSampler
    if cfg.max_anchors < 128:
        raise ValueError('Shared packing requires at least 128 sampled-start budget slots')
    sampler=TokenBudgetBatchSampler(lengths,cfg.max_sequence_length,world,rank,
                                    seed=cfg.seed,bucket_width=128,max_records=128)
    sampler.set_epoch(epoch)
    return list(sampler)


def install():
    from four_anchor import packed_data
    packed_data.epoch_batches=epoch_batches
    training=sys.modules.get('four_anchor.packed_train')
    if training is not None:
        training.epoch_batches=epoch_batches
