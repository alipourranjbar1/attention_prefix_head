"""Length-bucketed, document-isolated token-budget packing.

A pack is a batch of independent, right-padded documents, not a concatenated
conversation. Both real tokens and B * padded_length are <= token_budget.
Every original dataset index is visited once per epoch; the upstream seeded
replacement of short rows is preserved. Only distributed tail padding repeats
packs (at most world_size - 1), and that padding is explicitly reported.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler
from tqdm import tqdm

from .tokenized_data import SavedTokenDataset

VERSION = 1


def index_metadata(dataset):
    return dict(version=VERSION, fingerprint=dataset.dataset._fingerprint,
                rows=len(dataset), max_length=dataset.max_length,
                min_loss_tokens=dataset.min_loss_tokens)


def build_index(dataset, cache_dir):
    metadata = index_metadata(dataset)
    key = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()[:24]
    path = Path(cache_dir) / f"packing_index_{key}.npz"
    if path.exists():
        load_index(dataset, path)
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    lengths = np.zeros(len(dataset), dtype=np.int32)
    eligible = np.zeros(len(dataset), dtype=bool)
    position = 0
    view = dataset.dataset.with_format("numpy")
    with tqdm(total=len(dataset), desc="Indexing records for packing", unit="record") as bar:
        for batch in view.iter(batch_size=1024):
            for i in range(len(batch["input_ids"])):
                ids = np.asarray(batch["input_ids"][i])
                mask = np.asarray(batch["loss_mask"][i])
                if ids.ndim == 2 and ids.shape[0] == 1:
                    ids = ids[0]
                if mask.ndim == 2 and mask.shape[0] == 1:
                    mask = mask[0]
                if ids.ndim != 1 or mask.ndim != 1:
                    raise ValueError(f"Saved row {position} must contain one sequence")
                ids, mask = ids[:dataset.max_length], mask[:dataset.max_length]
                if ids.shape != mask.shape:
                    raise ValueError(f"input_ids/loss_mask mismatch at row {position}")
                if "attention_mask" in batch:
                    attention = np.asarray(batch["attention_mask"][i]).reshape(-1)[:len(ids)]
                    if attention.shape != ids.shape or not np.all(attention == 1):
                        raise ValueError(f"Saved row {position} must be unpadded")
                if not np.all(np.isfinite(mask)) or np.any(mask < 0):
                    raise ValueError(f"Invalid loss_mask at row {position}")
                lengths[position] = len(ids)
                eligible[position] = bool(len(ids) and mask.sum() >= dataset.min_loss_tokens)
                position += 1
            bar.update(len(batch["input_ids"]))
    resolved = np.arange(len(dataset), dtype=np.int64)
    # Match OnTheFlyDataset.__getitem__, including its ten-attempt limit.
    for index in np.flatnonzero(~eligible):
        index = int(index)
        replacements = random.Random(index)
        row = index
        for _ in range(dataset._MAX_ATTEMPTS):
            if eligible[row]:
                resolved[index] = row
                break
            row = replacements.randrange(len(dataset))
        else:
            raise ValueError(f"No eligible replacement for row {index}; upstream would also fail")
    effective = lengths[resolved]
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("wb") as f:
        np.savez(f, metadata=json.dumps(metadata, sort_keys=True), lengths=effective,
                 resolved_rows=resolved, replaced_rows=int((resolved != np.arange(len(dataset))).sum()))
    os.replace(temporary, path)
    return path


def load_index(dataset, path):
    with np.load(path, allow_pickle=False) as saved:
        metadata = json.loads(str(saved["metadata"]))
        if metadata != index_metadata(dataset):
            raise ValueError("Packing index does not match this dataset/record limit")
        lengths = saved["lengths"].copy()
        replaced = int(saved["replaced_rows"])
    if lengths.shape != (len(dataset),) or np.any(lengths <= 0):
        raise ValueError("Malformed packing index")
    return lengths, replaced


class TokenBudgetBatchSampler(Sampler):
    """Same number of calls on every rank, deterministic across resume/epochs.

    Fixed width length buckets give an epoch-invariant pack count. Records are
    shuffled inside buckets, then packs are shuffled. The RNG is independent
    of Torch/model RNG. No records are dropped to make ranks equally long.
    """
    def __init__(self, lengths, token_budget, world_size=1, rank=0, seed=42,
                 bucket_width=128, max_records=128):
        self.lengths = np.asarray(lengths, dtype=np.int64)
        self.token_budget = int(token_budget)
        self.world_size, self.rank, self.seed = int(world_size), int(rank), int(seed)
        self.bucket_width, self.max_records = int(bucket_width), int(max_records)
        self.epoch = 0
        if not len(self.lengths) or np.any(self.lengths <= 0):
            raise ValueError("Packing requires nonempty positive record lengths")
        if min(self.token_budget, self.world_size, self.bucket_width, self.max_records) < 1:
            raise ValueError("Packing budgets and sizes must be positive")
        if not 0 <= self.rank < self.world_size or self.lengths.max() > self.token_budget:
            raise ValueError("Invalid rank, or a record exceeds the token budget")
        upper = np.minimum(((self.lengths + self.bucket_width - 1) // self.bucket_width)
                           * self.bucket_width, self.token_budget)
        self.groups = [(int(bound), np.flatnonzero(upper == bound)) for bound in np.unique(upper)]
        self.pack_count = sum(math.ceil(len(indices) / self.capacity(bound))
                              for bound, indices in self.groups)
        self.steps = math.ceil(self.pack_count / self.world_size)

    def capacity(self, bound):
        return min(self.max_records, self.token_budget // bound)

    def __len__(self):
        return self.steps

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def global_packs(self):
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch]))
        packs = []
        for bound, indices in self.groups:
            shuffled = rng.permutation(indices)
            capacity = self.capacity(bound)
            packs.extend(shuffled[i:i+capacity].tolist() for i in range(0, len(shuffled), capacity))
        order = rng.permutation(len(packs))
        packs = [packs[int(i)] for i in order]
        extra = self.steps * self.world_size - len(packs)
        packs.extend(packs[i % self.pack_count].copy() for i in range(extra))
        return packs

    def __iter__(self):
        yield from self.global_packs()[self.rank::self.world_size]

    def describe(self):
        allocated = sum(bound * len(indices) for bound, indices in self.groups)
        return dict(packing="length_bucketed_independent_documents", input_rows=len(self.lengths),
                    real_tokens=int(self.lengths.sum()), tokens_per_rank_budget=self.token_budget,
                    padded_tokens_per_rank_limit=self.token_budget, bucket_width=self.bucket_width,
                    max_records_per_pack=self.max_records, world_size=self.world_size,
                    packs_before_distributed_padding=self.pack_count,
                    repeated_tail_packs=self.steps*self.world_size-self.pack_count,
                    steps_per_epoch=self.steps,
                    mean_records_per_pack=float(len(self.lengths)/self.pack_count),
                    mean_real_tokens_per_pack=float(self.lengths.sum()/self.pack_count),
                    overall_token_budget_utilization=float(self.lengths.sum()/(self.pack_count*self.token_budget)),
                    token_utilization_lower_bound=float(self.lengths.sum()/allocated))


class PackCollator:
    def __init__(self, token_budget):
        self.token_budget = int(token_budget)

    def __call__(self, records):
        lengths = torch.tensor([r["input_ids"].shape[-1] for r in records], dtype=torch.long)
        width, batch = int(lengths.max()), len(records)
        if width * batch > self.token_budget:
            raise ValueError("Packing/index mismatch: padded tokens exceed per-rank budget")
        result = {"lengths": lengths}
        for key in ("input_ids", "loss_mask", "attention_mask"):
            padded = torch.zeros(batch, width, dtype=records[0][key].dtype)
            for i, record in enumerate(records):
                padded[i, :int(lengths[i])] = record[key].reshape(-1)
            result[key] = padded
        return result


def make_loader(args, world_size, rank):
    dataset = SavedTokenDataset(args.train_data_path, args.max_length, 4*args.block_size)
    lengths, _ = load_index(dataset, args.packing_index)
    sampler = TokenBudgetBatchSampler(lengths, args.packed_token_budget, world_size, rank,
                                     args.seed, args.packing_bucket_width,
                                     min(args.packing_max_records, args.num_anchors))
    generator = torch.Generator().manual_seed(args.seed + rank)
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=args.dataloader_num_workers,
                        collate_fn=PackCollator(args.packed_token_budget), generator=generator)
    return loader, sampler


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--max-length", type=int, default=3072)
    p.add_argument("--token-budget", type=int, default=9216)
    p.add_argument("--num-anchors", type=int, default=768)
    p.add_argument("--world-size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--bucket-width", type=int, default=128)
    p.add_argument("--max-records", type=int, default=128)
    a = p.parse_args()
    if a.num_anchors < 1 or a.epochs < 1:
        raise ValueError("num-anchors and epochs must be positive")
    ds = SavedTokenDataset(a.dataset, a.max_length, 64)
    index = build_index(ds, a.cache_dir)
    lengths, replaced = load_index(ds, index)
    sampler = TokenBudgetBatchSampler(lengths, a.token_budget, a.world_size,
                                     bucket_width=a.bucket_width,
                                     max_records=min(a.max_records, a.num_anchors))
    report = dict(**sampler.describe(), max_record_length=a.max_length,
                  anchors_per_rank_pack=a.num_anchors, short_rows_replaced=replaced,
                  index_file=str(index.resolve()), dataset_fingerprint=ds.dataset._fingerprint,
                  epochs=a.epochs, total_steps=a.epochs*len(sampler))
    output = Path(a.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
