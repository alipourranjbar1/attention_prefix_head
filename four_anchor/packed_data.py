"""3072-token distributed packing, with independent teacher prefills per document."""
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import argparse
import hashlib
import json

import numpy as np
import torch
from filelock import FileLock
from torch.utils.data import Dataset

from speculators.train.distributed_batch_sampler import MultipackDistributedBatchSamplerV2
from .data import load_records, filter_training_records


def prepare_records(cfg):
    cache = Path(cfg.data_cache_dir or Path(cfg.output) / ".data_cache")
    data, summary = filter_training_records(load_records(cfg.data, cfg.split), cfg.max_sequence_length, cache)
    if cfg.train_data_ratio < 1:
        split = data.train_test_split(train_size=cfg.train_data_ratio, seed=cfg.seed)
        data = split["train"]
        summary["held_out_rows"] = len(split["test"])
    summary["training_rows"] = len(data)
    key = f"lengths_{data._fingerprint}_{cfg.max_sequence_length}"
    path = cache / (key + ".npy")
    with FileLock(str(path) + ".lock"):
        if not path.exists():
            import pyarrow.compute as pc
            lengths = np.concatenate([pc.list_value_length(batch["input_ids"]).to_numpy()
                                      for batch in data.with_format("arrow").iter(batch_size=8192)])
            np.save(path, np.minimum(lengths, cfg.max_sequence_length).astype(np.int64))
    return data, np.load(path), summary


def epoch_batches(cfg, lengths, fingerprint, world, rank, epoch):
    """Cache the actual upstream multipack plan, not an estimated step budget."""
    identity = ["multipack_v1", fingerprint, cfg.max_sequence_length, cfg.seed, world, rank, epoch]
    key = hashlib.sha256(json.dumps(identity).encode()).hexdigest()
    cache = Path(cfg.data_cache_dir or Path(cfg.output) / ".data_cache")
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"pack_{key}.npz"
    with FileLock(str(path) + ".lock"):
        if not path.exists():
            sampler = MultipackDistributedBatchSamplerV2(cfg.max_sequence_length, lengths, world, rank, seed=cfg.seed)
            sampler.set_epoch(epoch)
            batches = list(sampler)
            if not batches:
                raise ValueError("Not enough training records for one packed batch on every rank")
            np.savez(path, indices=np.concatenate(batches), offsets=np.cumsum([0] + [len(b) for b in batches]))
    with np.load(path) as saved:
        indices, offsets = saved["indices"], saved["offsets"]
    return [indices[a:b] for a, b in zip(offsets[:-1], offsets[1:])]


def schedule(cfg, data, lengths, world, verbose=True):
    counts = []
    for epoch in range(cfg.epochs):
        count = len(epoch_batches(cfg, lengths, data._fingerprint, world, 0, epoch))
        counts.append((count + cfg.gradient_accumulation - 1) // cfg.gradient_accumulation)
        if verbose:
            print(json.dumps({"event": "packing_plan", "epoch": epoch + 1, "packed_batches_per_rank": count,
                              "optimizer_steps": counts[-1]}), flush=True)
    return counts


class TeacherRecords(Dataset):
    """Workers only have CPU metadata; target transformer lives on server NPUs."""
    def __init__(self, data, cfg, target_config, local_runner=None):
        self.data, self.cfg, self.target_config = data, cfg, target_config
        self.local_runner, self.remote = local_runner, None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        cfg = self.cfg
        row = self.data[int(index)]
        ids = torch.tensor(row["input_ids"][:cfg.max_sequence_length], dtype=torch.long)[None]
        if (ids < 0).any() or (ids >= self.target_config.vocab_size).any():
            raise ValueError("Dataset contains an out-of-vocabulary token")
        weights = torch.tensor(row.get("loss_mask", [1.] * ids.numel())[:ids.numel()], dtype=torch.float32)
        if self.local_runner is not None:
            runner = self.local_runner
            device = runner.target.get_input_embeddings().weight.device
            state = runner.prefill(ids.to(device))
            all_features = state.features.reshape(1, ids.numel(), len(runner.layer_ids), -1)
            final = all_features[:, :, runner.layer_ids.index(self.target_config.num_hidden_layers - 1)].cpu()
            features = all_features[:, :, :len(cfg.target_layer_ids)].flatten(2).cpu()
        else:
            if self.remote is None:
                from .remote_teacher import RemoteTeacher
                marker = SimpleNamespace(config=self.target_config, get_input_embeddings=lambda:
                                         SimpleNamespace(weight=torch.empty(0, dtype=getattr(torch, cfg.dtype))))
                self.remote = RemoteTeacher(marker, cfg.target_layer_ids, cfg.vllm_endpoint,
                                            cfg.hidden_states_dir, cfg.teacher_model_name,
                                            cfg.request_timeout, cfg.max_retries, include_final=True)
            state = self.remote.prefill(ids)
            features, final = state.features, state.final_hidden
        return {"ids": ids[0], "weights": weights, "features": features[0], "final": final[0]}

    def close(self):
        if self.remote is not None:
            self.remote.close()


def collate_documents(rows):
    lengths = [len(row["ids"]) for row in rows]
    out = {key: torch.cat([row[key] for row in rows]) for key in ("ids", "weights", "features", "final")}
    out["positions"] = torch.cat([torch.arange(n) for n in lengths])
    out["documents"] = torch.repeat_interleave(torch.arange(len(rows)), torch.tensor(lengths))
    out["offsets"] = torch.tensor(np.cumsum([0] + lengths[:-1]), dtype=torch.long)
    out["lengths"] = torch.tensor(lengths, dtype=torch.long)
    return out


def main():
    from .train import TrainConfig
    parser = argparse.ArgumentParser(description="Count actual packed optimizer steps without starting NPUs")
    parser.add_argument("--config", required=True)
    parser.add_argument("--world-size", required=True, type=int)
    args = parser.parse_args()
    cfg = TrainConfig(**json.loads(Path(args.config).read_text()))
    data, lengths, summary = prepare_records(cfg)
    counts = schedule(cfg, data, lengths, args.world_size)
    result = {"event": "packed_training_plan", **summary, "sequence_length_per_rank": cfg.max_sequence_length,
              "world_size": args.world_size, "steps_per_epoch": counts, "total_steps": sum(counts),
              "training_tokens": int(lengths.sum()), "max_blocks_per_batch": cfg.max_anchors,
              "loss_fn": cfg.loss_fn, "dataset_fingerprint": data._fingerprint}
    Path(cfg.output).mkdir(parents=True, exist_ok=True)
    (Path(cfg.output) / "packing_plan.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
