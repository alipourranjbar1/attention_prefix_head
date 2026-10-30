"""Prompt ingestion, teacher-data preparation, and tokenized dataset loading."""
from pathlib import Path
import hashlib
import json

import torch


def prompt_tokens(row, tokenizer, max_length=None):
    if "messages" in row:
        ids = tokenizer.apply_chat_template(row["messages"], tokenize=True,
                                            add_generation_prompt=True, enable_thinking=False)
    elif "input_ids" in row:
        ids = list(row["input_ids"])
        # Prepared training records contain both prompt and completion.
        if "loss_mask" in row:
            first = next((i for i, v in enumerate(row["loss_mask"]) if v), len(ids))
            ids = ids[:first]
    elif "prompt" in row or "text" in row:
        text = row["prompt"] if "prompt" in row else row["text"]
        ids = tokenizer(text, add_special_tokens=True)["input_ids"]
    else:
        raise ValueError("Expected messages, prompt, text, or input_ids")
    if not ids:
        if tokenizer.bos_token_id is None:
            raise ValueError("Prompt must contain at least one token")
        ids = [tokenizer.bos_token_id]
    if max_length is not None:
        ids = ids[-max_length:]
    return ids


def load_records(path, split="train"):
    from datasets import DatasetDict, load_dataset, load_from_disk
    path = Path(path)
    if path.is_dir():
        data = load_from_disk(str(path))
        return data[split] if isinstance(data, DatasetDict) else data
    return load_dataset("json", data_files=str(path), split="train")


def _trainable_batch(batch, max_sequence_length):
    """Arrow batch predicate: match training_example's usable prefix window.

    Keep the scan in Arrow instead of decoding billions of tokens into Python.
    Invalid schemas/weights are errors; a well-formed unsupervised row is not.
    """
    import numpy as np
    import pyarrow as pa
    import pyarrow.compute as pc

    ids = batch["input_ids"].combine_chunks()
    if ids.null_count:
        raise ValueError("input_ids contains a null sequence")
    ids = pc.list_slice(ids, 0, max_sequence_length)
    lengths = pc.list_value_length(ids)
    if "loss_mask" not in batch.column_names:
        return pc.greater(lengths, 1).to_numpy(zero_copy_only=False)
    masks = batch["loss_mask"].combine_chunks()
    if masks.null_count:
        raise ValueError("loss_mask contains a null sequence")
    masks = pc.list_slice(masks, 0, max_sequence_length)
    if pc.any(pc.not_equal(pc.list_value_length(masks), lengths)).as_py():
        raise ValueError("loss_mask must align one-to-one with input_ids")
    flat = pc.list_flatten(masks)
    if flat.null_count:
        raise ValueError("loss_mask contains a null weight")
    weights = pc.cast(flat, pa.float64())
    if pc.any(pc.invert(pc.is_finite(weights))).as_py() or pc.any(pc.less(weights, 0)).as_py():
        raise ValueError("loss_mask weights must be finite and nonnegative")
    # Position zero cannot be predicted from a nonempty prefix. The stop index
    # also excludes supervision that disappears under training-time truncation.
    candidates = pc.list_slice(masks, 1, max_sequence_length)
    positive = pc.greater(pc.cast(pc.list_flatten(candidates), pa.float64()), 0)
    parents = pc.filter(pc.list_parent_indices(candidates), positive)
    keep = np.zeros(len(batch), dtype=bool)
    keep[parents.to_numpy(zero_copy_only=False)] = True
    return keep


def filter_training_records(data, max_sequence_length, cache_dir):
    """Cache a shared row-index filter before creating any distributed sampler.

    The original dataset is read-only. A file lock lets all local DDP ranks use
    one scan and the same retained rows. The fingerprint excludes output paths,
    so a resumed run can use another cache directory without changing its data.
    """
    from filelock import FileLock

    if not len(data):
        raise ValueError("Empty dataset")
    if "input_ids" not in data.column_names:
        raise ValueError("Training needs tokenized input_ids; run `dflash4a prepare` first")
    if max_sequence_length < 2:
        raise ValueError("max_sequence_length must be at least two")
    identity = {"filter_version": 1, "source_fingerprint": data._fingerprint,
                "max_sequence_length": max_sequence_length}
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    cache = Path(cache_dir).expanduser().resolve()
    cache.mkdir(parents=True, exist_ok=True)
    index_file = cache / f"trainable_{fingerprint}.arrow"
    with FileLock(str(index_file) + ".lock"):
        filtered = data.with_format("arrow").filter(
            _trainable_batch, batched=True, batch_size=512,
            fn_kwargs={"max_sequence_length": max_sequence_length},
            cache_file_name=str(index_file), load_from_cache_file=True,
            new_fingerprint=fingerprint, desc="Filtering rows without usable supervision",
        ).with_format(None)
        summary = {**identity, "input_rows": len(data), "trainable_rows": len(filtered),
                   "dropped_rows": len(data) - len(filtered), "index_file": str(index_file)}
        index_file.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    if not len(filtered):
        raise ValueError("No trainable rows: no positive loss_mask after a nonempty prefix within max_sequence_length")
    # Preserve the original fingerprint/order when nothing was excluded.
    return (data if len(filtered) == len(data) else filtered), summary


def training_example(data, rng, num_proposals, max_sequence_length, vocab_size, device, row_index=None):
    for _ in range(1000):
        row = data[rng.randrange(len(data)) if row_index is None else row_index]
        if "input_ids" not in row:
            raise ValueError("Training needs tokenized input_ids; run `dflash4a prepare` first")
        ids = list(row["input_ids"])[:max_sequence_length]
        mask = list(row.get("loss_mask", [1] * len(row["input_ids"])))[:max_sequence_length]
        if len(mask) != len(ids):
            raise ValueError("loss_mask must align one-to-one with input_ids")
        if any(v < 0 for v in mask):
            raise ValueError("loss_mask weights must be nonnegative")
        if any(t < 0 or t >= vocab_size for t in ids):
            raise ValueError("Dataset contains an out-of-vocabulary token")
        starts = [i for i in range(1, len(ids)) if mask[i] > 0]
        if not starts:
            if row_index is not None:
                raise ValueError(f"Dataset row {row_index} has no supervised token after a nonempty prefix")
            continue
        start = rng.choice(starts)
        labels = ids[start:start + num_proposals]
        weights = mask[start:start + num_proposals]
        missing = num_proposals - len(labels)
        return (torch.tensor([ids[:start]], device=device, dtype=torch.long),
                torch.tensor([labels + [0] * missing], device=device, dtype=torch.long),
                torch.tensor([weights + [0] * missing], device=device, dtype=torch.float32))
    raise ValueError("Could not find a trainable example with nonempty prefix and positive loss_mask")


def prepare_data(target, tokenizer, source, destination, max_new_tokens=256,
                 max_prompt_tokens=1536, limit=None, split="train"):
    """Regenerate completions from the target; write loss_mask=0 over the prompt."""
    from .runtime import Decoder, TargetRunner
    records = load_records(source, split)
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite prepared data: {path}")
    runner = TargetRunner(target, [0])
    device = next(target.parameters()).device
    try:
        with path.open("w") as stream:
            for i, row in enumerate(records):
                if limit is not None and i >= limit:
                    break
                ids = prompt_tokens(row, tokenizer, max_prompt_tokens)
                out = Decoder(runner, None).generate(torch.tensor([ids], device=device), max_new_tokens, mode="ar")
                stream.write(json.dumps({"input_ids": ids + out.token_ids,
                                         "loss_mask": [0] * len(ids) + [1] * len(out.token_ids),
                                         "seq_len": len(ids) + len(out.token_ids), "source_index": i}) + "\n")
                stream.flush()
    finally:
        runner.close()
