"""Prompt ingestion, teacher-data preparation, and tokenized dataset loading."""
from pathlib import Path
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


def training_example(data, rng, num_proposals, max_sequence_length, vocab_size, device):
    for _ in range(1000):
        row = data[rng.randrange(len(data))]
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
