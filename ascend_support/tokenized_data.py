"""Read existing token IDs without adding a new sampling or packing method."""
import torch
from datasets import Dataset, load_from_disk
from specforge.data.on_the_fly import OnTheFlyDataset


class SavedTokenDataset(OnTheFlyDataset):
    """Use upstream's exact index-seeded short-row replacement policy.

    Only _sample differs: input_ids/loss_mask already exist, so tokenization
    is skipped. No multi-document packing, train/validation split or anchor
    selection is introduced here. Anchor sampling remains in OnlineDFlashModel.
    """

    def __init__(self, path, max_length, min_loss_tokens):
        self.dataset = load_from_disk(str(path))
        if not isinstance(self.dataset, Dataset):
            raise ValueError("DATASET must be one HF save_to_disk Dataset, not DatasetDict")
        if not {"input_ids", "loss_mask"} <= set(self.dataset.column_names):
            raise ValueError("Dataset needs input_ids and loss_mask")
        if len(self.dataset) == 0:
            raise ValueError("Dataset is empty")
        self.max_length = int(max_length)
        self.min_loss_tokens = int(min_loss_tokens)

    def _sample(self, row):
        record = self.dataset[row]
        ids = torch.as_tensor(record["input_ids"], dtype=torch.long)
        loss = torch.as_tensor(record["loss_mask"])
        if ids.ndim == 2 and ids.shape[0] == 1:
            ids = ids[0]
        if loss.ndim == 2 and loss.shape[0] == 1:
            loss = loss[0]
        if ids.ndim != 1 or loss.ndim != 1 or ids.shape != loss.shape:
            raise ValueError(f"Malformed input_ids/loss_mask at row {row}")
        ids, loss = ids[:self.max_length], loss[:self.max_length]
        if ids.numel() == 0 or loss.sum().item() < self.min_loss_tokens:
            return None
        if "attention_mask" in record:
            attention = torch.as_tensor(record["attention_mask"]).reshape(-1)[:len(ids)]
            if attention.shape != ids.shape or not torch.all(attention == 1):
                raise ValueError("Saved records must be unpadded; collator handles padding")
        return {"input_ids": ids.unsqueeze(0),
                "loss_mask": loss.unsqueeze(0),
                "attention_mask": torch.ones_like(ids).unsqueeze(0)}
