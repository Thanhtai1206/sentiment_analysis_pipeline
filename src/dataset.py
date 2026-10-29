"""Dataset & DataLoader cho PhoBERT."""
from __future__ import annotations

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import DataCollatorWithPadding

from src.config import resolve


class SentimentDataset(Dataset):
    """Tokenize từng câu; KHÔNG đẩy tensor lên GPU ở đây (việc đó làm trong vòng train),
    nhờ vậy DataLoader dùng được num_workers > 0 và pin_memory."""

    def __init__(self, texts, tokenizer, labels=None, max_length: int = 128):
        self.texts = list(texts)
        self.labels = None if labels is None else list(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(str(self.texts[idx]), truncation=True, max_length=self.max_length)
        item = {k: torch.tensor(v) for k, v in enc.items() if k in ("input_ids", "attention_mask")}
        if self.labels is not None:
            item["labels"] = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        return item


def load_processed(cfg: dict, split: str) -> pd.DataFrame:
    path = resolve(cfg["data"]["processed_dir"]) / f"{split}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Chưa có {path}. Hãy chạy: python -m src.preprocessing")
    return pd.read_csv(path, keep_default_na=False)


def build_loader(df: pd.DataFrame, tokenizer, cfg: dict, shuffle: bool, batch_size: int) -> DataLoader:
    ds = SentimentDataset(df["text_clean"], tokenizer, df["label"], cfg["model"]["max_length"])
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=DataCollatorWithPadding(tokenizer),  # dynamic padding theo batch
        num_workers=cfg["training"].get("num_workers", 0),
        pin_memory=torch.cuda.is_available(),
    )
