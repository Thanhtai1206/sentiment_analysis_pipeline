"""Tiền xử lý văn bản.

QUAN TRỌNG: train / validation / test / demo đều phải đi qua CÙNG hàm `preprocess`
để tránh lệch phân phối giữa lúc huấn luyện và lúc suy luận (train-serving skew).

Chạy độc lập:  python -m src.preprocessing   -> sinh data/processed/{train,validation,test}.csv
"""
from __future__ import annotations

import argparse
import re

import pandas as pd

from src.config import load_config, resolve

# Chỉ chuẩn hóa từ VIẾT TẮT. Không thay "thầy"/"cô" vì tiếng Việt tách âm tiết
# bằng dấu cách nên \bcô\b sẽ khớp nhầm cả "cô đọng", "cô lập"...
TEENCODE_MAP = {
    r"\bgv\b": "giảng viên",
    r"\bsv\b": "sinh viên",
    r"\b(?:đc|dc)\b": "được",
    r"\b(?:ko|khg|khong|kh|k)\b": "không",
    r"\b(?:ok|oke|okela|okay)\b": "ổn",
    r"\bvs\b": "với",
}
_TEENCODE_RE = [(re.compile(p), r) for p, r in TEENCODE_MAP.items()]
_SPECIAL_RE = re.compile(r"[^\w\s,.!?\-+%]")
_SPACE_RE = re.compile(r"\s+")

_segmenter = None


def clean_text(text, lowercase: bool = True, normalize_teencode: bool = True) -> str:
    if not isinstance(text, str):
        return ""
    if lowercase:
        text = text.lower()
    if normalize_teencode:
        for pattern, repl in _TEENCODE_RE:
            text = pattern.sub(repl, text)
    text = _SPECIAL_RE.sub(" ", text)
    return _SPACE_RE.sub(" ", text).strip()


def segment_words(text: str, segmenter: str = "none") -> str:
    """Tách từ tiếng Việt (PhoBERT khuyến nghị). 'none' -> giữ nguyên."""
    if segmenter in (None, "none") or not text:
        return text
    if segmenter == "underthesea":
        global _segmenter
        if _segmenter is None:
            from underthesea import word_tokenize  # import lười, chỉ khi cần

            _segmenter = word_tokenize
        return _segmenter(text, format="text")
    raise ValueError(f"word_segmenter không hỗ trợ: {segmenter}")


def preprocess(text, cfg_pre: dict | None = None) -> str:
    cfg_pre = cfg_pre or {}
    text = clean_text(
        text,
        lowercase=cfg_pre.get("lowercase", True),
        normalize_teencode=cfg_pre.get("normalize_teencode", True),
    )
    return segment_words(text, cfg_pre.get("word_segmenter", "none"))


def load_split(cfg: dict, split: str) -> pd.DataFrame:
    """Đọc 1 tập dữ liệu thô, trả về DataFrame 2 cột chuẩn: text, label."""
    d = cfg["data"]
    fname = {"train": d["train_file"], "validation": d["val_file"], "test": d["test_file"]}[split]
    df = pd.read_csv(resolve(d["raw_dir"]) / fname)
    df = df.rename(columns={d["text_col"]: "text", d["label_col"]: "label"})[["text", "label"]]
    df = df.dropna().reset_index(drop=True)
    df["label"] = df["label"].astype(int)
    return df


def prepare_data(cfg: dict) -> dict[str, pd.DataFrame]:
    """Làm sạch cả 3 tập bằng cùng 1 hàm và lưu vào data/processed/."""
    out_dir = resolve(cfg["data"]["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    splits = {}
    for split in ("train", "validation", "test"):
        df = load_split(cfg, split)
        df["text_clean"] = df["text"].map(lambda t: preprocess(t, cfg["preprocessing"]))
        n_before = len(df)
        df = df[df["text_clean"].str.len() > 0].reset_index(drop=True)
        if split == "train":
            df = df.drop_duplicates(subset=["text_clean", "label"]).reset_index(drop=True)
        df.to_csv(out_dir / f"{split}.csv", index=False, encoding="utf-8")
        print(f"[{split:10s}] {n_before:>6,} -> {len(df):>6,} mẫu | nhãn: "
              f"{df['label'].value_counts().sort_index().to_dict()}")
        splits[split] = df
    _check_overlap(splits)
    return splits


def _check_overlap(splits: dict[str, pd.DataFrame]) -> None:
    """Cảnh báo nếu câu trong test cũng xuất hiện trong train (rò rỉ dữ liệu)."""
    train_set = set(splits["train"]["text_clean"])
    for name in ("validation", "test"):
        dup = splits[name]["text_clean"].isin(train_set).sum()
        print(f"  ⚠ {name}: {dup} câu trùng nguyên văn với train "
              f"({dup / len(splits[name]):.1%})" if dup else f"  ✓ {name}: không trùng với train")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Làm sạch dữ liệu thô -> data/processed/")
    ap.add_argument("--config", default=None)
    prepare_data(load_config(ap.parse_args().config))
