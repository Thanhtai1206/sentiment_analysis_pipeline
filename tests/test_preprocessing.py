import pandas as pd
import pytest

from src.config import load_config, resolve
from src.preprocessing import clean_text, load_split


@pytest.mark.parametrize("raw, expected", [
    ("GV dạy rất HAY!!!", "giảng viên dạy rất hay!!!"),
    ("ko hiểu bài", "không hiểu bài"),
    ("bài giảng ok", "bài giảng ổn"),
    ("thầy dạy tạm ổn", "thầy dạy tạm ổn"),      # bản cũ biến thành "tạm tạm ổn"
    ("nội dung cô đọng", "nội dung cô đọng"),    # bản cũ biến thành "giảng viên đọng"
    ("  nhiều   khoảng   trắng ", "nhiều khoảng trắng"),
    ("icon 😍 vui", "icon vui"),
    (None, ""),
    (123, ""),
])
def test_clean_text(raw, expected):
    assert clean_text(raw) == expected


@pytest.mark.parametrize("split", ["train", "validation", "test"])
def test_raw_data_schema(split):
    df = load_split(load_config(), split)
    assert set(df.columns) == {"text", "label"}
    assert set(df["label"].unique()) <= {0, 1, 2}
    assert len(df) > 0


def test_test_set_not_leaked_into_train():
    cfg = load_config()
    tr = set(load_split(cfg, "train")["text"])
    te = load_split(cfg, "test")["text"]
    assert te.isin(tr).mean() < 0.01
