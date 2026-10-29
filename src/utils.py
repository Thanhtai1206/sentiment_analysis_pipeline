"""Các hàm tiện ích dùng chung: seed, device, lưu JSON."""
from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np


def set_seed(seed: int = 42) -> None:
    """Cố định seed để kết quả huấn luyện có thể tái lập."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_device():
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_json(obj, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: str | Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)
