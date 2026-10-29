"""Đọc file cấu hình YAML và chuẩn hóa đường dẫn về thư mục gốc project."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "config.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["labels"] = {int(k): v for k, v in cfg["labels"].items()}
    return cfg


def resolve(path: str | Path) -> Path:
    """Đường dẫn tương đối trong config -> đường dẫn tuyệt đối từ gốc project."""
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p
