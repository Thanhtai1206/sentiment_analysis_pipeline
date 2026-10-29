"""Suy luận cho 1 câu hoặc cả file CSV.

    python -m src.predict "giảng viên nhiệt tình, dễ hiểu"
    python -m src.predict --file data/sample/sample_input.csv --text-col comment
"""
from __future__ import annotations

import argparse
from functools import lru_cache
from pathlib import Path

import numpy as np

from src.config import load_config, resolve
from src.evaluate import apply_threshold
from src.preprocessing import preprocess
from src.utils import load_json


class SentimentPredictor:
    """Nạp model 1 lần, dùng lại cho mọi lần gọi. Không nạp ngay khi import module."""

    def __init__(self, model_dir: str | Path | None = None, device: str | None = None):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        cfg = load_config()
        self.model_dir = resolve(model_dir or cfg["model"]["output_dir"])
        if not (self.model_dir / "config.json").exists():
            raise FileNotFoundError(
                f"Không tìm thấy checkpoint tại {self.model_dir}. "
                "Hãy huấn luyện (python -m src.train) hoặc chép checkpoint vào đây."
            )
        inf_path = self.model_dir / "inference_config.json"
        inf = load_json(inf_path) if inf_path.exists() else {}
        # checkpoint mới: ngưỡng tìm trên validation; checkpoint cũ: dùng fallback trong config
        self.threshold = inf["threshold"] if "threshold" in inf else cfg["threshold"].get("fallback_value")
        self.max_length = inf.get("max_length", cfg["model"]["max_length"])
        self.pre_cfg = inf.get("preprocessing", cfg["preprocessing"])
        self.labels = {int(k): v for k, v in inf.get("labels", cfg["labels"]).items()}

        # Ưu tiên tokenizer lưu kèm checkpoint (chạy offline); nếu không có thì tải từ Hub
        tok_src = self.model_dir if (self.model_dir / "tokenizer_config.json").exists() \
            else inf.get("base_model", cfg["model"]["pretrained_name"])
        self.tokenizer = AutoTokenizer.from_pretrained(tok_src)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_dir).to(self.device).eval()
        self._torch = torch

    def predict_proba(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        cleaned = [preprocess(t, self.pre_cfg) for t in texts]
        out = []
        for i in range(0, len(cleaned), batch_size):
            enc = self.tokenizer(cleaned[i:i + batch_size], padding=True, truncation=True,
                                 max_length=self.max_length, return_tensors="pt").to(self.device)
            enc = {k: v for k, v in enc.items() if k in ("input_ids", "attention_mask")}
            with self._torch.no_grad():
                out.append(self._torch.softmax(self.model(**enc).logits, dim=-1).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, len(self.labels)))

    def predict(self, text: str) -> dict:
        if not isinstance(text, str) or not text.strip():
            return {"label_id": -1, "label": "Không có dữ liệu", "confidence": 0.0, "probabilities": {}}
        probs = self.predict_proba([text])[0]
        pred = int(apply_threshold(probs[None, :], self.threshold)[0])
        n_tokens = len(self.tokenizer(preprocess(text, self.pre_cfg), truncation=True,
                                      max_length=self.max_length)["input_ids"])
        return {
            "label_id": pred,
            "label": self.labels[pred],
            "confidence": float(probs[pred]),
            "probabilities": {self.labels[i]: float(p) for i, p in enumerate(probs)},
            "threshold": self.threshold,
            "n_tokens": n_tokens,  # số token THỰC (không tính padding)
        }

    def predict_batch(self, texts: list[str]) -> list[int]:
        return apply_threshold(self.predict_proba(list(texts)), self.threshold).tolist()


@lru_cache(maxsize=1)
def get_predictor() -> SentimentPredictor:
    return SentimentPredictor()


def predict_sentiment(text: str) -> dict:
    """Giữ tên hàm cũ để demo/báo cáo không phải đổi."""
    return get_predictor().predict(text)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("text", nargs="?", help="Câu cần dự đoán")
    ap.add_argument("--file", help="CSV đầu vào")
    ap.add_argument("--text-col", default="comment")
    ap.add_argument("--out", help="CSV đầu ra (mặc định: <file>_predicted.csv)")
    a = ap.parse_args()
    p = get_predictor()
    if a.file:
        import pandas as pd

        df = pd.read_csv(a.file)
        probs = p.predict_proba(df[a.text_col].fillna("").astype(str).tolist())
        df["pred"] = apply_threshold(probs, p.threshold)
        df["pred_label"] = df["pred"].map(p.labels)
        df["confidence"] = probs.max(axis=1).round(4)
        out = a.out or str(Path(a.file).with_name(Path(a.file).stem + "_predicted.csv"))
        df.to_csv(out, index=False, encoding="utf-8-sig")
        print(df[[a.text_col, "pred_label", "confidence"]].to_string(index=False))
        print(f"\n-> Đã lưu {out}")
    else:
        r = p.predict(a.text or "tạm thời chưa có .")
        print(f"Nhãn: {r['label']} (độ tin cậy {r['confidence']:.2%}) | ngưỡng={r['threshold']}")
        for k, v in r["probabilities"].items():
            print(f"  {k:12s}: {v:.2%}")
