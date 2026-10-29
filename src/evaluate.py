"""Đánh giá mô hình: suy luận theo batch, ngưỡng lớp Trung tính, metric, ma trận nhầm lẫn.

Chạy độc lập (đánh giá checkpoint đã lưu trên tập test):
    python -m src.evaluate --split test
"""
from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from src.config import load_config, resolve
from src.utils import load_json, save_json

NEUTRAL = 1


# ----------------------------------------------------------------- ngưỡng
def apply_threshold(probs: np.ndarray, threshold: float | None) -> np.ndarray:
    """threshold=None -> argmax. Ngược lại: P(neutral) >= t -> 1, còn lại so 0 vs 2."""
    probs = np.asarray(probs)
    if threshold is None:
        return probs.argmax(axis=1)
    non_neutral = np.where(probs[:, 0] >= probs[:, 2], 0, 2)
    return np.where(probs[:, NEUTRAL] >= threshold, NEUTRAL, non_neutral)


def tune_threshold(probs, y_true, t_min=0.1, t_max=0.6, step=0.02) -> tuple[float, float]:
    """Quét ngưỡng tối đa hóa Macro-F1. CHỈ gọi trên tập validation."""
    best_t, best_f1 = None, f1_score(y_true, apply_threshold(probs, None), average="macro")
    for t in np.arange(t_min, t_max + 1e-9, step):
        f1 = f1_score(y_true, apply_threshold(probs, float(t)), average="macro")
        if f1 > best_f1:
            best_t, best_f1 = round(float(t), 4), f1
    return best_t, best_f1


# ----------------------------------------------------------------- metric
def compute_metrics(y_true, y_pred, label_names: dict[int, str]) -> dict:
    labels = sorted(label_names)
    rep = classification_report(
        y_true, y_pred, labels=labels, target_names=[label_names[i] for i in labels],
        output_dict=True, zero_division=0,
    )
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
        "report": rep,
    }


def report_to_frame(metrics: dict) -> pd.DataFrame:
    return pd.DataFrame(metrics["report"]).transpose()


def plot_confusion(y_true, y_pred, label_names: dict[int, str], title: str, path, normalize=False):
    labels = sorted(label_names)
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true" if normalize else None)
    names = [label_names[i] for i in labels]
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt=".2f" if normalize else "d", cmap="Blues",
                xticklabels=names, yticklabels=names, ax=ax)
    ax.set_xlabel("Dự đoán")
    ax.set_ylabel("Nhãn thật")
    ax.set_title(title)
    fig.tight_layout()
    path = resolve(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return cm


# ----------------------------------------------------------------- suy luận
def predict_proba(model, loader, device, fp16: bool = False):
    """Trả về (probs [N,3], labels [N] hoặc None, loss trung bình hoặc None)."""
    import torch

    model.eval()
    all_probs, all_labels, total_loss, n = [], [], 0.0, 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("labels", None)
            with torch.autocast(device_type=device.type, enabled=fp16 and device.type == "cuda"):
                logits = model(**batch).logits.float()
            if labels is not None:
                total_loss += torch.nn.functional.cross_entropy(logits, labels, reduction="sum").item()
                n += labels.numel()
                all_labels.append(labels.cpu().numpy())
            all_probs.append(torch.softmax(logits, dim=-1).cpu().numpy())
    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels) if all_labels else None
    return probs, labels, (total_loss / n if n else None)


def save_predictions(df: pd.DataFrame, probs, preds, path) -> pd.DataFrame:
    out = df[["text", "text_clean", "label"]].copy()
    out["pred"] = preds
    for i in range(probs.shape[1]):
        out[f"prob_{i}"] = probs[:, i].round(4)
    out["correct"] = out["label"] == out["pred"]
    path = resolve(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False, encoding="utf-8-sig")
    return out


def evaluate_checkpoint(cfg: dict, split: str = "test") -> dict:
    """Nạp checkpoint đã huấn luyện và đánh giá trên 1 tập (mặc định: test)."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from src.dataset import build_loader, load_processed
    from src.utils import get_device

    model_dir = resolve(cfg["model"]["output_dir"])
    inf_cfg = load_json(model_dir / "inference_config.json")
    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)

    df = load_processed(cfg, split)
    loader = build_loader(df, tokenizer, cfg, shuffle=False, batch_size=cfg["training"]["eval_batch_size"])
    probs, y, _ = predict_proba(model, loader, device)
    return summarize(cfg, df, probs, y, inf_cfg.get("threshold"), split, prefix="phobert")


def summarize(cfg, df, probs, y, threshold, split, prefix="phobert") -> dict:
    """Tính metric (argmax & có ngưỡng), lưu CSV + hình + file dự đoán."""
    names = cfg["labels"]
    res_dir = cfg["results"]
    out = {}
    variants = {"argmax": None}
    if threshold is not None:
        variants["threshold"] = threshold
    for tag, t in variants.items():
        preds = apply_threshold(probs, t)
        m = compute_metrics(y, preds, names)
        report_to_frame(m).to_csv(resolve(res_dir["metrics_dir"]) / f"{prefix}_{split}_{tag}.csv",
                                  encoding="utf-8-sig")
        plot_confusion(y, preds, names, f"PhoBERT – {split} ({tag}{'' if t is None else f' t={t}'})",
                       f"{res_dir['figures_dir']}/{prefix}_cm_{split}_{tag}.png")
        out[tag] = {"accuracy": m["accuracy"], "macro_f1": m["macro_f1"], "weighted_f1": m["weighted_f1"],
                    "threshold": t}
        print(f"[{split} | {tag:9s}] acc={m['accuracy']:.4f}  macro-F1={m['macro_f1']:.4f}")
    final_t = variants.get("threshold")
    save_predictions(df, probs, apply_threshold(probs, final_t),
                     f"{res_dir['predictions_dir']}/{prefix}_{split}_predictions.csv")
    save_json(out, resolve(res_dir["metrics_dir"]) / f"{prefix}_{split}_summary.json")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--split", default="test", choices=["validation", "test"])
    a = ap.parse_args()
    cfg = load_config(a.config)
    for k in ("metrics_dir", "figures_dir", "predictions_dir"):
        resolve(cfg["results"][k]).mkdir(parents=True, exist_ok=True)
    evaluate_checkpoint(cfg, a.split)
