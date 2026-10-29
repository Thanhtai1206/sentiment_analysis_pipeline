"""Fine-tune PhoBERT cho phân loại sắc thái 3 lớp.

Quy trình chống rò rỉ dữ liệu (anti-leakage):
  * train      -> cập nhật trọng số, tính class weights
  * validation -> early stopping (chọn epoch tốt nhất theo Macro-F1) + tìm ngưỡng Trung tính
  * test       -> chỉ đánh giá DUY NHẤT 1 lần ở cuối, không dùng để chọn bất cứ thứ gì

Chạy:  python -m src.train                 (dùng configs/config.yaml)
       python -m src.train --epochs 1      (ghi đè nhanh vài tham số)
"""
from __future__ import annotations

import argparse
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from src.config import load_config, resolve
from src.dataset import build_loader, load_processed
from src.evaluate import apply_threshold, compute_metrics, predict_proba, summarize, tune_threshold
from src.preprocessing import prepare_data
from src.utils import get_device, save_json, set_seed


def compute_class_weights(labels, num_labels: int) -> np.ndarray:
    """w_c = N / (K * n_c), tính từ phân bố nhãn THỰC của tập train."""
    counts = np.bincount(np.asarray(labels), minlength=num_labels).astype(float)
    return counts.sum() / (num_labels * np.maximum(counts, 1))


def train(cfg: dict) -> dict:
    set_seed(cfg["seed"])
    device = get_device()
    tcfg, mcfg = cfg["training"], cfg["model"]
    use_amp = bool(tcfg.get("fp16")) and device.type == "cuda"
    for k in ("metrics_dir", "figures_dir", "predictions_dir"):
        resolve(cfg["results"][k]).mkdir(parents=True, exist_ok=True)
    print(f"Thiết bị: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    # ---------------- dữ liệu
    if not (resolve(cfg["data"]["processed_dir"]) / "train.csv").exists():
        prepare_data(cfg)
    train_df, val_df, test_df = (load_processed(cfg, s) for s in ("train", "validation", "test"))
    tokenizer = AutoTokenizer.from_pretrained(mcfg["pretrained_name"])
    train_loader = build_loader(train_df, tokenizer, cfg, True, tcfg["batch_size"])
    val_loader = build_loader(val_df, tokenizer, cfg, False, tcfg["eval_batch_size"])
    test_loader = build_loader(test_df, tokenizer, cfg, False, tcfg["eval_batch_size"])
    print(f"Train {len(train_df):,} | Val {len(val_df):,} | Test {len(test_df):,}")

    # ---------------- mô hình
    id2label = {i: n for i, n in cfg["labels"].items()}
    model = AutoModelForSequenceClassification.from_pretrained(
        mcfg["pretrained_name"], num_labels=mcfg["num_labels"],
        id2label=id2label, label2id={v: k for k, v in id2label.items()},
    ).to(device)

    weights = None
    if tcfg.get("use_class_weights", True):
        weights = compute_class_weights(train_df["label"], mcfg["num_labels"])
        print(f"Class weights (từ train): {np.round(weights, 4).tolist()}")
        weights = torch.tensor(weights, dtype=torch.float, device=device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)

    no_decay = ("bias", "LayerNorm.weight")
    params = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": tcfg["weight_decay"]},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(params, lr=float(tcfg["learning_rate"]))
    total_steps = len(train_loader) * tcfg["epochs"]
    scheduler = get_linear_schedule_with_warmup(optimizer, int(tcfg["warmup_ratio"] * total_steps), total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # ---------------- vòng huấn luyện + early stopping trên VALIDATION
    out_dir = resolve(mcfg["output_dir"])
    history, best_f1, best_epoch, patience_left = [], -1.0, 0, tcfg["early_stopping_patience"]
    for epoch in range(1, tcfg["epochs"] + 1):
        model.train()
        t0, running = time.time(), 0.0
        for step, batch in enumerate(train_loader, 1):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            labels = batch.pop("labels")
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(**batch).logits
            loss = loss_fn(logits.float(), labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["max_grad_norm"])
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running += loss.item()
            if step % 50 == 0:
                print(f"  epoch {epoch} step {step}/{len(train_loader)} loss={loss.item():.4f}")

        train_loss = running / len(train_loader)
        val_probs, val_y, val_loss = predict_proba(model, val_loader, device, use_amp)
        val_m = compute_metrics(val_y, apply_threshold(val_probs, None), cfg["labels"])
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                        "val_accuracy": val_m["accuracy"], "val_macro_f1": val_m["macro_f1"]})
        print(f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
              f"val_macroF1={val_m['macro_f1']:.4f} ({time.time() - t0:.0f}s)")

        if val_m["macro_f1"] > best_f1:
            best_f1, best_epoch, patience_left = val_m["macro_f1"], epoch, tcfg["early_stopping_patience"]
            model.save_pretrained(out_dir)
            tokenizer.save_pretrained(out_dir)  # lưu kèm tokenizer -> demo chạy offline được
            print(f"  ✓ Lưu checkpoint tốt nhất (epoch {epoch}) -> {out_dir}")
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"  Early stopping: val Macro-F1 không cải thiện {tcfg['early_stopping_patience']} epoch.")
                break

    hist_df = pd.DataFrame(history)
    hist_df.to_csv(resolve(cfg["results"]["metrics_dir"]) / "training_history.csv", index=False)
    plot_history(hist_df, best_epoch, resolve(cfg["results"]["figures_dir"]) / "loss_curves.png")

    # ---------------- nạp lại checkpoint tốt nhất, tìm ngưỡng trên VALIDATION
    model = AutoModelForSequenceClassification.from_pretrained(out_dir).to(device)
    val_probs, val_y, _ = predict_proba(model, val_loader, device, use_amp)
    threshold = None
    if cfg["threshold"]["enabled"]:
        th = cfg["threshold"]
        threshold, f1_t = tune_threshold(val_probs, val_y, th["search_min"], th["search_max"], th["search_step"])
        print(f"Ngưỡng Trung tính tối ưu trên VALIDATION: {threshold} (val macro-F1={f1_t:.4f})")

    save_json({
        "threshold": threshold,
        "max_length": mcfg["max_length"],
        "preprocessing": cfg["preprocessing"],
        "labels": {str(k): v for k, v in cfg["labels"].items()},
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_f1,
        "base_model": mcfg["pretrained_name"],
    }, out_dir / "inference_config.json")

    # ---------------- đánh giá validation + TEST (một lần duy nhất)
    summarize(cfg, val_df, val_probs, val_y, threshold, "validation")
    test_probs, test_y, _ = predict_proba(model, test_loader, device, use_amp)
    return summarize(cfg, test_df, test_probs, test_y, threshold, "test")


def plot_history(hist: pd.DataFrame, best_epoch: int, path) -> None:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.plot(hist["epoch"], hist["train_loss"], "o-", label="Train loss (weighted CE)")
    a1.plot(hist["epoch"], hist["val_loss"], "s-", label="Validation loss (CE)")
    a1.set_xlabel("Epoch"); a1.set_ylabel("Loss"); a1.legend(); a1.set_title("Đường cong loss")
    a2.plot(hist["epoch"], hist["val_macro_f1"], "o-", color="tab:green")
    a2.axvline(best_epoch, ls="--", color="gray", label=f"best epoch = {best_epoch}")
    a2.set_xlabel("Epoch"); a2.set_ylabel("Macro-F1"); a2.legend(); a2.set_title("Validation Macro-F1")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Fine-tune PhoBERT")
    ap.add_argument("--config", default=None)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--lr", type=float)
    a = ap.parse_args()
    cfg = load_config(a.config)
    if a.epochs: cfg["training"]["epochs"] = a.epochs
    if a.batch_size: cfg["training"]["batch_size"] = a.batch_size
    if a.lr: cfg["training"]["learning_rate"] = a.lr
    train(cfg)
