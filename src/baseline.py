"""Baseline cổ điển TF-IDF + Logistic Regression / LinearSVC để đối chứng với PhoBERT.

Chạy được trên CPU trong vài giây:   python -m src.baseline
"""
from __future__ import annotations

import argparse

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

from src.config import load_config, resolve
from src.dataset import load_processed
from src.evaluate import compute_metrics, plot_confusion, report_to_frame
from src.preprocessing import prepare_data
from src.utils import set_seed


def build_models(cfg):
    b = cfg["baseline"]
    tfidf = lambda: TfidfVectorizer(ngram_range=tuple(b["ngram_range"]), max_features=b["max_features"],
                                    sublinear_tf=True)
    seed = cfg["seed"]
    return {
        "tfidf_logreg": make_pipeline(tfidf(), LogisticRegression(max_iter=2000, random_state=seed)),
        "tfidf_logreg_balanced": make_pipeline(
            tfidf(), LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)),
        "tfidf_svc_balanced": make_pipeline(tfidf(), LinearSVC(class_weight="balanced", random_state=seed)),
    }


def run(cfg) -> pd.DataFrame:
    set_seed(cfg["seed"])
    if not (resolve(cfg["data"]["processed_dir"]) / "train.csv").exists():
        prepare_data(cfg)
    train_df, val_df, test_df = (load_processed(cfg, s) for s in ("train", "validation", "test"))
    r = cfg["results"]
    for k in ("metrics_dir", "figures_dir", "predictions_dir"):
        resolve(r[k]).mkdir(parents=True, exist_ok=True)

    rows = []
    for name, model in build_models(cfg).items():
        model.fit(train_df["text_clean"], train_df["label"])
        row = {"model": name}
        for split, df in (("validation", val_df), ("test", test_df)):
            pred = model.predict(df["text_clean"])
            m = compute_metrics(df["label"], pred, cfg["labels"])
            row.update({f"{split}_accuracy": m["accuracy"], f"{split}_macro_f1": m["macro_f1"],
                        f"{split}_f1_neutral": m["report"][cfg["labels"][1]]["f1-score"]})
            if split == "test":
                report_to_frame(m).to_csv(resolve(r["metrics_dir"]) / f"baseline_{name}_test.csv",
                                          encoding="utf-8-sig")
                plot_confusion(df["label"], pred, cfg["labels"], f"{name} – test",
                               f"{r['figures_dir']}/baseline_cm_{name}.png")
                out = df[["text", "label"]].assign(pred=pred)
                out.to_csv(resolve(r["predictions_dir"]) / f"baseline_{name}_test_predictions.csv",
                           index=False, encoding="utf-8-sig")
        rows.append(row)
        print(f"{name:24s} val macro-F1={row['validation_macro_f1']:.4f} | test macro-F1={row['test_macro_f1']:.4f}")
    res = pd.DataFrame(rows)
    res.to_csv(resolve(r["metrics_dir"]) / "baseline_summary.csv", index=False)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    run(load_config(ap.parse_args().config))
