"""Chạy thử TOÀN BỘ pipeline train -> evaluate -> predict với 1 model RoBERTa siêu nhỏ
khởi tạo ngẫu nhiên (không cần internet, không cần GPU, ~1 phút).
Mục đích: bắt lỗi code, KHÔNG phải để đo độ chính xác.   pytest -m slow
"""
import copy

import pandas as pd
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")


def _tiny_model(path, texts):
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast, RobertaConfig, RobertaForSequenceClassification

    tok = Tokenizer(models.WordLevel(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.train_from_iterator(texts, trainers.WordLevelTrainer(
        special_tokens=["<s>", "<pad>", "</s>", "<unk>"]))
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, bos_token="<s>", eos_token="</s>",
                                   pad_token="<pad>", unk_token="<unk>")
    fast.save_pretrained(path)
    cfg = RobertaConfig(vocab_size=fast.vocab_size, hidden_size=32, num_hidden_layers=1,
                        num_attention_heads=2, intermediate_size=64, max_position_embeddings=140,
                        pad_token_id=1, num_labels=3)
    RobertaForSequenceClassification(cfg).save_pretrained(path)


@pytest.mark.slow
def test_full_pipeline(tmp_path):
    from src.config import load_config
    from src.preprocessing import load_split, preprocess

    cfg = copy.deepcopy(load_config())
    proc = tmp_path / "processed"; proc.mkdir()
    for split, n in (("train", 300), ("validation", 100), ("test", 100)):
        raw = load_split(cfg, split)
        df = pd.concat([g.head(n // 3) for _, g in raw.groupby("label")])
        df["text_clean"] = df["text"].map(preprocess)
        df.to_csv(proc / f"{split}.csv", index=False)
    tiny = tmp_path / "tiny"
    _tiny_model(tiny, pd.read_csv(proc / "train.csv")["text_clean"].tolist())

    cfg["data"]["processed_dir"] = str(proc)
    cfg["model"]["pretrained_name"] = str(tiny)
    cfg["model"]["output_dir"] = str(tmp_path / "out_model")
    cfg["training"].update(epochs=2, batch_size=16, eval_batch_size=32, fp16=False)
    cfg["results"] = {k: str(tmp_path / "results" / k) for k in
                      ("dir", "figures_dir", "metrics_dir", "predictions_dir")}

    from src.train import train
    summary = train(cfg)
    assert "argmax" in summary
    assert (tmp_path / "out_model" / "inference_config.json").exists()
    assert (tmp_path / "results" / "predictions_dir" / "phobert_test_predictions.csv").exists()

    from src.predict import SentimentPredictor
    r = SentimentPredictor(tmp_path / "out_model").predict("giảng viên nhiệt tình")
    assert r["label_id"] in (0, 1, 2)
    assert abs(sum(r["probabilities"].values()) - 1) < 1e-4
