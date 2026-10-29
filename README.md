# Phân tích sắc thái phản hồi sinh viên (UIT-VSFC) — PhoBERT

Fine-tune **PhoBERT** (`vinai/phobert-base`) phân loại phản hồi sinh viên tiếng Việt thành 3 lớp
`0 = Tiêu cực`, `1 = Trung tính`, `2 = Tích cực`, so sánh với baseline TF-IDF, kèm phân tích lỗi và demo Gradio.

| | |
|---|---|
| Dữ liệu | UIT-VSFC – train 11 426 / validation 1 583 / test 3 166 câu |
| Thách thức | mất cân bằng nặng: lớp Trung tính chỉ ~4% |
| Metric chính | **Macro-F1** (accuracy dễ gây ảo giác vì lớp 0 và 2 chiếm ~96%) |
| Xử lý mất cân bằng | class-weighted Cross-Entropy + ngưỡng ưu tiên lớp Trung tính |
| Framework | PyTorch + HuggingFace Transformers, scikit-learn, Gradio |

---

## Cấu trúc thư mục

```text
sentiment_analysis_pipeline/
├── configs/
│   └── config.yaml              # MỌI tham số (đường dẫn, siêu tham số, ngưỡng...) ở 1 chỗ
├── data/
│   ├── raw/                     # dữ liệu gốc UIT-VSFC – KHÔNG chỉnh sửa
│   ├── external/                # new_review*.csv – dữ liệu ngoài, chưa dùng trong huấn luyện
│   ├── processed/               # tự sinh bởi `python -m src.preprocessing` (không commit)
│   └── sample/sample_input.csv  # vài câu mẫu để thử predict
├── src/                         # toàn bộ logic – notebook chỉ gọi lại
│   ├── config.py                # đọc config, chuẩn hóa đường dẫn
│   ├── preprocessing.py         # clean_text / tách từ / sinh data/processed
│   ├── dataset.py               # SentimentDataset + DataLoader (dynamic padding)
│   ├── baseline.py              # TF-IDF + LogisticRegression / LinearSVC
│   ├── train.py                 # fine-tune PhoBERT, early stopping, tìm ngưỡng trên validation
│   ├── evaluate.py              # metric, ngưỡng, ma trận nhầm lẫn, file dự đoán
│   ├── predict.py               # suy luận 1 câu / cả file CSV (CLI + class SentimentPredictor)
│   └── utils.py
├── notebooks/
│   ├── 01_eda_data_pipeline.ipynb
│   ├── 02_train_phobert.ipynb
│   └── 03_error_analysis.ipynb
├── demo/
│   ├── app.py                   # giao diện Gradio (python demo/app.py)
│   └── demo.ipynb
├── models/                      # checkpoint (không commit, xem models/README.md)
├── results/
│   ├── figures/  metrics/  predictions/  error_analysis/
│   └── archive_week5/           # kết quả cũ trước khi refactor (giữ để đối chiếu báo cáo)
├── reports/                     # báo cáo .docx, slide, mã nguồn LaTeX
├── tests/                       # pytest
├── requirements.txt
└── .gitignore
```

## Cài đặt

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows   (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
```

## Chạy toàn bộ pipeline

```bash
python -m src.preprocessing      # 1. làm sạch dữ liệu -> data/processed/
python -m src.baseline           # 2. baseline TF-IDF (CPU, vài giây)
python -m src.train              # 3. fine-tune PhoBERT (nên dùng GPU: Colab/Kaggle T4)
python -m src.evaluate --split test   # (tùy chọn) đánh giá lại checkpoint đã lưu
python -m src.predict "giảng viên nhiệt tình, dễ hiểu"
python -m src.predict --file data/sample/sample_input.csv
python demo/app.py               # 4. demo Gradio tại http://127.0.0.1:7860
pytest                           # kiểm thử nhanh;  pytest -m slow  -> chạy thử cả pipeline với model mini
```

Hoặc chạy lần lượt 3 notebook trong `notebooks/` (cho kết quả giống hệt vì cùng gọi `src/`).

## Quy trình đánh giá (chống rò rỉ dữ liệu)

| Tập | Được dùng để |
|---|---|
| **train** | cập nhật trọng số; tính class weights từ phân bố nhãn thực |
| **validation** | early stopping (giữ epoch có Macro-F1 cao nhất) và **tìm ngưỡng lớp Trung tính** |
| **test** | chỉ đánh giá **một lần** ở cuối – không dùng để chọn bất kỳ siêu tham số nào |

Luật ngưỡng: nếu `P(Trung tính) ≥ t` → nhãn 1, ngược lại chọn lớp lớn hơn giữa 0 và 2.
Ngưỡng `t` được lưu vào `models/phobert-sentiment/inference_config.json` và tự động dùng lại trong `predict.py` / demo.

## Kết quả

Baseline (đã chạy lại bằng `python -m src.baseline`, seed 42):

| Mô hình | Val Macro-F1 | Test Accuracy | Test Macro-F1 | Test F1 Trung tính |
|---|---:|---:|---:|---:|
| TF-IDF + LogisticRegression | 0.675 | 0.890 | 0.652 | 0.132 |
| TF-IDF + LogisticRegression (balanced) | 0.757 | 0.865 | 0.714 | 0.344 |
| TF-IDF + LinearSVC (balanced) | 0.763 | 0.893 | 0.724 | 0.338 |
| PhoBERT (argmax) | *chạy `src.train`* | | | |
| PhoBERT + ngưỡng (tìm trên validation) | *chạy `src.train`* | | | |

Sau khi huấn luyện, số liệu PhoBERT nằm ở `results/metrics/phobert_test_summary.json`,
bảng tổng hợp ở `results/metrics/final_comparison.csv` (notebook 02).

## Ghi chú

- Câu nhập vào demo đi qua **cùng** hàm `preprocess()` như lúc huấn luyện.
- `preprocessing.word_segmenter: underthesea` trong config bật tách từ (PhoBERT được pre-train trên văn bản đã tách từ) – cần `pip install underthesea` và **huấn luyện lại**.
- Dữ liệu `data/external/new_review*.csv` chưa rõ nguồn gốc nhãn và phân bố khác UIT-VSFC nên không trộn vào train/test.
