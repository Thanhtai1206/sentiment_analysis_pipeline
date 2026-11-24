# 🎓 ĐỒ ÁN MÁY HỌC: PHÂN TÍCH CẢM XÚC PHẢN HỒI SINH VIÊN (UIT-VSFC)
> **Vietnamese Students' Feedback Corpus Sentiment Analysis — From Classical ML to PhoBERT**

---

## 📌 Giới Thiệu Đề Tài
Dự án giải quyết bài toán **Phân loại cảm xúc văn bản tiếng Việt (Sentiment Analysis)** trên bộ dữ liệu phản hồi của sinh viên Việt Nam (**UIT-VSFC**), phân loại các câu đánh giá thành 3 lớp cảm xúc:
* **0 — Tiêu cực (Negative)**
* **1 — Trung tính (Neutral)** *(Lớp thiểu số, chỉ chiếm ~4.3% dữ liệu)*
* **2 — Tích cực (Positive)**

Đồ án đi theo quy trình thực nghiệm tăng dần: **EDA & Baseline ML (TF-IDF + LinearSVC)** $\rightarrow$ **Xử lý Mất cân bằng dữ liệu & Tách từ tiếng Việt** $\rightarrow$ **Thử nghiệm ViSoBERT** $\rightarrow$ **Fine-tuning PhoBERT (Weighted Loss)** $\rightarrow$ **Tinh chỉnh ngưỡng quyết định trên tập Validation (Zero Data Leakage)** và **Nghiệm thu độc lập trên tập Test**.

---

## 📁 Cấu Trúc Kho Mã Nguồn (Repository Structure)

```text
ML_Mr Phuoc/
│
├── 📂 02_Qua_Trinh/                          # Minh chứng tiến trình nghiên cứu qua 4 giai đoạn
│   ├── 1_Notebooks/ & 1_Result/              # Giai đoạn 1: EDA & Baseline ML (TF-IDF + LinearSVC)
│   ├── 2_Notebooks/ & 2_Result/              # Giai đoạn 2: Imbalanced Data (Class Weight/ROS/SMOTE) & Tách từ (Underthesea/VnCoreNLP)
│   ├── 3_Notebooks/ & 3_Result/              # Giai đoạn 3: Thử nghiệm mô hình ViSoBERT
│   └── 4_Notebooks/ & 4_Result/              # Giai đoạn 4: Xây dựng DataLoader & Fine-tuning PhoBERT
│
├── 📂 03_Source_Code_Final/                  # Bộ mã nguồn chính thức hoàn chỉnh
│   ├── notebooks/
│   │   ├── 01_final_data_pipeline.ipynb      # Pipeline tiền xử lý đồng bộ & DataLoader PyTorch
│   │   ├── 02_final_model.ipynb              # Huấn luyện, Threshold Tuning (Validation) & Nghiệm thu (Test)
│   │   └── 03_error_analysis_demo.ipynb      # Phân tích mẫu lỗi thực nghiệm
│   ├── src/
│   │   └── predict.py                        # Backend suy luận độc lập (ngưỡng tối ưu 0.49)
│   └── demo/
│       └── demo.ipynb                        # Giao diện tương tác trực quan Gradio UI
│
├── 📂 04_Results/                            # Kết quả nghiệm thu chính thức trên tập Test
│   ├── final_metrics.csv                     # Bảng chỉ số Precision, Recall, F1-Score chính thức
│   ├── phobert_confusion_matrix.png          # Ma trận nhầm lẫn tối hậu (Ngưỡng = 0.49)
│   ├── loss_curves.png                       # Biểu đồ đường cong Loss
│   └── error_analysis.csv                    # Danh sách các câu dự đoán sai kèm xác suất 3 lớp
│
├── 📂 05_Data/                               # Bộ dữ liệu UIT-VSFC
│   ├── raw/                                  # train.csv, validation.csv, test.csv
│   ├── processed/                            # train_cleaned.csv
│   └── sample/                               # sample_input.csv
│
├── 📂 06_Model_Checkpoints/                  # Cấu hình & Trọng số mô hình PhoBERT
│   ├── config.json
│   └── model.safetensors                     # (Lưu trữ trên Google Drive do giới hạn 100MB của GitHub)
│
├── 📄 .gitignore                             # Bỏ qua file trọng số nặng & cache
├── 📄 requirements.txt                       # Danh sách thư viện Python phụ thuộc
└── 📄 README.md                              # Tài liệu hướng dẫn đồ án
```

---

## 📊 Kết Quả Thực Nghiệm & Đối Sánh Mô Hình (Tập Test — 3.166 mẫu)

### 1. So sánh tiến hóa qua các giai đoạn
| Giai đoạn / Mô hình | Kỹ thuật nổi bật | Accuracy | Đúng lớp Trung tính (167 mẫu) |
| :--- | :--- | :---: | :---: |
| **Baseline LinearSVC thuần** | TF-IDF `ngram(1,2)` | 89.10% | 28 / 167 (16.8%) |
| **LinearSVC + `balanced`** | Cost-sensitive Learning | 88.60% | 59 / 167 (35.3%) |
| **LinearSVC + SMOTE** | Over-sampling tổng hợp | 87.90% | 73 / 167 (43.7%) |
| **LogReg + Tách từ (Underthesea / VnCoreNLP)** | Word Segmentation tiếng Việt | ~85.70% | 83–84 / 167 (~50%) |
| **ViSoBERT Oversampled** | Pre-trained Transformer | 91.47% | 69 / 167 (41.3%) |
| **PhoBERT Final (Đề xuất)** | **Weighted Loss + Validation Threshold (0.49)** | **92.26%** | **89 / 167 (53.3%)** |

### 2. Chỉ số chi tiết mô hình PhoBERT Final (Zero Data Leakage)
| Lớp Cảm Xúc | Precision | Recall | F1-Score | Số lượng mẫu (Support) |
| :--- | :---: | :---: | :---: | :---: |
| **Tiêu cực (Nhãn 0)** | 93.40% | 95.46% | **94.42%** | 1,409 |
| **Trung tính (Nhãn 1)** | 54.27% | 53.29% | **53.78%** | 167 |
| **Tích cực (Nhãn 2)** | 95.20% | 93.52% | **94.35%** | 1,590 |
| **Accuracy** | | | **92.26%** | 3,166 |
| **Macro Average** | 80.96% | 80.76% | **80.85%** | 3,166 |
| **Weighted Average** | 92.24% | 92.26% | **92.24%** | 3,166 |

> 💡 **Phương pháp luận chuẩn ML:** Ngưỡng quyết định tối ưu (`0.49`) cho lớp Trung tính được tìm hoàn toàn trên **tập Validation**, sau đó khóa ngưỡng để nghiệm thu độc lập một lần duy nhất trên **tập Test** (đảm bảo Zero Data Leakage).

---

## 🛠️ Cài Đặt & Hướng Dẫn Sử Dụng

### 1. Cài đặt môi trường (Python 3.10+)
```bash
pip install -r requirements.txt
```

### 2. Tải Trọng Số Mô Hình (`model.safetensors`)
Do file trọng số nặng ~540MB vượt giới hạn của GitHub, vui lòng tải `model.safetensors` từ Google Drive và đặt vào thư mục `06_Model_Checkpoints/` (chung thư mục với `config.json`).

### 3. Chạy Demo Tương Tác (Gradio UI)
Mở và chạy [`03_Source_Code_Final/demo/demo.ipynb`](file:///d:/ML_Mr%20Phuoc/03_Source_Code_Final/demo/demo.ipynb), hoặc chạy trực tiếp qua CLI:
```bash
python 03_Source_Code_Final/src/predict.py
```

### 4. Huấn Luyện & Đánh Giá Lại
Mở [`03_Source_Code_Final/notebooks/02_final_model.ipynb`](file:///d:/ML_Mr%20Phuoc/03_Source_Code_Final/notebooks/02_final_model.ipynb):
* `SKIP_TRAINING = True`: Nạp checkpoint có sẵn để tinh chỉnh ngưỡng trên Validation và xuất kết quả Test (~1–2 phút).
* `SKIP_TRAINING = False`: Huấn luyện mới hoàn toàn 5 epochs từ `vinai/phobert-base` có theo dõi Validation Loss.
