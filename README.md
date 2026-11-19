# 🎓 ĐỒ ÁN MÁY HỌC: PHÂN TÍCH CẢM XÚC PHẢN HỒI SINH VIÊN (UIT-VSFC)
> **Vietnamese Students' Feedback Corpus Sentiment Analysis using PhoBERT & Machine Learning**

---

## 📌 Giới Thiệu Đề Tài
Dự án tập trung giải quyết bài toán **Phân loại cảm xúc văn bản (Sentiment Analysis)** dựa trên tập dữ liệu phản hồi của sinh viên Việt Nam (**UIT-VSFC**). Mục tiêu là phân loại các câu đánh giá môn học/giảng viên thành 3 sắc thái cảm xúc:
* **0 - Tiêu cực (Negative)**
* **1 - Trung tính (Neutral)** *(Lớp thiểu số, phân bố lệch dữ liệu)*
* **2 - Tích cực (Positive)**

Đồ án được triển khai theo phương pháp luận nghiên cứu khoa học chuẩn mực: Đi từ **Mô hình học máy truyền thống (Baseline ML)** $\rightarrow$ Xử lý **Mất cân bằng lớp (Imbalanced Data)** $\rightarrow$ Huấn luyện mô hình **Học sâu Tiền huấn luyện (PhoBERT)** $\rightarrow$ Tinh chỉnh ngưỡng quyết định trên tập **Validation** và nghiệm thu độc lập trên tập **Test**.

---

## 📁 Cấu Trúc Kho Mã Nguồn (Repository Structure)

```text
ML_Mr Phuoc/
│
├── 📂 01_Bao_Cao_Final/                      # Hồ sơ tổng kết nghiệm thu đồ án
│   ├── Nhom4_Tuan6_4(Revised).pdf            # Báo cáo học thuật toàn diện (Final Report)
│   └── ThuyetTrinh_Nhom4.pptx                # Slide thuyết trình bảo vệ đồ án
│
├── 📂 02_Qua_Trinh/                          # Minh chứng tiến độ nghiên cứu thực nghiệm từng tuần
│   ├── Tuan_1/                               # Khảo sát đề tài & phân tích tập dữ liệu
│   ├── Tuan_2/                               # EDA & Xây dựng mô hình Baseline ML truyền thống
│   ├── Tuan_3/                               # Thực nghiệm xử lý mất cân bằng lớp & tách từ tiếng Việt
│   ├── Tuan_4/                               # Xây dựng DataLoader & Fine-tuning PhoBERT
│   └── Tuan_5/                               # Tối ưu hóa mô hình, kiểm soát Overfitting & Slide tuần 5
│
├── 📂 03_Source_Code_Final/                  # Bộ mã nguồn chính thức hoàn chỉnh
│   ├── notebooks/
│   │   ├── 01_final_data_pipeline.ipynb      # Pipeline tiền xử lý & DataLoader PyTorch
│   │   ├── 02_final_model.ipynb              # Quá trình huấn luyện, Tuning ngưỡng & Nghiệm thu
│   │   └── 03_error_analysis_demo.ipynb      # Phân tích mẫu lỗi thực nghiệm trên tập Test
│   ├── src/
│   │   └── predict.py                        # Backend suy luận độc lập & Ngưỡng động
│   └── demo/
│       └── demo.ipynb                        # Ứng dụng giao diện đồ họa tương tác Gradio UI
│
├── 📂 04_Results/                            # Báo cáo định lượng & Đồ thị đối sánh
│   ├── final_metrics.csv                     # Bảng chỉ số hiệu năng phân loại chính thức (F1, P, R)
│   ├── phobert_confusion_matrix.png          # Ma trận nhầm lẫn tối hậu trên tập Test
│   ├── overall_comparison_chart.png          # Biểu đồ so sánh hiệu năng Baseline ML vs PhoBERT
│   ├── loss_curves.png                       # Đường cong hội tụ Train Loss vs Validation Loss
│   └── error_analysis.csv                    # Danh sách các câu dự đoán sai thực tế
│
├── 📂 05_Data/                               # Dữ liệu nghiên cứu (UIT-VSFC)
│   ├── raw/                                  # Dữ liệu gốc: train.csv, validation.csv, test.csv
│   ├── processed/                            # Dữ liệu đã làm sạch: train_cleaned.csv
│   └── sample/                               # Dữ liệu mẫu test nhanh: sample_input.csv
│
├── 📂 06_Model_Checkpoints/                  # Checkpoint cấu hình & Trọng số mô hình
│   ├── config.json                           # Cấu hình kiến trúc PhoBERT Sequence Classification
│   └── model.safetensors                     # (Lưu trữ trên Cloud do giới hạn dung lượng GitHub)
│
├── 📄 .gitignore                             # Cấu hình bỏ qua các file checkpoint nặng và cache
├── 📄 requirements.txt                       # Danh mục các thư viện Python phụ thuộc
└── 📄 README.md                              # Tài liệu hướng dẫn sử dụng đồ án
```

---

## 📊 Kết Quả Thực Nghiệm Trên Tập Test (Official Metrics)

Mô hình **PhoBERT** sau khi tối ưu hóa ngưỡng quyết định trên tập **Validation** đạt kết quả vượt trội trên tập **Test độc lập (3.166 mẫu)**:

| Lớp Cảm Xúc | Precision | Recall | F1-Score | Số lượng mẫu (Support) |
| :--- | :---: | :---: | :---: | :---: |
| **Tiêu cực (Nhãn 0)** | 93.40% | 95.46% | **94.42%** | 1,409 |
| **Trung tính (Nhãn 1)** | 54.27% | 53.29% | **53.78%** | 167 |
| **Tích cực (Nhãn 2)** | 95.20% | 93.52% | **94.35%** | 1,590 |
| **Độ chính xác (Accuracy)** | | | **92.26%** | 3,166 |
| **Macro Average** | 80.96% | 80.76% | **80.85%** | 3,166 |
| **Weighted Average** | 92.24% | 92.26% | **92.24%** | 3,166 |

> 💡 **Điểm sáng phương pháp luận:** Lớp Trung tính (Nhãn 1) bị mất cân bằng trầm trọng (chỉ chiếm ~5% tập dữ liệu). Nhờ áp dụng **Class Weights** trong `CrossEntropyLoss` kết hợp kỹ thuật **Threshold Tuning** (ngưỡng tối ưu ~0.24 tìm trên tập Validation), F1-Score của lớp Trung tính được cải thiện rõ rệt từ mức kém (< 40%) lên **53.78%**.

---

## 🛠️ Cài Đặt & Môi Trường

Khuyến nghị sử dụng **Python 3.10+**.

```bash
# 1. Clone repository về máy tính
git clone https://github.com/Thanhtai1206/ML_Mr-Phuoc.git
cd ML_Mr-Phuoc

# 2. Cài đặt các thư viện cần thiết
pip install -r requirements.txt
```

---

## 🚀 Hướng Dẫn Sử Dụng

### 1. Tải Trọng Số Mô Hình (Model Checkpoint)
Do file `model.safetensors` nặng **~540MB** vượt quá giới hạn 100MB của GitHub, trọng số được lưu trữ an toàn tại:
* 🔗 **Link tải Checkpoint (Google Drive):** `https://drive.google.com/file/d/1DKqcA_6CW4lXydoyfsIAhxXQaOS1JJto/view?usp=sharing`
* Sau khi tải về, đặt file `model.safetensors` vào thư mục `06_Model_Checkpoints/` (nằm chung thư mục với `config.json`).

### 2. Chạy Giao Diện Tương Tác Demo (Gradio UI)
Mở và chạy file [03_Source_Code_Final/demo/demo.ipynb](file:///d:/ML_Mr%20Phuoc/03_Source_Code_Final/demo/demo.ipynb) hoặc khởi chạy qua script:
```bash
python 03_Source_Code_Final/src/predict.py
```
Giao diện Gradio sẽ khởi chạy tại cổng local: `http://127.0.0.1:7860`. Bạn có thể gõ câu nhận xét bất kỳ để xem dự đoán xác suất 3 lớp trong thời gian thực.

### 3. Huấn Luyện & Đánh Giá Lại Mô Hình
Toàn bộ quy trình huấn luyện và đánh giá được đóng gói hoàn chỉnh trong [03_Source_Code_Final/notebooks/02_final_model.ipynb](file:///d:/ML_Mr%20Phuoc/03_Source_Code_Final/notebooks/02_final_model.ipynb):
* `SKIP_TRAINING = True`: Nạp checkpoint có sẵn để tinh chỉnh ngưỡng và xuất bảng điểm nghiệm thu (mất ~1-2 phút).
* `SKIP_TRAINING = False`: Huấn luyện lại 5 epochs từ đầu có theo dõi Validation Loss.

---

## 👥 Nhóm Tác Giả
* **Môn học:** Máy học (Machine Learning)
* **Giảng viên hướng dẫn:** ThS. [Tên Giảng Viên]
* **Nhóm thực hiện:** Nhóm 04
