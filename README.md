# Đồ Án Machine Learning: Phân Tích Sắc Thái Phản Hồi Sinh Viên (UIT-VSFC)
---

## 📁 Cấu Trúc Cây Thư Mục Kho Mã Nguồn (Repository Structure)

`	ext
ML_Mr Phuoc/
├── 01_Bao_Cao_Final/                      # Báo cáo tổng kết PDF và Slide thuyết trình bảo vệ
│   ├── Nhom4_Tuan6_4(Revised).pdf
│   └── ThuyetTrinh_Nhom4.pptx
├── 02_Minh_Chung_Tien_Do/                 # Toàn bộ minh chứng tiến độ học thuật & code qua từng tuần
│   ├── Tuan_1/ (Báo cáo đề tài & khảo sát)
│   ├── Tuan_2/ (Báo cáo + Notebooks EDA & Baseline ML truyền thống)
│   ├── Tuan_3/ (Báo cáo + Notebooks xử lý mất cân bằng & tách từ tiếng Việt)
│   ├── Tuan_4/ (Báo cáo + Notebooks huấn luyện Deep Learning PhoBERT)
│   └── Tuan_5/ (Báo cáo + Slide tiến độ hoàn thiện)
├── 03_Source_Code/                        # Toàn bộ mã nguồn chính thức của đồ án
│   ├── notebooks/                         # Các Jupyter Notebook pipeline chính
│   │   ├── 01_final_data_pipeline.ipynb
│   │   ├── 02_final_model.ipynb
│   │   └── 03_error_analysis_demo.ipynb
│   ├── src/
│   │   └── predict.py                     # Module suy luận (Inference)
│   └── demo/
│       └── demo.ipynb                     # Giao diện tương tác Gradio UI
├── 04_Results/                            # Đồ thị, ma trận nhầm lẫn và bảng điểm metric
├── 05_Data/                               # Dữ liệu phân chia (raw, processed, sample)
├── 06_Model_Checkpoints/                  # Checkpoint cấu hình và trọng số mô hình PhoBERT
├── requirements.txt                       # Danh mục thư viện và phiên bản cài đặt
└── README.md                              # Tài liệu hướng dẫn sử dụng đồ án
`

## Phân Tích Sắc Thái Tiếng Việt — PhoBERT

Dự án fine-tune mô hình **PhoBERT** cho bài toán phân loại sắc thái (Sentiment Analysis) trên văn bản tiếng Việt, với giao diện demo tích hợp **Gradio** chạy trực tiếp trong Jupyter Notebook.

---

## 📊 Tổng Quan

| Thông tin | Chi tiết |
| :--- | :--- |
| **Mô hình gốc** | `vinai/phobert-base` |
| **Loại task** | Sequence Classification (3 lớp) |
| **Ngôn ngữ** | Tiếng Việt |
| **Framework** | PyTorch + HuggingFace Transformers |
| **Giao diện** | Gradio (nhúng trong Notebook) |

### Nhãn phân loại

| Nhãn | Ý nghĩa |
| :---: | :--- |
| `0` | Tiêu cực |
| `1` | Trung tính |
| `2` | Tích cực |

---

## 🛠️ Yêu Cầu Cài Đặt

Cài đặt các thư viện cần thiết để chạy hệ thống:

```bash
pip install torch transformers gradio

## 🚀 Hướng Dẫn Sử Dụng

### 1. Chuẩn bị model checkpoint
Đặt thư mục model đã fine-tune vào `./model_checkpoints/`. Thư mục này cần chứa ít nhất các tệp tin sau:
- `config.json`
- `model.safetensors`

### 2. Chạy Notebook
Mở tệp tin `demo.ipynb` và thực thi tuần tự từng phân khối lệnh (cell):

| Cell | Mô tả |
| :--- | :--- |
| **Cell 1** | Import thư viện & cấu hình (device, label map, max length) |
| **Cell 2** | Nạp Tokenizer từ HuggingFace + Model từ checkpoint |
| **Cell 3** | Định nghĩa hàm `predict_sentiment()` |
| **Cell 4** | Khởi chạy giao diện Gradio trong Notebook |

### 3. Sử dụng giao diện
Sau khi chạy **Cell 4**, giao diện Gradio sẽ xuất hiện ngay trong notebook hoặc truy cập tại địa chỉ local: `http://127.0.0.1:7860`.
- Nhập câu phản hồi tiếng Việt vào ô nhập liệu.
- Nhấn **Dự đoán**.
- Giao diện sẽ hiển thị kết quả gồm: nhãn dự đoán, độ tin cậy, và biểu đồ xác suất của cả 3 lớp.

---

## ⚙️ Cấu Hình

Các thông số có thể chủ động điều chỉnh tại **Cell 1**:

```python
MODEL_PATH = "../model_checkpoints"  # Đường dẫn đến checkpoint fine-tuned
MAX_LENGTH = 256                     # Độ dài token tối đa

Để tạo đường dẫn public tạm thời nhằm chia sẻ demo từ xa, chỉnh sửa dòng lệnh cuối tại Cell 4:

```python
demo.launch(share=True)  # Tạo link public qua Gradio

## 📤 Đầu Ra Của `predict_sentiment()`

Hàm xử lý trả về cấu trúc dữ liệu dạng Dictionary đồng bộ dải xác suất phục vụ hiển thị:

```json
{
    "predicted_label": 2,
    "label_text":      "Tích cực (2)",
    "confidence":      "97.3%",
    "prob_0_negative": "1.2%",
    "prob_1_neutral":  "1.5%",
    "prob_2_positive": "97.3%",
    "n_tokens":        256
}

## 📌 Ghi Chú An Toàn Hệ Thống

* **Tokenizer luôn tải trực tuyến:** Bộ mã hóa (Tokenizer) luôn được tải trực tiếp từ HuggingFace Hub (`vinai/phobert-base`) vì không được lưu trữ kèm trong thư mục checkpoint.
* **Cấu hình phần cứng:** Mô hình tự động chuyển cấu hình chạy trên **CPU** nếu môi trường không tích hợp GPU; tốc độ phản hồi sẽ chậm hơn đối với các chuỗi văn bản dài.
* **Xử lý cảnh báo:** Cảnh báo `unauthenticated requests` từ HuggingFace có thể hoàn toàn bỏ qua hoặc khắc phục triệt để bằng cách cấu hình biến môi trường `HF_TOKEN`.