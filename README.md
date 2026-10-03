# Phân Tích Sắc Thái Phản Hồi Sinh Viên (UIT-VSFC) — PhoBERT
---

## 📁 Cấu Trúc Cây Thư Mục Kho Mã Nguồn (Repository Structure)

```text
Nhom04_Tuan5/
├── Nhom4_Tuan5_4.docx                    # Văn bản báo cáo tiến độ học thuật toàn diện
├── notebooks/
│   ├── 01_final_data_pipeline.ipynb      # Hạ tầng Pipeline nạp và xử lý dữ liệu PyTorch
│   ├── 02_final_model.ipynb              # Nhật ký huấn luyện, Fine-tuning & Early Stopping
│   └── 03_error_analysis_demo.ipynb      # Phân tích mẫu lỗi thực nghiệm trên mô hình
├── src/
│   └── predict.py                        # Backend suy luận tập trung & Can thiệp ngưỡng động 0.24
├── demo/
│   └── demo.ipynb                        # Ứng dụng giao diện đồ họa tương tác Gradio UI
├── model_checkpoints/
│   ├── config.json                       # Tệp cấu hình tham số kiến trúc Transformer
│   └── model.safetensors                 # Trọng số tối ưu đóng gói an toàn sau huấn luyện
├── results/
│   ├── final_metrics.csv                 # Bảng kết quả định lượng đối sánh 3 mức
│   ├── loss_curves.png                   # Biểu đồ đường cong hội tụ kiểm soát Overfitting
│   ├── phobert_confusion_matrix.png      # Đồ thị ma trận nhầm lẫn tối hậu PhoBERT
│   ├── overall_comparison_chart.png      # Biểu đồ so sánh hiệu năng tổng quan đồ án
│   └── error_analysis.csv                # Danh sách các mẫu lỗi phục vụ tra cứu thực nghiệm
└── requirements.txt                      # Danh mục thư viện và phiên bản cài đặt cố định
```
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

```
---

## ⚙️ Cấu Hình

Các thông số có thể chủ động điều chỉnh tại **Cell 1**:

```python
MODEL_PATH = "../model_checkpoints"  # Đường dẫn đến checkpoint fine-tuned
MAX_LENGTH = 256                     # Độ dài token tối đa

Để tạo đường dẫn public tạm thời nhằm chia sẻ demo từ xa, chỉnh sửa dòng lệnh cuối tại Cell 4:
```
```python
demo.launch(share=True)  # Tạo link public qua Gradio
```
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
```
## 📌 Ghi Chú An Toàn Hệ Thống

* **Tokenizer luôn tải trực tuyến:** Bộ mã hóa (Tokenizer) luôn được tải trực tiếp từ HuggingFace Hub (`vinai/phobert-base`) vì không được lưu trữ kèm trong thư mục checkpoint.
* **Cấu hình phần cứng:** Mô hình tự động chuyển cấu hình chạy trên **CPU** nếu môi trường không tích hợp GPU; tốc độ phản hồi sẽ chậm hơn đối với các chuỗi văn bản dài.
* **Xử lý cảnh báo:** Cảnh báo `unauthenticated requests` từ HuggingFace có thể hoàn toàn bỏ qua hoặc khắc phục triệt để bằng cách cấu hình biến môi trường `HF_TOKEN`.