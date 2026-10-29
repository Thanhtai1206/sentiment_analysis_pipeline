# models/

Thư mục chứa checkpoint (không đưa lên git vì nặng ~500 MB).

Sau khi chạy `python -m src.train`, thư mục `phobert-sentiment/` gồm:

| File | Ý nghĩa |
|---|---|
| `config.json`, `model.safetensors` | kiến trúc + trọng số PhoBERT đã fine-tune (epoch tốt nhất trên validation) |
| `tokenizer_config.json`, `vocab.txt`, `bpe.codes`, ... | tokenizer lưu kèm → demo chạy offline |
| `inference_config.json` | ngưỡng lớp Trung tính (tìm trên validation), max_length, cấu hình tiền xử lý |

Dùng checkpoint cũ (`model_checkpoints/`): chép `config.json` + `model.safetensors` vào `models/phobert-sentiment/`.
Khi đó tokenizer tải từ HuggingFace và ngưỡng lấy `threshold.fallback_value` trong `configs/config.yaml`.
