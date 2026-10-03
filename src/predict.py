import os
import sys
import torch
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 1. Cấu hình phần cứng tăng tốc tính toán (Ưu tiên GPU CUDA nếu có)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_CHECKPOINT_PATH = os.path.join(BASE_DIR, "..", "model_checkpoints")

# Khởi tạo biến toàn cục cho Tokenizer và Model để nạp bộ nhớ 1 lần duy nhất
try:
    print("🔄 Đang nạp bộ trọng số PhoBERT Base hoàn toàn từ local...")
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_CHECKPOINT_PATH)
    
    model.to(DEVICE)
    model.eval()  
    print("✅ Nạp mô hình PhoBERT thành công! Hệ thống đã sẵn sàng suy luận.")
except Exception as e:
    print(f"❌ Lỗi nạp mô hình: {e}")
    print("Vui lòng kiểm tra lại các file tokenizer trong thư mục model_checkpoints/")
    sys.exit(1)

def clean_text(text: str) -> str:
    """Hàm tiền xử lý chuẩn hóa teencode học thuật đồng bộ hệ thống."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'\b(gv|thầy|cô)\b', 'giảng viên', text)
    text = re.sub(r'\b(sv)\b', 'sinh viên', text)
    text = re.sub(r'\b(slide|slđ)\b', 'bài giảng', text)
    text = re.sub(r'\b(đc)\b', 'được', text)
    text = re.sub(r'\b(khg|ko|kh|k)\b', 'không', text)
    text = re.sub(r'\b(ok|ổn)\b', 'tạm ổn', text)
    text = re.sub(r'[^\w\s,.\-+]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def predict_sentiment(text: str) -> dict:
    """
    Hàm xử lý khép kín: Nhận chuỗi văn bản thô -> Làm sạch -> Tokenize BPE -> Forward GPU -> Trả xác suất 3 lớp.
    Áp dụng thuật toán nới ngưỡng động 0.24 của Tuần 5.
    """
    if not text or not text.strip():
        return {"label": "Không có dữ liệu", "label_id": -1, "confidence": 0.0, "probabilities": {}}
        
    # Ánh xạ nhãn số sang chuỗi văn bản học thuật để hiển thị trực quan
    id2label = {0: "Tiêu cực", 1: "Trung tính", 2: "Tích cực"}
    
    # Thực thi bước tiền xử lý ngôn ngữ học trước khi trích xuất vector
    cleaned_text = clean_text(text)
    
    with torch.no_grad():
        # Tiến hành Tokenize, cắt chuỗi và chèn padding đồng bộ Max Length = 256
        inputs = tokenizer(
            cleaned_text,
            padding='max_length',
            truncation=True,
            max_length=256,
            return_tensors="pt"
        )
        
        # Đẩy dữ liệu ma trận Tensor lên thiết bị tính toán (GPU hoặc CPU)
        input_ids = inputs['input_ids'].to(DEVICE)
        attention_mask = inputs['attention_mask'].to(DEVICE)
        
        # Mô hình tiến hành forward pass trích xuất Logits 3 chiều
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        
        # Tính toán phân phối xác suất phần trăm bằng hàm Softmax mềm dẻo
        probabilities = torch.softmax(logits, dim=1).squeeze(0).cpu().tolist()
        

        if probabilities[1] >= 0.24:
            prediction_id = 1
        else:
  
            prediction_id = 0 if probabilities[0] > probabilities[2] else 2
            
    return {
        "label_id": prediction_id,
        "label": id2label[prediction_id],
        "confidence": probabilities[prediction_id],
        "probabilities": {
            "Tiêu cực (Nhãn 0)": probabilities[0],
            "Trung tính (Nhãn 1)": probabilities[1],
            "Tích cực (Nhãn 2)": probabilities[2]
        },
        "n_tokens": int(inputs['input_ids'].shape[1])
    }


if __name__ == "__main__":

    test_sentence = "tạm thời chưa có ."
    result = predict_sentiment(test_sentence)
    print(f"\n📝 Câu thực nghiệm: '{test_sentence}'")
    print(f"🎯 Nhãn dự đoán tối hậu: {result['label']} (Độ tin cậy: {result['confidence']:.2%})")
    print(f"📊 Chi tiết phân phối xác suất mềm: {result['probabilities']}")