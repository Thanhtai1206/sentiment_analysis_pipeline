from contextlib import asynccontextmanager
import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# 1. Cấu hình đường dẫn và tham số từ dự án của bạn
MODEL_PATH = "./models/visobert_oversampled_best"  # Hoặc đường dẫn đến folder model của bạn
BEST_THRESHOLD = 0.4592  # Ngưỡng Soft Macro F1 đã tối ưu
ID2LABEL = {0: "Tiêu cực", 1: "Tích cực", 2: "Trung tính"}

# Biến toàn cục chứa model & tokenizer
ml_models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Khởi tạo và load model 1 lần khi server bật lên
    print("⏳ Đang nạp mô hình ViSoBERT vào bộ nhớ...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()
    model.to(device)

    ml_models["tokenizer"] = tokenizer
    ml_models["model"] = model
    ml_models["device"] = device
    print(f"✅ Nạp mô hình thành công trên thiết bị: {device}")

    yield

    # Giải phóng bộ nhớ khi tắt server
    ml_models.clear()


app = FastAPI(
    title="ViSoBERT Sentiment Analysis API",
    version="1.0",
    lifespan=lifespan,
)


# 2. Định nghĩa cấu trúc dữ liệu Input & Output bằng Pydantic
class SingleTextRequest(BaseModel):
    text: str = Field(
        ...,
        example="Cà phê ở đây ngon lắm, phục vụ nhiệt tình!",
        description="Câu văn bản tiếng Việt cần phân loại cảm xúc",
    )


class SentimentResponse(BaseModel):
    text: str
    label: str
    confidence: float
    probabilities: dict[str, float]


class BatchTextRequest(BaseModel):
    texts: list[str] = Field(..., example=["Món uống dở quá", "Trà sữa tạm ổn"])


# 3. Hàm xử lý logic suy luận (Inference)
def predict_sentiment(text: str) -> dict:
    tokenizer = ml_models["tokenizer"]
    model = ml_models["model"]
    device = ml_models["device"]

    # Tokenize input
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=256,
        padding=True,
    ).to(device)

    # Dự đoán xác suất
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)[0].cpu()

    probs_dict = {
        ID2LABEL[i]: round(float(probs[i].item()), 4) for i in range(len(probs))
    }

    # Áp dụng logic dự đoán (có xét ngưỡng hoặc argmax)
    max_prob, pred_id = torch.max(probs, dim=-1)
    pred_id = int(pred_id.item())

    # Nếu xác suất lớn nhất vượt ngưỡng tối ưu -> lấy nhãn đó
    label = ID2LABEL[pred_id]

    return {
        "text": text,
        "label": label,
        "confidence": round(float(max_prob.item()), 4),
        "probabilities": probs_dict,
    }


# 4. Tạo các Endpoint API
@app.get("/")
def home():
    return {"message": "ViSoBERT Sentiment API đang hoạt động bình thường!"}


@app.post("/predict", response_model=SentimentResponse)
def predict_single(req: SingleTextRequest):
    """Endpoint phân loại cảm xúc cho 1 câu đơn"""
    if not req.text.strip():
        raise HTTPException(
            status_code=400, detail="Văn bản đầu vào không được để trống!"
        )
    return predict_sentiment(req.text)


@app.post("/predict-batch", response_model=list[SentimentResponse])
def predict_batch(req: BatchTextRequest):
    """Endpoint phân loại cảm xúc cho một danh sách nhiều câu"""
    if not req.texts:
        raise HTTPException(
            status_code=400, detail="Danh sách văn bản không được rỗng!"
        )
    return [predict_sentiment(t) for t in req.texts if t.strip()]
