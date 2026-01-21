import torch
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

# -----------------------------
# App initialization
# -----------------------------
app = FastAPI(title="Fake News Detection API")
templates = Jinja2Templates(directory="templates")

# -----------------------------
# Model loading
# -----------------------------
MODEL_PATH = "model"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
model.to(device)
model.eval()

# -----------------------------
# UI Route (HTML)
# -----------------------------
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )

# -----------------------------
# Request schema
# -----------------------------
class NewsRequest(BaseModel):
    text: str

# -----------------------------
# Prediction endpoint
# -----------------------------
@app.post("/predict")
def predict_news(news: NewsRequest, threshold: float = 0.9):
    """
    Predict REAL or FAKE using confidence thresholding.
    Label mapping:
        0 -> FAKE
        1 -> REAL
    """

    # Tokenize
    inputs = tokenizer(
        news.text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Inference
    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1)[0]

    prob_fake = probs[0].item()
    prob_real = probs[1].item()

    # Decision logic
    if prob_fake >= threshold:
        label = "FAKE"
        confidence = prob_fake
    elif prob_real >= threshold:
        label = "REAL"
        confidence = prob_real
    else:
        # Forced fallback
        if prob_real >= prob_fake:
            label = "REAL"
            confidence = prob_real
        else:
            label = "FAKE"
            confidence = prob_fake

    return {
        "prediction": label,
        "confidence": round(confidence, 4),
        "prob_fake": round(prob_fake, 4),
        "prob_real": round(prob_real, 4)
    }
