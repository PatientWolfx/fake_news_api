import torch
from fastapi import FastAPI, Request 
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

# Initialize app
app = FastAPI(title="Fake News Detection API")

templates = Jinja2Templates(directory="templates")

# Load tokenizer & model
MODEL_PATH = "fake_news_distilbert_model"

tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()

# UI route
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
    
# Request schema
class NewsRequest(BaseModel):
    text: str

# Prediction endpoint
@app.post("/predict")
def predict_news(news: NewsRequest):
    inputs = tokenizer(
        news.text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    )

    with torch.no_grad():
        outputs = model(**inputs)

    # Softmax over logits
    probs = torch.softmax(outputs.logits, dim=1).squeeze(0)

    # IMPORTANT: label mapping
    # 0 -> FAKE
    # 1 -> REAL
    prob_fake = probs[0].item()
    prob_real = probs[1].item()

    # High-confidence decision
    if prob_fake >= threshold:
        label = "FAKE"
        confidence = prob_fake

    elif prob_real >= threshold:
        label = "REAL"
        confidence = prob_real

    else:
        # Forced fallback (still ONLY 2 labels)
        if prob_real >= prob_fake:
            label = "REAL"
            confidence = prob_real
        else:
            label = "FAKE"
            confidence = prob_fake

    return {
        "prediction": label,
        "confidence": round(confidence, 4)
    }