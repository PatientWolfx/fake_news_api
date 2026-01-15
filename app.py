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

    logits = outputs.logits
    probs = torch.nn.functional.softmax(logits, dim=1)

    confidence, predicted_class = torch.max(probs, dim=1)
    label_map = {0: "FAKE", 1: "REAL"}

    data = {}

    return {
        "prediction": label_map[predicted_class.item()], 
        "confidence": round(confidence.item(), 4),
    }