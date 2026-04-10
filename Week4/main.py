"""
Week 4: FastAPI — AI-Driven Citizen Grievance & Sentiment Analysis System
"""

import warnings
warnings.filterwarnings("ignore")

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import torch
import numpy as np
import re
import os
from transformers import RobertaTokenizer, RobertaForSequenceClassification

# ── App ───────────────────────────────────────────────────
app = FastAPI(
    title="Citizen Grievance NLP API",
    description="Classifies civic complaints into departments and scores urgency.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Load models ───────────────────────────────────────────
print("Loading models...")

dept_model = joblib.load("Models/best_model.pkl")
tfidf      = joblib.load("Models/tfidf_vectorizer.pkl")
print("  ✓ Department classifier loaded")

SENT_MODEL_PATH = "Models/roberta_sentiment_model"
USE_ROBERTA     = False

try:
    sent_tokenizer = RobertaTokenizer.from_pretrained(SENT_MODEL_PATH)
    sent_model     = RobertaForSequenceClassification.from_pretrained(SENT_MODEL_PATH)
    sent_model     = sent_model.to(device)
    sent_model.eval()
    USE_ROBERTA = True
    print("  ✓ RoBERTa sentiment model loaded from local files")
except Exception as e:
    print(f"  ⚠ RoBERTa not available: {e}")
    print("  → Using rule-based sentiment (reliable fallback)")

label_encoder = joblib.load("Models/sentiment_label_encoder.pkl")
print(f"  ✓ Label encoder: {list(label_encoder.classes_)}")
print(f"  ✓ Sentiment mode: {'RoBERTa' if USE_ROBERTA else 'Rule-based'}")
print("\nAPI ready!")

# ── Priority mapping ──────────────────────────────────────
BASE_SCORES = {
    "Critical/Urgent": 4.0,
    "Negative":        3.0,
    "Neutral":         2.0,
    "Positive":        1.0,
}

# ── Rule-based sentiment keywords ────────────────────────
SENTIMENT_RULES = {
    "Critical/Urgent": [
        "emergency", "danger", "dangerous", "hazard", "urgent", "immediately",
        "critical", "life", "safety", "fire", "flood", "collapse", "severe",
        "gas leak", "electric shock", "accident", "violent", "weapon",
        "injury", "dead", "death", "immediately", "serious", "fatal",
        "lives at risk", "people will die", "someone will die", "sos",
        "help immediately", "act now", "respond now", "immediate action"
    ],
    "Negative": [
        "broken", "damaged", "dirty", "smell", "odor", "loud", "noise",
        "illegal", "blocked", "abandoned", "missing", "not working",
        "unsanitary", "overflowing", "leaking", "crack", "mold",
        "rodent", "rat", "violation", "failed", "unresolved", "problem",
        "issue", "complaint", "no action", "not fixed", "ignored",
        "disappointing", "frustrated", "terrible", "horrible", "awful",
        "pothole", "garbage", "trash", "overflowing", "stink", "pest",
        "broken", "damaged", "neglected", "disgusting", "unacceptable"
    ],
    "Positive": [
        "resolved", "fixed", "clean", "working", "repaired", "improved",
        "excellent", "good", "great", "thank", "appreciate", "satisfied",
        "done", "completed", "happy", "pleased", "wonderful"
    ],
}

def rule_based_sentiment(text: str):
    """Reliable keyword-based sentiment — used when RoBERTa is unavailable."""
    t = text.lower()
    for sentiment, keywords in SENTIMENT_RULES.items():
        if any(kw in t for kw in keywords):
            confidence = 0.88 if sentiment == "Critical/Urgent" else 0.82
            return sentiment, confidence
    return "Neutral", 0.75

# ── Schemas ───────────────────────────────────────────────
class ComplaintRequest(BaseModel):
    text: str
    model_config = {
        "json_schema_extra": {
            "example": {
                "text": "Large pothole on main road near Ward 5. Very dangerous for vehicles."
            }
        }
    }

class PredictionResponse(BaseModel):
    complaint_text: str
    department:     str
    sentiment:      str
    priority_score: float
    confidence:     float
    priority_label: str

# ── Helpers ───────────────────────────────────────────────
def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def predict_department(text: str) -> str:
    vec = tfidf.transform([clean_text(text)])
    return dept_model.predict(vec)[0]

def predict_sentiment(text: str):
    if USE_ROBERTA:
        encoding = sent_tokenizer(
            str(text), max_length=128, padding="max_length",
            truncation=True, return_tensors="pt"
        )
        with torch.no_grad():
            out   = sent_model(
                input_ids=encoding["input_ids"].to(device),
                attention_mask=encoding["attention_mask"].to(device)
            )
            probs = torch.softmax(out.logits, dim=1).cpu().numpy()[0]
        pred_idx = int(np.argmax(probs))
        return label_encoder.classes_[pred_idx], round(float(probs[pred_idx]), 4)
    else:
        return rule_based_sentiment(text)

def get_priority_label(score: float) -> str:
    if score >= 3.5:   return "CRITICAL"
    elif score >= 2.5: return "HIGH"
    elif score >= 1.5: return "MEDIUM"
    else:              return "LOW"

# ── Routes ────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "message": "Citizen Grievance NLP API is running",
        "docs":    "Visit /docs for Swagger UI",
        "predict": 'POST /predict with JSON: {"text": "your complaint"}'
    }

@app.get("/health")
def health_check():
    return {
        "status":        "healthy",
        "models_loaded": True,
        "sentiment_mode": "RoBERTa" if USE_ROBERTA else "Rule-based",
        "device":        str(device)
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(request: ComplaintRequest):
    """
    Accepts raw citizen complaint text. Returns:
    - department: which civic department should handle it
    - sentiment: Critical/Urgent, Negative, or Neutral
    - priority_score: 0.0 to 4.0 (higher = more urgent)
    - priority_label: CRITICAL / HIGH / MEDIUM / LOW
    - confidence: model confidence (0.0 to 1.0)
    """
    text = request.text.strip()

    if not text:
        raise HTTPException(status_code=400, detail="Complaint text cannot be empty.")
    if len(text) < 5:
        raise HTTPException(status_code=400, detail="Complaint text is too short.")

    try:
        department            = predict_department(text)
        sentiment, confidence = predict_sentiment(text)
        priority_score        = round(BASE_SCORES.get(sentiment, 2.0) * confidence, 3)
        priority_label        = get_priority_label(priority_score)

        return PredictionResponse(
            complaint_text = text,
            department     = department,
            sentiment      = sentiment,
            priority_score = priority_score,
            confidence     = confidence,
            priority_label = priority_label
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
