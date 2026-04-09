"""
Week 4: FastAPI — AI-Driven Citizen Grievance & Sentiment Analysis System
"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import torch
import numpy as np
import re
import os
import uvicorn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from transformers import AutoTokenizer, RobertaForSequenceClassification

# Resolve paths relative to this file so the app works from any working directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── App setup ─────────────────────────────────────────────
app = FastAPI(
    title="Citizen Grievance NLP API",
    description="Classifies civic complaints into departments and scores urgency.",
    version="1.0.0"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Load models ───────────────────────────────────────────
print("Loading models...")

# Load department pipeline — try as pipeline first, then as separate model+vectorizer
try:
    dept_pipeline = joblib.load(os.path.join(BASE_DIR, "Models", "best_model.pkl"))
    # Test if it has named_steps (sklearn Pipeline)
    if hasattr(dept_pipeline, 'named_steps'):
        USE_PIPELINE = True
        print("  ✓ Department pipeline loaded (Pipeline mode)")
    else:
        # It's a standalone classifier — need separate tfidf
        USE_PIPELINE = False
        dept_model   = dept_pipeline
        tfidf        = joblib.load(os.path.join(BASE_DIR, "Models", "tfidf_vectorizer.pkl"))
        print("  ✓ Department classifier + TF-IDF loaded (standalone mode)")
except Exception as e:
    print(f"  ✗ Failed to load department model: {e}")
    raise

# Load sentiment model
SENTIMENT_MODEL_PATH = os.path.join(BASE_DIR, "Models", "roberta_sentiment_model")
try:
    model_file = os.path.join(SENTIMENT_MODEL_PATH, "model.safetensors")
    if not os.path.exists(model_file):
        raise FileNotFoundError("model.safetensors missing")
    # Tokenizer is standard roberta-base (same for all RoBERTa variants);
    # load it from HuggingFace to avoid corrupted/incompatible tokenizer.json.
    sent_tokenizer = AutoTokenizer.from_pretrained("roberta-base", use_fast=True)
    # Load fine-tuned model weights from local path
    sent_model     = RobertaForSequenceClassification.from_pretrained(SENTIMENT_MODEL_PATH)
    print("  ✓ Sentiment model loaded (fine-tuned weights, roberta-base tokenizer)")
except Exception as e:
    print(f"  ⚠ Local model issue: {e}")
    print("  → Loading roberta-base from HuggingFace (untrained fallback)...")
    sent_tokenizer = AutoTokenizer.from_pretrained("roberta-base", use_fast=True)
    sent_model     = RobertaForSequenceClassification.from_pretrained(
                         "roberta-base", num_labels=3)
    print("  ✓ roberta-base loaded as fallback")

sent_model    = sent_model.to(device)
sent_model.eval()
label_encoder = joblib.load(os.path.join(BASE_DIR, "Models", "sentiment_label_encoder.pkl"))
print(f"  ✓ Label encoder loaded | Device: {device}")
print("\nAPI ready!")

# ── Priority mapping ──────────────────────────────────────
BASE_SCORES = {
    "Critical/Urgent": 4.0,
    "Negative":        3.0,
    "Neutral":         2.0,
    "Positive":        1.0,
}

# ── Schemas ───────────────────────────────────────────────
class ComplaintRequest(BaseModel):
    text: str
    model_config = {
        "json_schema_extra": {
            "example": {"text": "Large pothole on main road near Ward 5. Very dangerous."}
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
    """Clean text for department classifier."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def predict_department(text: str) -> str:
    """Predict civic department."""
    cleaned = clean_text(text)
    if USE_PIPELINE:
        # Pipeline handles vectorization internally
        return dept_pipeline.predict([cleaned])[0]
    else:
        # Standalone: vectorize first, then predict
        vec = tfidf.transform([cleaned])
        return dept_model.predict(vec)[0]

def predict_sentiment(text: str):
    """Predict sentiment + confidence using RoBERTa."""
    encoding = sent_tokenizer(
        str(text),
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    with torch.no_grad():
        out   = sent_model(
            input_ids=encoding["input_ids"].to(device),
            attention_mask=encoding["attention_mask"].to(device)
        )
        probs = torch.softmax(out.logits, dim=1).cpu().numpy()[0]
    pred_idx = int(np.argmax(probs))
    return label_encoder.classes_[pred_idx], round(float(probs[pred_idx]), 4)

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
        "device":        str(device)
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(request: ComplaintRequest):
    """
    Accepts raw citizen complaint text. Returns:
    - department: which civic department handles this
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


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
