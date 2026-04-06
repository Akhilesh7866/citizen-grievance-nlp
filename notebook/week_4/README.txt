# AI-Driven Citizen Grievance System - Week 4

## 📌 Module: Model Serialization & Output Design

## 🚀 Features
- Serialized ML model (Random Forest)
- Serialized TF-IDF vectorizer
- Prediction pipeline
- Structured JSON output (output design)
- Confidence score + priority level
- FastAPI integration

## ⚙️ Setup

1. Install dependencies:
pip install -r requirements.txt

2. Run API:
uvicorn app:app --reload

3. Open:
http://127.0.0.1:8000

## 📡 API Endpoint

POST /predict

Example:
{
  "text": "Water leakage problem in my area"
}

## 📂 Folder Structure

week4_model_serialization_output/
│
├── save_model.py
├── predict.py
├── app.py
├── requirements.txt
├── README.md
├── model.pkl
└── tfidf_vectorizer.pkl
