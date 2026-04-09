# Week 4 — FastAPI Deployment

REST API that serves the trained NLP models from Weeks 2 and 3.

## Folder Structure
```
Week 4/
├── main.py            ← FastAPI application
├── requirements.txt   ← Dependencies
├── test_api.py        ← Test script
├── README.md          ← This file
└── Models/
    ├── best_model.pkl
    ├── tfidf_vectorizer.pkl
    ├── sentiment_label_encoder.pkl
    └── roberta_sentiment_model/
```

## Setup & Run

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start the API
uvicorn main:app --reload
```

## Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Welcome message |
| GET | `/health` | Health check |
| POST | `/predict` | Predict department + sentiment + priority |
| GET | `/docs` | Interactive Swagger UI |

## Sample Request & Response

**Request:**
```json
POST /predict
{
  "text": "Large pothole on main road near Ward 5. Very dangerous."
}
```

**Response:**
```json
{
  "complaint_text": "Large pothole on main road near Ward 5. Very dangerous.",
  "department":     "Roads & Transport",
  "sentiment":      "Negative",
  "priority_score": 2.85,
  "confidence":     0.95,
  "priority_label": "HIGH"
}
```

## Priority Score Guide

| Score | Label | Action |
|-------|-------|--------|
| 3.5 – 4.0 | CRITICAL | Immediate dispatch |
| 2.5 – 3.4 | HIGH | Same day response |
| 1.5 – 2.4 | MEDIUM | Within 48 hours |
| 0.0 – 1.4 | LOW | Routine queue |

## Test the API

```bash
python test_api.py
```

Or open `http://127.0.0.1:8000/docs` in your browser for the Swagger UI.
