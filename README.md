# 🚀 AI-Driven Citizen Grievance Analysis System

An end-to-end NLP system that automatically analyzes citizen complaints, classifies them into relevant government departments, detects sentiment, and assigns a priority score for faster resolution.

---

## 📌 Problem Statement

Government complaint systems are often:

* Manual and slow
* Poorly organized across departments
* Lacking prioritization for urgent issues

Officials must manually read thousands of complaints, causing delays and inefficiency.

---

## 🎯 Objective

To build an AI-powered system that:

* Automatically routes complaints to the correct department
* Detects sentiment (urgency level)
* Assigns a priority score for faster action

---

## 🧠 Features

✅ Department Classification (Random Forest)
✅ Sentiment Analysis (RoBERTa / BERT)
✅ Priority Scoring System
✅ REST API using FastAPI
✅ Interactive Swagger UI

---

## 🏗️ Project Architecture

```text
User Input (Complaint Text)
        ↓
Text Preprocessing
        ↓
TF-IDF Vectorization
        ↓
Random Forest → Department
        ↓
BERT / RoBERTa → Sentiment
        ↓
Priority Score Calculation
        ↓
API Response (JSON)
```

---

# 📓 Notebooks

## citizen_grievance_nlp_FINAL.ipynb

Single notebook covering all 3 weeks of model development.

---

## 🟢 Week 1 — Data Collection, Preprocessing & EDA

* **Dataset:** NYC 311 Service Requests (~300,698 complaints)

* **Preprocessing:**

  * Lowercase
  * URL removal
  * Special character removal
  * Stopword removal
  * spaCy Lemmatization

* **Output:**

  * `cleaned_text` column
  * ~63% token reduction

* **EDA:**

  * Word Cloud
  * Unigram / Bigram / Trigram analysis

---

## 🔵 Week 2 — Department Classification

* **Vectorization:** TF-IDF (unigrams + bigrams)

* **Models:**

  * Random Forest (200 trees)
  * Logistic Regression

* **Validation:** 5-Fold Stratified Cross-Validation

* **Results:**

  * Accuracy: **97.09%**
  * Macro F1 Score: **87.50%**

* **Saved Files:**

  * `rf_department_model.pkl`
  * `tfidf_vectorizer.pkl`

---

## 🟣 Week 3 — Sentiment Analysis & Urgency Scoring

* **Model:** RoBERTa (`roberta-base`) fine-tuned on ~6,000 samples

* **Classes:**

  * Critical/Urgent
  * Negative
  * Neutral

* **Priority Formula:**

```text
priority_score = base_score × confidence
```

| Sentiment       | Score |
| --------------- | ----- |
| Critical/Urgent | 4     |
| Negative        | 3     |
| Neutral         | 2     |
| Positive        | 1     |

* **Saved Files:**

  * `roberta_sentiment_model/`
  * `sentiment_label_encoder.pkl`

---

# 🔴 Week 4 — FastAPI Deployment

REST API that serves trained models.

---

## 📁 Folder Structure

```text
grievance_api/
│
├── app/
│   ├── models/
│   ├── schemas/
│   ├── utils/
│   └── main.py
│
├── artifacts/
│   ├── rf_department_model.pkl
│   ├── tfidf_vectorizer.pkl
│   ├── sentiment_label_encoder.pkl
│   └── roberta_sentiment_model/
│
├── notebooks/
│   └── citizen_grievance_nlp_FINAL.ipynb
│
├── requirements.txt
└── README.md
```

---

## ⚙️ How to Run(Google colab)
1. Open in Google Colab
2. Enable GPU — Runtime → Change runtime type → T4 GPU
3. Upload `NYC311data.csv` when prompted
4. Run all cells top to bottom

---

## ⚙️ Setup & Run

```bash
# 1. Create virtual environment
python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate  # Mac/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start API
uvicorn app.main:app --reload
```

---

## 🌐 API Endpoints

| Method | Endpoint   | Description                               |
| ------ | ---------- | ----------------------------------------- |
| GET    | `/`        | Welcome message                           |
| GET    | `/health`  | Health check                              |
| POST   | `/predict` | Predict department + sentiment + priority |
| GET    | `/docs`    | Swagger UI                                |

---

## 📥 Sample Request

```json
POST /predict
{
  "text": "Large pothole on main road near Ward 5. Very dangerous."
}
```

---

## 📤 Sample Response

```json
{
  "complaint_text": "Large pothole on main road near Ward 5. Very dangerous.",
  "department": "Roads & Transport",
  "sentiment": "Negative",
  "priority_score": 2.85,
  "confidence": 0.95,
  "priority_label": "HIGH"
}
```

---

## 📊 Priority Score Guide

| Score     | Label    | Action             |
| --------- | -------- | ------------------ |
| 3.5 – 4.0 | CRITICAL | Immediate dispatch |
| 2.5 – 3.4 | HIGH     | Same day response  |
| 1.5 – 2.4 | MEDIUM   | Within 48 hours    |
| 0.0 – 1.4 | LOW      | Routine queue      |

---

## 🧪 Test the API

```bash
python test_api.py
```

OR open:

```text
http://127.0.0.1:8000/docs
```

---

## ⚠️ Limitations

* Sentiment labels are rule-based (not human annotated)
* TF-IDF struggles with unseen words
* Model depends on dataset quality

---

## 🚀 Future Improvements

* Improve sentiment dataset with manual labels
* Use advanced transformers (DistilBERT, RoBERTa-large)
* Deploy on cloud (AWS / Render)
* Add frontend dashboard

---


