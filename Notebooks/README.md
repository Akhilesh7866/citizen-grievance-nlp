# Notebooks

## citizen_grievance_nlp_FINAL.ipynb

Single notebook covering all 3 weeks of model development.

### Week 1 — Data Collection, Preprocessing & EDA
- **Dataset:** NYC 311 Service Requests (300,698 complaints)
- **Preprocessing:** Lowercase → URL removal → Special char removal → Stopword removal → spaCy Lemmatization
- **Output:** `cleaned_text` column, ~63% token reduction
- **EDA:** Word Cloud, Unigram / Bigram / Trigram frequency charts

### Week 2 — Department Classification
- **Vectorization:** TF-IDF (unigrams + bigrams, 107 features)
- **Models:** Random Forest (200 trees) + Logistic Regression
- **Validation:** 5-Fold Stratified Cross-Validation
- **Results:** Accuracy **97.09%** | Macro F1 **87.50%**

### Week 3 — Sentiment Analysis & Urgency Scoring
- **Model:** RoBERTa (`roberta-base`) fine-tuned on 6,000 balanced samples
- **Classes:** Critical/Urgent · Negative · Neutral
- **Priority Score:** `base_score × confidence` → range 0.0 to 4.0
- **Saved files:** `roberta_sentiment_model/` · `sentiment_label_encoder.pkl`

### How to Run
1. Open in Google Colab
2. Enable GPU — Runtime → Change runtime type → T4 GPU
3. Upload `NYC311data.csv` when prompted
4. Run all cells top to bottom
