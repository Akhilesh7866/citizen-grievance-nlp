# app/models/sentiment.py

import joblib
import torch
import numpy as np
from typing import Tuple
from transformers import AutoTokenizer, BertForSequenceClassification
import logging

logger = logging.getLogger(__name__)

# Priority scoring configuration
BASE_SCORES = {
    'Critical/Urgent': 4.0,
    'Negative': 3.0,
    'Neutral': 2.0,
    'Positive': 1.0,
}

SEVERITY_MAP = {
    'Critical/Urgent': 'Critical',
    'Negative': 'High',
    'Neutral': 'Medium',
    'Positive': 'Low',
}

RECOMMENDED_ACTIONS = {
    'Critical/Urgent': 'Immediate dispatch required. Escalate to emergency services.',
    'Negative': 'Assign to relevant department within 24 hours.',
    'Neutral': 'Add to standard queue for processing.',
    'Positive': 'Log and close if resolved. Send satisfaction survey.',
}


class SentimentAnalyzer:
    """BERT model for sentiment analysis and priority scoring."""

    def __init__(
        self,
        model_path: str,
        label_encoder_path: str,
        device: str = None
    ):
        self.model = None
        self.tokenizer = None
        self.label_encoder = None
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._load(model_path, label_encoder_path)

    def _load(self, model_path: str, label_encoder_path: str):
        """Load model, tokenizer, and label encoder from disk."""
        try:
            # ✅ FIX: Use AutoTokenizer (important fix)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)

            # Load BERT model
            self.model = BertForSequenceClassification.from_pretrained(model_path)
            self.model.to(self.device)
            self.model.eval()

            # Load label encoder
            self.label_encoder = joblib.load(label_encoder_path)

            logger.info(f"Sentiment model loaded on {self.device}")
            logger.info(f"Classes: {list(self.label_encoder.classes_)}")

        except Exception as e:
            logger.error(f"Failed to load sentiment model: {e}")
            raise

    def predict(self, text: str) -> Tuple[str, float]:
        """
        Predict sentiment for a single complaint.
        """
        if not text or not text.strip():
            return "Neutral", 0.0

        # Tokenize input
        encoding = self.tokenizer(
            text,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Move to device
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)

        # Predict
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]

        pred_idx = int(np.argmax(probs))
        label = str(self.label_encoder.classes_[pred_idx])
        confidence = float(probs[pred_idx])

        return label, round(confidence, 4)

    def get_priority_score(self, text: str) -> dict:
        """
        Calculate priority score for a complaint.
        """
        sentiment, confidence = self.predict(text)

        base_score = BASE_SCORES.get(sentiment, 2.0)
        priority_score = round(base_score * confidence, 3)
        severity = SEVERITY_MAP.get(sentiment, 'Medium')
        action = RECOMMENDED_ACTIONS.get(sentiment, 'Process normally.')

        return {
            "priority_score": priority_score,
            "severity": severity,
            "recommended_action": action,
            "sentiment": sentiment,
            "confidence": confidence
        }

    def is_loaded(self) -> bool:
        return self.model is not None and self.tokenizer is not None