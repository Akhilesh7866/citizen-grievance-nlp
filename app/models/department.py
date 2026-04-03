# app/models/department.py

import joblib
import numpy as np
from typing import Tuple
import logging
import os

logger = logging.getLogger(__name__)


class DepartmentClassifier:
    """Random Forest model for department classification."""

    def __init__(self, model_path: str, vectorizer_path: str):
        self.model = None
        self.vectorizer = None
        self.classes = None
        self._load(model_path, vectorizer_path)

    def _load(self, model_path: str, vectorizer_path: str):
        """Load model and vectorizer from disk."""
        try:
            print("\n🔍 DEBUG START ------------------------")
            print("Model path:", model_path)
            print("Vectorizer path:", vectorizer_path)
            print("File exists:", os.path.exists(vectorizer_path))

            # Load model
            self.model = joblib.load(model_path)

            # Load vectorizer
            self.vectorizer = joblib.load(vectorizer_path)

            # 🔥 IMPORTANT DEBUG
            print("Vectorizer type:", type(self.vectorizer))
            print("Is TF-IDF fitted (idf_ exists):", hasattr(self.vectorizer, "idf_"))

            if not hasattr(self.vectorizer, "idf_"):
                raise ValueError("❌ TF-IDF vectorizer is NOT fitted!")

            self.classes = self.model.classes_

            print("Classes:", self.classes)
            print("🔍 DEBUG END --------------------------\n")

            logger.info(f"Department model loaded. Classes: {list(self.classes)}")

        except Exception as e:
            logger.error(f"Failed to load department model: {e}")
            raise

    def predict(self, text: str) -> Tuple[str, float]:
        """
        Predict department for a single complaint.
        """
        if not text or not text.strip():
            return "Unknown", 0.0

        try:
            # Transform text using TF-IDF
            text_vectorized = self.vectorizer.transform([text])

            # Prediction
            prediction = self.model.predict(text_vectorized)[0]

            # Probability
            probas = self.model.predict_proba(text_vectorized)[0]
            confidence = float(np.max(probas))

            return str(prediction), round(confidence, 4)

        except Exception as e:
            print("❌ Prediction error:", str(e))
            raise

    def is_loaded(self) -> bool:
        return self.model is not None and self.vectorizer is not None