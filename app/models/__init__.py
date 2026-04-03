# app/models/__init__.py
from .department import DepartmentClassifier
from .sentiment import SentimentAnalyzer

__all__ = ["DepartmentClassifier", "SentimentAnalyzer"]