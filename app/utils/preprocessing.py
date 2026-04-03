# app/utils/preprocessing.py
import re
import spacy

# Load spaCy model (shared across requests)
_nlp = None

def get_spacy_model():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    return _nlp

from spacy.lang.en.stop_words import STOP_WORDS


def clean_text(text: str) -> str:
    """
    Clean raw complaint text for model inference.
    Matches the preprocessing pipeline from Week 1.
    """
    if not isinstance(text, str):
        return ""

    # Step 1: Lowercase
    text = text.lower()

    # Step 2: Remove URLs and emails
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\S+@\S+", "", text)

    # Step 3: Remove special characters and digits
    text = re.sub(r"[^a-z\s]", " ", text)

    # Step 4: Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def clean_text_with_lemmatization(text: str) -> str:
    """
    Full preprocessing with lemmatization (for BERT input).
    """
    text = clean_text(text)
    if not text:
        return ""

    nlp = get_spacy_model()
    doc = nlp(text)

    tokens = [
        token.lemma_
        for token in doc
        if not token.is_stop
        and not token.is_punct
        and len(token.text) > 2
    ]

    return " ".join(tokens)