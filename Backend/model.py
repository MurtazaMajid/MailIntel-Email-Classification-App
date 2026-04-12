"""
model.py — Inference on pre-trained email classification models.

Project: Email Classification Pipeline — Murtaza Majid
Dataset: 5,000 personal Gmail emails
Classes: Spam | Important | Work | Promotion | Personal

Models
──────
  "svm"         → svm_model.joblib        (Linear SVM, 79.5% accuracy)
                  uses vectorizer.pkl     (TF-IDF, max_features=5000)

  "naive_bayes" → naive_bayes_model.joblib (Multinomial NB, 72.6% accuracy)
                  uses vectorizer.pkl     (same TF-IDF vectorizer)

  "lstm"        → lstm_model.keras        (LSTM, 81.0% accuracy — best model)
                  uses tokenizer.pkl      (Keras Tokenizer, vocab=5000, oov='<OOV>')
                  sequence padded to maxlen=100, padding='post', truncating='post'

Default model: svm (fastest, no TensorFlow overhead, good accuracy)
"""

from __future__ import annotations

import os
import pickle
import logging
import numpy as np
from dataclasses import dataclass
from functools import lru_cache

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Config — must match training settings exactly
# ─────────────────────────────────────────────

MODEL_DIR   = os.getenv("MODEL_DIR", "Models")
LSTM_MAXLEN = int(os.getenv("LSTM_MAX_LEN", "100"))
VOCAB_SIZE  = 5000

CLASSES = ["Spam", "Important", "Work", "Promotion", "Personal"]

AVAILABLE_MODELS = ["svm", "naive_bayes", "lstm"]

MODEL_DISPLAY_NAMES = {
    "svm":         "SVM — Linear Kernel (79.5% accuracy) ★ Default",
    "naive_bayes": "Naive Bayes (72.6% accuracy)",
    "lstm":        "LSTM — Deep Learning (81.0% accuracy) — Best",
}

MODEL_DESCRIPTIONS = {
    "svm":         "Support Vector Classifier with TF-IDF features. Fast and reliable.",
    "naive_bayes": "Multinomial Naive Bayes with TF-IDF features. Fastest, good baseline.",
    "lstm":        "LSTM neural network with learned embeddings. Highest accuracy overall.",
}

# ─────────────────────────────────────────────
# File paths
# ─────────────────────────────────────────────

def _path(filename: str) -> str:
    return os.path.join(MODEL_DIR, filename)

PATHS = {
    "svm":         _path("svm_model.joblib"),
    "naive_bayes": _path("naive_bayes_model.joblib"),
    "lstm":        _path("lstm_model.keras"),
    "vectorizer":  _path("vectorizer.pkl"),
    "tokenizer":   _path("tokenizer.pkl"),
    "token_auth":  _path("token.pkl"),
}


# ─────────────────────────────────────────────
# Result type
# ─────────────────────────────────────────────

@dataclass
class ModelResult:
    predicted_class: str
    probability: float
    model_used: str
    all_scores: dict


# ─────────────────────────────────────────────
# Lazy model loading
# ─────────────────────────────────────────────

@lru_cache(maxsize=1)
def _load_vectorizer():
    logger.info("Loading vectorizer from %s", PATHS["vectorizer"])
    with open(PATHS["vectorizer"], "rb") as f:
        return pickle.load(f)


@lru_cache(maxsize=1)
def _load_tokenizer():
    logger.info("Loading tokenizer from %s", PATHS["tokenizer"])
    with open(PATHS["tokenizer"], "rb") as f:
        return pickle.load(f)


@lru_cache(maxsize=1)
def _load_svm():
    import joblib
    logger.info("Loading SVM from %s", PATHS["svm"])
    return joblib.load(PATHS["svm"])


@lru_cache(maxsize=1)
def _load_naive_bayes():
    import joblib
    logger.info("Loading Naive Bayes from %s", PATHS["naive_bayes"])
    return joblib.load(PATHS["naive_bayes"])


@lru_cache(maxsize=1)
def _load_lstm():
    from tensorflow import keras
    logger.info("Loading LSTM from %s", PATHS["lstm"])
    return keras.models.load_model(PATHS["lstm"])


# ─────────────────────────────────────────────
# Text prep
# ─────────────────────────────────────────────

def _get_combined_text(email: dict) -> str:
    subject = email.get("subject") or ""
    snippet = email.get("snippet") or ""
    body    = email.get("body")    or ""
    return f"{subject} {snippet} {body}".strip()


def _scores_dict(probas: np.ndarray, classes: list[str]) -> dict:
    return {cls: round(float(p), 4) for cls, p in zip(classes, probas)}


# ─────────────────────────────────────────────
# Model runners
# ─────────────────────────────────────────────

def _run_svm(email: dict) -> ModelResult:
    text       = _get_combined_text(email)
    vectorizer = _load_vectorizer()
    model      = _load_svm()

    features        = vectorizer.transform([text])
    predicted_class = str(model.predict(features)[0])
    classes         = list(model.classes_) if hasattr(model, "classes_") else CLASSES

    if hasattr(model, "predict_proba"):
        probas      = model.predict_proba(features)[0]
        scores      = _scores_dict(probas, classes)
        probability = round(float(np.max(probas)), 4)
    else:
        scores      = {c: (1.0 if c == predicted_class else 0.0) for c in classes}
        probability = 1.0

    return ModelResult(
        predicted_class=predicted_class,
        probability=probability,
        model_used="svm",
        all_scores=scores,
    )


def _run_naive_bayes(email: dict) -> ModelResult:
    text       = _get_combined_text(email)
    vectorizer = _load_vectorizer()
    model      = _load_naive_bayes()

    features        = vectorizer.transform([text])
    predicted_class = str(model.predict(features)[0])
    probas          = model.predict_proba(features)[0]
    classes         = list(model.classes_) if hasattr(model, "classes_") else CLASSES
    scores          = _scores_dict(probas, classes)

    return ModelResult(
        predicted_class=predicted_class,
        probability=round(float(np.max(probas)), 4),
        model_used="naive_bayes",
        all_scores=scores,
    )


def _run_lstm(email: dict) -> ModelResult:
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    text      = _get_combined_text(email)
    tokenizer = _load_tokenizer()
    model     = _load_lstm()

    sequence = tokenizer.texts_to_sequences([text])
    padded   = pad_sequences(
        sequence,
        maxlen=LSTM_MAXLEN,
        padding="post",
        truncating="post",
    )

    raw             = model.predict(padded, verbose=0)[0]
    predicted_idx   = int(np.argmax(raw))
    predicted_class = CLASSES[predicted_idx]
    probability     = round(float(np.max(raw)), 4)
    scores          = _scores_dict(raw, CLASSES)

    return ModelResult(
        predicted_class=predicted_class,
        probability=probability,
        model_used="lstm",
        all_scores=scores,
    )


# ─────────────────────────────────────────────
# Public interface — default model is SVM
# ─────────────────────────────────────────────

def classify_email(email: dict, model: str = "svm") -> ModelResult:
    """
    Classify a single email using the specified pre-trained model.
    Default: svm — fast, no TensorFlow overhead, solid accuracy.
    """
    if model not in AVAILABLE_MODELS:
        raise ValueError(f"Unknown model '{model}'. Choose from: {AVAILABLE_MODELS}")

    logger.info("Classifying email id=%s with model=%s", email.get("id"), model)

    if model == "svm":
        return _run_svm(email)
    elif model == "naive_bayes":
        return _run_naive_bayes(email)
    elif model == "lstm":
        return _run_lstm(email)


def classify_batch(emails: list[dict], model: str = "svm") -> list[ModelResult]:
    """Classify a list of emails. Default model: svm."""
    results: list[ModelResult] = []
    for email in emails:
        try:
            results.append(classify_email(email, model))
        except Exception as exc:
            logger.error("Failed to classify email %s: %s", email.get("id"), exc)
            results.append(ModelResult(
                predicted_class="unknown",
                probability=0.0,
                model_used=model,
                all_scores={c: 0.0 for c in CLASSES},
            ))
    return results
