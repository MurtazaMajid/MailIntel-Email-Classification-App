"""
main.py — FastAPI application entrypoint.

Endpoints:
    GET  /health              — liveness check
    GET  /models              — list available models with display names
    GET  /emails              — list classified emails from DB
    GET  /emails/{id}         — single email detail
    POST /run                 — fetch + classify + store emails
    GET  /stats               — class distribution counts

Run locally:
    uvicorn app.main:app --reload --port 8000
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.gmail import fetch_emails
from app.model import classify_batch, AVAILABLE_MODELS, MODEL_DISPLAY_NAMES, MODEL_DESCRIPTIONS
from app.db import (
    upsert_batch,
    get_all_emails,
    get_email_by_id,
    get_emails_by_class,
    get_unprocessed_ids,
    get_stats,
)
from app.utils import setup_logging, format_result

setup_logging()
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Gmail Classifier API",
    description="Fetch Gmail messages and classify them using pre-trained ML models.",
    version="1.0.0",
)

# ─────────────────────────────────────────────
# CORS — reads ALLOWED_ORIGINS from env so you
# can lock it down to your Vercel URL in prod:
#   ALLOWED_ORIGINS=https://your-app.vercel.app
# Falls back to * for local dev only.
# ─────────────────────────────────────────────
_raw_origins = os.getenv("ALLOWED_ORIGINS", "*")
ALLOWED_ORIGINS = [o.strip() for o in _raw_origins.split(",")] if _raw_origins != "*" else ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# Request / Response schemas
# ─────────────────────────────────────────────

class RunRequest(BaseModel):
    model: str = "lstm"          # "svm" | "naive_bayes" | "lstm"
    max_emails: int = 50
    query: str = ""              # Gmail search query (optional)
    skip_existing: bool = True   # skip emails already in DB


class RunResponse(BaseModel):
    fetched: int
    classified: int
    stored: int
    model_used: str
    summary: list[str]


class EmailRecord(BaseModel):
    id: str
    sender: str
    subject: Optional[str] = None
    snippet: Optional[str] = None
    timestamp: Optional[str] = None
    predicted_class: Optional[str] = None
    probability: Optional[float] = None
    model_used: Optional[str] = None
    processed_at: Optional[str] = None


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/models")
def list_models():
    """List all available pre-trained classification models."""
    return {
        "models": AVAILABLE_MODELS,
        "display_names": MODEL_DISPLAY_NAMES,
        "descriptions": MODEL_DESCRIPTIONS,
        "default": "lstm",
    }


@app.get("/stats")
def stats():
    """Return class distribution counts from the DB."""
    return get_stats()


@app.get("/emails", response_model=list[EmailRecord])
def list_emails(
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    predicted_class: Optional[str] = Query(None),
):
    """
    Return classified emails from Supabase.
    Filter by ?predicted_class=Spam|Important|Work|Promotion|Personal
    """
    if predicted_class:
        return get_emails_by_class(predicted_class, limit=limit)
    return get_all_emails(limit=limit, offset=offset)


@app.get("/emails/{email_id}", response_model=EmailRecord)
def get_email(email_id: str):
    record = get_email_by_id(email_id)
    if not record:
        raise HTTPException(status_code=404, detail="Email not found")
    return record


@app.post("/run", response_model=RunResponse)
def run_pipeline(body: RunRequest):
    """
    Full pipeline:
      1. Validate model name
      2. Fetch emails from Gmail (via models/tokens.pkl)
      3. Skip already-processed emails (optional)
      4. Run classification with chosen model
      5. Store results in Supabase
    """
    if body.model not in AVAILABLE_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model '{body.model}'. Choose from: {AVAILABLE_MODELS}",
        )

    # 1. Fetch
    logger.info("Fetching up to %d emails (query=%r)", body.max_emails, body.query)
    emails = fetch_emails(max_results=body.max_emails, query=body.query)
    fetched_count = len(emails)

    if not emails:
        return RunResponse(
            fetched=0, classified=0, stored=0,
            model_used=body.model, summary=[],
        )

    # 2. Skip already processed
    if body.skip_existing:
        all_ids = [e["id"] for e in emails]
        new_ids = set(get_unprocessed_ids(all_ids))
        emails = [e for e in emails if e["id"] in new_ids]
        logger.info(
            "Skipping %d already-processed; %d new to classify",
            fetched_count - len(emails), len(emails),
        )

    if not emails:
        return RunResponse(
            fetched=fetched_count, classified=0, stored=0,
            model_used=body.model, summary=["All emails already processed."],
        )

    # 3. Classify
    logger.info("Classifying %d emails with model=%s", len(emails), body.model)
    results = classify_batch(emails, model=body.model)

    # 4. Store
    stored = upsert_batch(emails, results)
    summary = [format_result(e, r) for e, r in zip(emails, results)]

    return RunResponse(
        fetched=fetched_count,
        classified=len(results),
        stored=stored,
        model_used=body.model,
        summary=summary,
    )
