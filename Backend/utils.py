"""
utils.py — Shared utility helpers for the email classifier app.
"""

from __future__ import annotations

import os
import logging
import re
from datetime import datetime


# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────

def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# ─────────────────────────────────────────────
# Environment check (base only — no API keys needed)
# ─────────────────────────────────────────────

def check_env() -> list[str]:
    """Returns list of missing required env vars (empty = all good)."""
    required = ["SUPABASE_URL", "SUPABASE_SERVICE_KEY"]
    return [var for var in required if not os.getenv(var)]


# ─────────────────────────────────────────────
# Text helpers
# ─────────────────────────────────────────────

def sanitize_text(text: str, max_chars: int = 2000) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_chars]


def extract_domain(email_address: str) -> str:
    match = re.search(r"[\w.+-]+@([\w-]+\.[\w.]+)", email_address)
    return match.group(1).lower() if match else ""


def format_timestamp(iso_ts: str) -> str:
    try:
        dt = datetime.fromisoformat(iso_ts)
        return dt.strftime("%b %d, %Y %H:%M UTC")
    except ValueError:
        return iso_ts


# ─────────────────────────────────────────────
# Result formatting
# Matches the 5 training classes:
# Spam | Important | Work | Promotion | Personal
# ─────────────────────────────────────────────

CLASS_EMOJI = {
    "Spam":      "🚫",
    "Important": "⚠️",
    "Work":      "💼",
    "Promotion": "📢",
    "Personal":  "👤",
    "unknown":   "❓",
}


def format_result(email: dict, result) -> str:
    """One-line human-readable summary of a classification result."""
    emoji   = CLASS_EMOJI.get(result.predicted_class, "❓")
    subject = (email.get("subject") or "(no subject)")[:60]
    return (
        f"{emoji} [{result.predicted_class:10s}] "
        f"({result.probability:.0%} confidence) — {subject}"
    )


def results_to_dicts(emails: list[dict], results: list) -> list[dict]:
    """Combine email metadata + ModelResult into flat dicts for JSON / DB."""
    output = []
    for email, result in zip(emails, results):
        output.append({
            "id":              email["id"],
            "sender":          email.get("sender", ""),
            "subject":         email.get("subject", ""),
            "snippet":         email.get("snippet", ""),
            "timestamp":       email.get("timestamp", ""),
            "predicted_class": result.predicted_class,
            "probability":     result.probability,
            "model_used":      result.model_used,
            "all_scores":      result.all_scores,
        })
    return output


# ─────────────────────────────────────────────
# Pagination
# ─────────────────────────────────────────────

def paginate(items: list, page: int, page_size: int = 20) -> dict:
    total = len(items)
    start = (page - 1) * page_size
    end   = start + page_size
    return {
        "items":       items[start:end],
        "page":        page,
        "page_size":   page_size,
        "total":       total,
        "total_pages": max(1, -(-total // page_size)),
    }