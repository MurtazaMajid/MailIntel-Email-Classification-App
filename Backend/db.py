"""
db.py — Supabase database layer.
Handles all read/write operations against the `emails` table.

Table schema (from Supabase):
    id              text        PK
    sender          text
    subject         text
    snippet         text
    timestamp       text
    predicted_class text
    probability     float8
    model_used      boolean     ← stored as text in practice; adapt as needed
    processed_at    timestamp
"""

from __future__ import annotations

import os
import logging
from datetime import datetime, timezone
from typing import Optional

from supabase import create_client, Client  # type: ignore

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Client singleton
# ─────────────────────────────────────────────

_client: Optional[Client] = None


def get_client() -> Client:
    global _client
    if _client is None:
        url = os.environ["SUPABASE_URL"]
        key = os.environ["SUPABASE_SERVICE_KEY"]  # service key for server-side writes
        _client = create_client(url, key)
    return _client


# ─────────────────────────────────────────────
# Write operations
# ─────────────────────────────────────────────

def upsert_email(
    email: dict,
    predicted_class: str,
    probability: float,
    model_used: str,
) -> dict | None:
    """
    Insert or update a single email record.

    Uses upsert so re-processing the same email ID is safe.
    """
    db = get_client()
    row = {
        "id": email["id"],
        "sender": email.get("sender", ""),
        "subject": email.get("subject", ""),
        "snippet": email.get("snippet", ""),
        "timestamp": email.get("timestamp", ""),
        "predicted_class": predicted_class,
        "probability": probability,
        "model_used": model_used,          # stored as text; column is boolean in DDL
        "processed_at": datetime.now(timezone.utc).isoformat(),
    }

    try:
        response = db.table("emails").upsert(row).execute()
        logger.info("Upserted email %s → %s (%.2f)", email["id"], predicted_class, probability)
        return response.data[0] if response.data else None
    except Exception as exc:
        logger.error("DB upsert failed for %s: %s", email["id"], exc)
        raise


def upsert_batch(
    emails: list[dict],
    results: list,       # list[ModelResult] — imported lazily to avoid circular import
) -> int:
    """
    Bulk-upsert a list of emails + their classification results.
    Returns the count of successfully written rows.
    """
    db = get_client()
    rows = []
    for email, result in zip(emails, results):
        rows.append(
            {
                "id": email["id"],
                "sender": email.get("sender", ""),
                "subject": email.get("subject", ""),
                "snippet": email.get("snippet", ""),
                "timestamp": email.get("timestamp", ""),
                "predicted_class": result.predicted_class,
                "probability": result.probability,
                "model_used": result.model_used,
                "processed_at": datetime.now(timezone.utc).isoformat(),
            }
        )

    if not rows:
        return 0

    try:
        response = db.table("emails").upsert(rows).execute()
        count = len(response.data) if response.data else 0
        logger.info("Bulk upserted %d rows", count)
        return count
    except Exception as exc:
        logger.error("Bulk upsert failed: %s", exc)
        raise


# ─────────────────────────────────────────────
# Read operations
# ─────────────────────────────────────────────

def get_all_emails(limit: int = 100, offset: int = 0) -> list[dict]:
    """Return emails ordered by processed_at descending."""
    db = get_client()
    response = (
        db.table("emails")
        .select("*")
        .order("processed_at", desc=True)
        .range(offset, offset + limit - 1)
        .execute()
    )
    return response.data or []


def get_email_by_id(email_id: str) -> dict | None:
    db = get_client()
    response = db.table("emails").select("*").eq("id", email_id).limit(1).execute()
    data = response.data
    return data[0] if data else None


def get_emails_by_class(predicted_class: str, limit: int = 50) -> list[dict]:
    db = get_client()
    response = (
        db.table("emails")
        .select("*")
        .eq("predicted_class", predicted_class)
        .order("processed_at", desc=True)
        .limit(limit)
        .execute()
    )
    return response.data or []


def get_unprocessed_ids(candidate_ids: list[str]) -> list[str]:
    """
    Given a list of Gmail message IDs, return those NOT yet in the DB.
    Useful to skip re-processing emails we've already classified.
    """
    if not candidate_ids:
        return []
    db = get_client()
    response = (
        db.table("emails")
        .select("id")
        .in_("id", candidate_ids)
        .execute()
    )
    existing = {row["id"] for row in (response.data or [])}
    return [eid for eid in candidate_ids if eid not in existing]


def get_stats() -> dict:
    """Return aggregate counts per class."""
    db = get_client()
    response = db.table("emails").select("predicted_class").execute()
    rows = response.data or []
    stats: dict[str, int] = {}
    for row in rows:
        cls = row.get("predicted_class", "unknown")
        stats[cls] = stats.get(cls, 0) + 1
    stats["total"] = len(rows)
    return stats
