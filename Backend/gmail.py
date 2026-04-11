"""
gmail.py — Fetches emails from Gmail using the Gmail API.
Handles OAuth2 authentication via token.pkl, token refresh, and raw email parsing.

Auth strategy (in priority order):
  1. Load credentials from token.pkl  (pickle format — your existing file)
  2. If expired, auto-refresh using the stored refresh_token
  3. Save refreshed credentials back to token.pkl
  No browser OAuth flow is needed as long as token.pkl is present and valid.
"""

import os
import pickle
import base64
import logging
from datetime import datetime
from typing import Optional

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

logger = logging.getLogger(__name__)

SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]

# Point GMAIL_TOKEN_PATH to wherever your token.pkl lives.
# Default: credentials/token.pkl
TOKEN_PATH = os.getenv("GMAIL_TOKEN_PATH", "model/tokens.pkl")


# ─────────────────────────────────────────────
# Auth
# ─────────────────────────────────────────────

def _load_credentials() -> Credentials:
    """Load Google credentials from a .pkl file."""
    if not os.path.exists(TOKEN_PATH):
        raise FileNotFoundError(
            f"token.pkl not found at '{TOKEN_PATH}'. "
            "Set GMAIL_TOKEN_PATH env var to the correct path."
        )
    with open(TOKEN_PATH, "rb") as f:
        creds = pickle.load(f)
    logger.debug("Loaded credentials from %s", TOKEN_PATH)
    return creds


def _save_credentials(creds: Credentials) -> None:
    """Persist refreshed credentials back to the .pkl file."""
    with open(TOKEN_PATH, "wb") as f:
        pickle.dump(creds, f)
    logger.debug("Saved refreshed credentials to %s", TOKEN_PATH)


def get_gmail_service():
    """
    Authenticate using token.pkl and return a Gmail API service object.
    Automatically refreshes expired tokens and re-saves the pkl.
    """
    creds = _load_credentials()

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            logger.info("Access token expired — refreshing...")
            creds.refresh(Request())
            _save_credentials(creds)
            logger.info("Token refreshed and saved to %s", TOKEN_PATH)
        else:
            raise RuntimeError(
                "Credentials in token.pkl are invalid and cannot be refreshed. "
                "Please regenerate your token.pkl via the OAuth flow."
            )

    return build("gmail", "v1", credentials=creds)


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _decode_body(payload: dict) -> str:
    """Recursively extract plain-text body from a Gmail message payload."""
    mime = payload.get("mimeType", "")

    if mime == "text/plain":
        data = payload.get("body", {}).get("data", "")
        if data:
            return base64.urlsafe_b64decode(data).decode("utf-8", errors="replace")

    if mime.startswith("multipart/"):
        for part in payload.get("parts", []):
            text = _decode_body(part)
            if text:
                return text

    return ""


def _parse_headers(headers: list[dict]) -> dict:
    """Return a flat dict of header name → value."""
    return {h["name"].lower(): h["value"] for h in headers}


# ─────────────────────────────────────────────
# Core fetch
# ─────────────────────────────────────────────

def fetch_emails(
    max_results: int = 50,
    query: str = "",
    label_ids: list[str] | None = None,
) -> list[dict]:
    """
    Fetch emails from the authenticated Gmail inbox.

    Args:
        max_results: Maximum number of emails to retrieve.
        query:       Gmail search query string (e.g. "is:unread after:2024/01/01").
        label_ids:   List of label IDs to filter by (default: ["INBOX"]).

    Returns:
        List of dicts with keys:
            id, sender, subject, snippet, timestamp, body
    """
    if label_ids is None:
        label_ids = ["INBOX"]

    service = get_gmail_service()
    emails: list[dict] = []

    try:
        list_params: dict = {
            "userId": "me",
            "maxResults": max_results,
            "labelIds": label_ids,
        }
        if query:
            list_params["q"] = query

        result = service.users().messages().list(**list_params).execute()
        messages = result.get("messages", [])

        for msg_ref in messages:
            msg_id = msg_ref["id"]
            try:
                msg = (
                    service.users()
                    .messages()
                    .get(userId="me", id=msg_id, format="full")
                    .execute()
                )

                payload = msg.get("payload", {})
                headers = _parse_headers(payload.get("headers", []))
                body = _decode_body(payload)

                # Timestamp: Gmail returns internalDate as milliseconds since epoch
                raw_ts = msg.get("internalDate", "0")
                ts = datetime.utcfromtimestamp(int(raw_ts) / 1000).isoformat()

                emails.append(
                    {
                        "id": msg_id,
                        "sender": headers.get("from", ""),
                        "subject": headers.get("subject", "(no subject)"),
                        "snippet": msg.get("snippet", ""),
                        "timestamp": ts,
                        "body": body.strip(),
                    }
                )

            except HttpError as e:
                logger.warning("Could not fetch message %s: %s", msg_id, e)

    except HttpError as e:
        logger.error("Gmail API error: %s", e)
        raise

    logger.info("Fetched %d emails", len(emails))
    return emails


def fetch_single_email(email_id: str) -> dict | None:
    """Fetch a single email by its Gmail message ID."""
    service = get_gmail_service()
    try:
        msg = (
            service.users()
            .messages()
            .get(userId="me", id=email_id, format="full")
            .execute()
        )
        payload = msg.get("payload", {})
        headers = _parse_headers(payload.get("headers", []))
        body = _decode_body(payload)
        raw_ts = msg.get("internalDate", "0")
        ts = datetime.utcfromtimestamp(int(raw_ts) / 1000).isoformat()

        return {
            "id": email_id,
            "sender": headers.get("from", ""),
            "subject": headers.get("subject", "(no subject)"),
            "snippet": msg.get("snippet", ""),
            "timestamp": ts,
            "body": body.strip(),
        }
    except HttpError as e:
        logger.error("Failed to fetch email %s: %s", email_id, e)
        return None
