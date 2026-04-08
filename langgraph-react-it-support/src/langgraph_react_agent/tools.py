from __future__ import annotations

from datetime import datetime
from io import BytesIO
from pathlib import Path

import pandas as pd
from langchain_core.tools import tool

_TICKETS_PATH: Path | None = None
_USE_COS: bool = False
_COS_CLIENT = None
_COS_BUCKET: str | None = None
_COS_KEY: str | None = None


def configure_tools(
    tickets_path: Path,
    use_cos: bool = False,
    cos_client=None,
    cos_bucket: str | None = None,
    cos_key: str | None = None,
) -> None:
    global _TICKETS_PATH, _USE_COS, _COS_CLIENT, _COS_BUCKET, _COS_KEY
    _TICKETS_PATH = tickets_path
    _USE_COS = use_cos
    _COS_CLIENT = cos_client
    _COS_BUCKET = cos_bucket
    _COS_KEY = cos_key
    _ensure_csv()


def _ensure_csv() -> None:
    if _USE_COS:
        if not _COS_CLIENT or not _COS_BUCKET or not _COS_KEY:
            raise ValueError("COS tools are not configured correctly.")
        try:
            _COS_CLIENT.get_object(Bucket=_COS_BUCKET, Key=_COS_KEY)
        except Exception:
            df = pd.DataFrame(columns=["issue", "date_added", "urgency", "status"])
            _COS_CLIENT.put_object(
                Bucket=_COS_BUCKET,
                Key=_COS_KEY,
                Body=df.to_csv(index=False).encode("utf-8"),
            )
        return

    if _TICKETS_PATH is None:
        raise ValueError("Local CSV tools are not configured. Call configure_tools first.")

    _TICKETS_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not _TICKETS_PATH.exists():
        df = pd.DataFrame(columns=["issue", "date_added", "urgency", "status"])
        df.to_csv(_TICKETS_PATH, index=False)


def _read_df() -> pd.DataFrame:
    _ensure_csv()
    if _USE_COS:
        response = _COS_CLIENT.get_object(Bucket=_COS_BUCKET, Key=_COS_KEY)
        body = response["Body"].read()
        return pd.read_csv(BytesIO(body))
    return pd.read_csv(_TICKETS_PATH)


def _write_df(df: pd.DataFrame) -> None:
    _ensure_csv()
    if _USE_COS:
        _COS_CLIENT.put_object(
            Bucket=_COS_BUCKET,
            Key=_COS_KEY,
            Body=df.to_csv(index=False).encode("utf-8"),
        )
        return
    df.to_csv(_TICKETS_PATH, index=False)


@tool
def find_tickets() -> str:
    """Returns all tickets in a compact JSON-like string."""
    df = _read_df()
    if df.empty:
        return "No tickets found."
    return df.to_json(orient="records")


@tool
def create_ticket(issue: str, urgency: str) -> str:
    """Creates a new support ticket. urgency must be low, medium, or high."""
    urgency_norm = urgency.strip().lower()
    if urgency_norm not in {"low", "medium", "high"}:
        return "Invalid urgency. Allowed values: low, medium, high."

    df = _read_df()
    new_ticket = {
        "issue": issue.strip(),
        "date_added": datetime.now().strftime("%m-%d-%Y"),
        "urgency": urgency_norm,
        "status": "open",
    }
    df.loc[len(df)] = new_ticket
    _write_df(df)
    return "New ticket successfully created!"


@tool
def get_todays_date() -> str:
    """Returns today's date in MM-DD-YYYY format."""
    return datetime.now().strftime("%m-%d-%Y")


TOOLS = [find_tickets, create_ticket, get_todays_date]
