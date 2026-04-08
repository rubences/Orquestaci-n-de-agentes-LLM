from __future__ import annotations

import json
import os
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

CREW_SERVER_URL = os.getenv("CREW_SERVER_URL", "http://127.0.0.1:8000")
BEE_SERVER_URL = os.getenv("BEE_SERVER_URL", "http://127.0.0.1:9000")


def acp_message(content: str, metadata: dict | None = None) -> dict:
    return {"parts": [{"content": content}], "metadata": metadata or {}}


def post_json(url: str, payload: dict) -> dict:
    resp = requests.post(url, json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()


def main() -> int:
    source_url = input("URL: ").strip()
    if not source_url:
        print("Debes introducir una URL.")
        return 1

    # 1) Research + Song writing (crew server)
    song_msg = post_json(
        f"{CREW_SERVER_URL}/acp/song_writer_agent",
        acp_message(source_url, {"stage": "song_writer"}),
    )
    song = song_msg.get("parts", [{}])[0].get("content", "")

    # 2) A&R critique (beeai server)
    ar_msg = post_json(
        f"{BEE_SERVER_URL}/acp/artist_repertoire_agent",
        acp_message(song, {"stage": "a&r"}),
    )
    critique = ar_msg.get("parts", [{}])[0].get("content", "")

    # 3) Markdown report (crew server)
    combined = f"## Generated Song\n\n{song}\n\n## A&R Feedback\n\n{critique}\n"
    report_msg = post_json(
        f"{CREW_SERVER_URL}/acp/markdown_report_agent",
        acp_message(combined, {"stage": "markdown_report"}),
    )
    report = report_msg.get("parts", [{}])[0].get("content", "")

    out_path = Path("a&r_feedback.md")
    out_path.write_text(report, encoding="utf-8")

    print("\n===== FINAL REPORT =====\n")
    print(report)
    print(f"\nSaved to: {out_path.resolve()}")

    # Save raw ACP exchanges for debugging
    Path("acp_trace.json").write_text(
        json.dumps({"song": song_msg, "ar": ar_msg, "report": report_msg}, indent=2),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
