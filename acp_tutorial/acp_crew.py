from __future__ import annotations

import re
from datetime import datetime
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="ACP crewAI-style server", version="1.0.0")


class ACPMessagePart(BaseModel):
    content: str


class ACPMessage(BaseModel):
    parts: list[ACPMessagePart]
    metadata: dict[str, Any] | None = None


def _extract_themes(text: str) -> list[str]:
    words = re.findall(r"[a-zA-Z]{4,}", text.lower())
    stop = {
        "https",
        "http",
        "from",
        "with",
        "that",
        "this",
        "have",
        "your",
        "about",
        "what",
        "into",
        "will",
        "their",
    }
    filtered = [w for w in words if w not in stop]
    uniq: list[str] = []
    for w in filtered:
        if w not in uniq:
            uniq.append(w)
        if len(uniq) >= 8:
            break
    return uniq if uniq else ["innovation", "agents", "automation"]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "server": "crew"}


@app.post("/acp/song_writer_agent")
def song_writer_agent(message: ACPMessage) -> dict[str, Any]:
    content = message.parts[0].content if message.parts else ""
    themes = _extract_themes(content)
    joined = ", ".join(themes[:5])

    song = f"""(Verse 1)
We started from a page and turned it into light,
Ideas from the wire, now singing through the night.

(Chorus)
{joined}, in rhythm we believe,
Human dreams and agent minds creating what we need.

(Verse 2)
Signals become stories, and stories become sound,
A chain of clever helpers making meaning all around.
"""

    return {
        "parts": [{"content": song}],
        "metadata": {
            "agent": "song_writer_agent",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "themes": themes,
        },
    }


@app.post("/acp/markdown_report_agent")
def markdown_report_agent(message: ACPMessage) -> dict[str, Any]:
    payload = message.parts[0].content if message.parts else ""
    report = f"""## Generated Song\n\n{payload}\n"""
    return {
        "parts": [{"content": report}],
        "metadata": {
            "agent": "markdown_report_agent",
            "timestamp": datetime.utcnow().isoformat() + "Z",
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("acp_crew:app", host="127.0.0.1", port=8000, reload=False)
