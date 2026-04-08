from __future__ import annotations

import os
from datetime import datetime
from typing import Any

import requests
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

load_dotenv()

app = FastAPI(title="ACP BeeAI-style A&R server", version="1.0.0")


class ACPMessagePart(BaseModel):
    content: str


class ACPMessage(BaseModel):
    parts: list[ACPMessagePart]
    metadata: dict[str, Any] | None = None


def _openrouter_critique(song: str) -> str:
    api_key = os.getenv("OPENROUTER_API_KEY", "")
    base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    model = os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-r1-distill-llama-70b:free")

    if not api_key:
        return "A&R Feedback unavailable: missing OPENROUTER_API_KEY."

    url = f"{base_url}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    prompt = (
        "You are an A&R specialist. Review the song and return concise markdown bullets with: "
        "Hit Potential Score (1-10), Target Audience, Strengths, Concerns, Market Comparison, Recommendation.\n\n"
        f"Song:\n{song}"
    )

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.4,
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception as exc:
        return f"A&R Feedback fallback due to API error: {exc}"


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "server": "beeai"}


@app.post("/acp/artist_repertoire_agent")
def artist_repertoire_agent(message: ACPMessage) -> dict[str, Any]:
    song = message.parts[0].content if message.parts else ""
    feedback = _openrouter_critique(song)

    return {
        "parts": [{"content": feedback}],
        "metadata": {
            "agent": "artist_repertoire_agent",
            "timestamp": datetime.utcnow().isoformat() + "Z",
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("artist_repertoire_agent:app", host="127.0.0.1", port=9000, reload=False)
