from __future__ import annotations

import json
import os
import uuid

import requests
from dotenv import load_dotenv

load_dotenv()

AGENT_URL = os.getenv("BEEAI_AGENT_URL", "http://127.0.0.1:9999")


def fetch_agent_card() -> dict:
    resp = requests.get(f"{AGENT_URL}/.well-known/agent-card.json", timeout=20)
    resp.raise_for_status()
    return resp.json()


def send_message(prompt: str) -> str:
    payload = {
        "jsonrpc": "2.0",
        "id": str(uuid.uuid4()),
        "method": "message/send",
        "params": {
            "message": {
                "role": "user",
                "parts": [{"type": "text", "text": prompt}],
            }
        },
    }

    resp = requests.post(f"{AGENT_URL}/", json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()

    if "error" in data:
        return f"RPC Error: {data['error']}"

    result = data.get("result", {})
    message = result.get("message", {})
    parts = message.get("parts", [])
    if not parts:
        return "No response parts returned."

    first = parts[0]
    return first.get("text", first.get("content", ""))


def main() -> int:
    try:
        card = fetch_agent_card()
        print("Connected to:", card.get("name"), "| protocol:", card.get("protocolVersion"))
    except Exception as exc:
        print("Unable to fetch AgentCard:", exc)
        return 1

    print("Type your message (Ctrl+C to exit):")

    while True:
        try:
            user_prompt = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye")
            break

        if not user_prompt:
            continue

        try:
            answer = send_message(user_prompt)
            print("Agent:", answer)
        except Exception as exc:
            print("Client error:", exc)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
