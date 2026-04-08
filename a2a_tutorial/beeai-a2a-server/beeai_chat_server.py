from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

load_dotenv(Path(__file__).resolve().parent / ".env")
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

APP_PORT = int(os.getenv("A2A_PORT", "9999"))
BASE_URL = os.getenv("BEEAI_AGENT_URL", f"http://127.0.0.1:{APP_PORT}")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-r1-distill-llama-70b:free")

app = FastAPI(title="BeeAI A2A Chat Server", version="1.0.0")


def think_tool(prompt: str) -> str:
    return (
        "Structured reasoning:\n"
        "1) Clarify objective\n"
        "2) Identify constraints\n"
        "3) Produce concise recommendation\n"
        f"Input analyzed: {prompt[:240]}"
    )


def duckduckgo_search_tool(query: str) -> str:
    try:
        resp = requests.get(
            "https://api.duckduckgo.com/",
            params={"q": query, "format": "json", "no_redirect": "1", "no_html": "1"},
            timeout=20,
        )
        resp.raise_for_status()
        data = resp.json()

        abstract = data.get("AbstractText", "").strip()
        related = data.get("RelatedTopics", [])
        first_related = ""
        if related and isinstance(related, list):
            first = related[0]
            if isinstance(first, dict):
                first_related = first.get("Text", "")

        out = abstract or first_related or "No concise result found from DuckDuckGo."
        return out[:1200]
    except Exception as exc:
        return f"DuckDuckGoSearchTool error: {exc}"


def open_meteo_tool(location: str) -> str:
    try:
        geo = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": location, "count": 1, "language": "en", "format": "json"},
            timeout=20,
        )
        geo.raise_for_status()
        gdata = geo.json()
        results = gdata.get("results", [])
        if not results:
            return f"OpenMeteoTool: location not found: {location}"

        first = results[0]
        lat, lon = first["latitude"], first["longitude"]
        city = first.get("name", location)

        wx = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat,
                "longitude": lon,
                "current": "temperature_2m,wind_speed_10m,weather_code",
            },
            timeout=20,
        )
        wx.raise_for_status()
        wdata = wx.json().get("current", {})

        return (
            f"Weather for {city}: temperature={wdata.get('temperature_2m')} C, "
            f"wind={wdata.get('wind_speed_10m')} km/h, code={wdata.get('weather_code')}"
        )
    except Exception as exc:
        return f"OpenMeteoTool error: {exc}"


def wikipedia_tool(topic: str) -> str:
    try:
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{topic.replace(' ', '_')}"
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        return (data.get("extract") or "No Wikipedia summary found.")[:1600]
    except Exception as exc:
        return f"WikipediaTool error: {exc}"


def tool_router(user_text: str) -> dict[str, str]:
    low = user_text.lower()

    outputs: dict[str, str] = {"ThinkTool": think_tool(user_text)}

    if any(k in low for k in ["weather", "temperatura", "clima"]):
        # try last word as naive location fallback
        location = user_text.split()[-1].strip("?.!,") if user_text.split() else "Madrid"
        outputs["OpenMeteoTool"] = open_meteo_tool(location)

    if any(k in low for k in ["search", "buscar", "news", "noticias"]):
        outputs["DuckDuckGoSearchTool"] = duckduckgo_search_tool(user_text)

    if any(k in low for k in ["wikipedia", "wiki", "tell me about", "hablame de", "háblame de"]):
        topic = user_text.replace("wikipedia", "").replace("wiki", "").strip() or "Artificial intelligence"
        outputs["WikipediaTool"] = wikipedia_tool(topic)

    if len(outputs) == 1:
        # Default web lookup to keep useful behavior.
        outputs["DuckDuckGoSearchTool"] = duckduckgo_search_tool(user_text)

    return outputs


def llm_summarize(user_text: str, tool_outputs: dict[str, str]) -> str:
    context = "\n\n".join([f"[{k}]\n{v}" for k, v in tool_outputs.items()])

    if not OPENROUTER_API_KEY:
        return f"Tool-based answer:\n{context[:2500]}"

    prompt = (
        "You are RequirementAgent. Answer the user request using the tool outputs below. "
        "Be concise, factual, and helpful.\n\n"
        f"User request:\n{user_text}\n\n"
        f"Tool outputs:\n{context}\n\n"
        "Final answer:"
    )

    try:
        resp = requests.post(
            f"{OPENROUTER_BASE_URL}/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": OPENROUTER_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2,
            },
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception as exc:
        return f"OpenRouter error, fallback to tool outputs. Details: {exc}\n\n{context[:2500]}"


def extract_user_text(payload: dict[str, Any]) -> str:
    params = payload.get("params", {})
    message = params.get("message", {})
    parts = message.get("parts", [])

    if parts and isinstance(parts, list):
        first = parts[0]
        if isinstance(first, dict):
            # Try common shapes: {text: ...} or {content: ...}
            if "text" in first:
                return str(first.get("text", ""))
            if "content" in first:
                return str(first.get("content", ""))

    # fallback for generic shape
    return str(params.get("input", ""))


@app.get("/.well-known/agent-card.json")
def agent_card() -> dict[str, Any]:
    return {
        "capabilities": {"streaming": True},
        "defaultInputModes": ["text"],
        "defaultOutputModes": ["text"],
        "description": "An agent that can search the web, check weather, use Wikipedia and reason step-by-step.",
        "name": "RequirementAgent",
        "preferredTransport": "JSONRPC",
        "protocolVersion": "0.3.0",
        "skills": [
            {
                "id": "RequirementAgent",
                "name": "RequirementAgent",
                "description": "Web search, weather data, Wikipedia lookup and reasoning tool orchestration.",
                "tags": ["think", "search", "weather", "wikipedia"],
            }
        ],
        "url": BASE_URL,
        "version": "1.0.0",
    }


@app.post("/")
async def a2a_rpc(request: Request):
    payload = await request.json()

    if payload.get("jsonrpc") != "2.0":
        raise HTTPException(status_code=400, detail="Only JSON-RPC 2.0 is supported")

    method = payload.get("method")
    if method not in {"message/send", "tasks/send"}:
        return JSONResponse(
            {
                "jsonrpc": "2.0",
                "id": payload.get("id"),
                "error": {"code": -32601, "message": f"Method not found: {method}"},
            }
        )

    user_text = extract_user_text(payload)
    tool_outputs = tool_router(user_text)
    final_answer = llm_summarize(user_text, tool_outputs)

    result = {
        "message": {
            "role": "agent",
            "parts": [{"type": "text", "text": final_answer}],
            "metadata": {"tools_used": list(tool_outputs.keys())},
        }
    }

    return JSONResponse({"jsonrpc": "2.0", "id": payload.get("id"), "result": result})


@app.get("/stream")
def stream(q: str):
    tool_outputs = tool_router(q)
    final_answer = llm_summarize(q, tool_outputs)

    async def event_gen():
        yield {"event": "run.created", "data": json.dumps({"status": "created"})}
        yield {"event": "run.in-progress", "data": json.dumps({"status": "in-progress"})}
        # Send final in a single text chunk for simplicity.
        yield {
            "event": "message.part",
            "data": json.dumps({"type": "text", "text": final_answer}),
        }
        yield {"event": "run.completed", "data": json.dumps({"status": "completed"})}

    return EventSourceResponse(event_gen())


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("beeai_chat_server:app", host="0.0.0.0", port=APP_PORT, reload=False)
