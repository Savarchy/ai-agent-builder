# server/llm.py
import os
import json
import httpx
from typing import AsyncGenerator, Optional, List, Dict, Any

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY env var is required")

HEADERS = {
    "Authorization": f"Bearer {OPENAI_API_KEY}",
    "Content-Type": "application/json",
}

def _build_messages(system: Optional[str], user: str) -> List[Dict[str, Any]]:
    msgs: List[Dict[str, Any]] = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": user})
    return msgs

async def stream_chat(system: Optional[str], user: str) -> AsyncGenerator[str, None]:
    """
    Async-generator that yields string chunks of the assistant's reply.
    """
    payload = {
        "model": MODEL,
        "messages": _build_messages(system, user),
        "stream": True,
        "temperature": 0.2,
    }

    # We’ll parse the "data: ..." Server-Sent-Events that OpenAI returns.
    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream(
            "POST", "https://api.openai.com/v1/chat/completions",
            headers=HEADERS, json=payload
        ) as resp:
            resp.raise_for_status()
            async for raw_line in resp.aiter_lines():
                if not raw_line or not raw_line.startswith("data: "):
                    continue
                data = raw_line[6:].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except Exception:
                    continue

                # Chat Completions streaming schema: choices[].delta.content
                for choice in obj.get("choices", []):
                    delta = choice.get("delta") or {}
                    txt = delta.get("content")
                    if txt:
                        yield txt

async def chat(system: Optional[str], user: str) -> str:
    """
    Non-streaming helper (optional). Returns the full text.
    """
    buf = []
    async for chunk in stream_chat(system, user):
        buf.append(chunk)
    return "".join(buf)
