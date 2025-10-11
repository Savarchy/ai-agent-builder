import os, json, httpx, asyncio
from typing import AsyncGenerator
MODEL = os.getenv("CHAT_MODEL", "llama3.2:3b")
OLLAMA = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
async def stream_chat(system: str, user: str) -> AsyncGenerator[str, None]:
    async with httpx.AsyncClient(timeout=None) as client:
        r = await client.post(f"{OLLAMA}/api/chat", json={
            "model": MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
            "stream": True
        })
        async for line in r.aiter_lines():
            if not line: continue
            if line.startswith('{'):
                try:
                    obj = json.loads(line)
                    msg = obj.get("message", {}).get("content", "")
                    if msg: yield msg
                except: pass
