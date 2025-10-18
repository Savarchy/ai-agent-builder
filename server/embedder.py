# server/embedder.py
from __future__ import annotations

import os
import httpx
import logging
logger = logging.getLogger(__name__)

def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    azure_url = _azure_embed_url()
    if azure_url:
        logger.info("Embedding provider: Azure OpenAI")
        ...
    else:
        logger.info("Embedding provider: Standard OpenAI")
        ...

# -------- Normal OpenAI (default) --------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# -------- Azure OpenAI (optional) --------
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")              # e.g. https://myres.openai.azure.com
AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
AZURE_EMBED_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBED_MODEL")   # your *deployment* name

def _openai_client() -> httpx.Client:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set")
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    return httpx.Client(base_url=OPENAI_BASE_URL, headers=headers, timeout=httpx.Timeout(30.0))

def _azure_embed_url() -> str:
    if not (AZURE_ENDPOINT and AZURE_API_KEY and AZURE_EMBED_DEPLOYMENT):
        return ""  # not using Azure
    return f"{AZURE_ENDPOINT}/openai/deployments/{AZURE_EMBED_DEPLOYMENT}/embeddings?api-version={AZURE_API_VERSION}"

def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Returns embeddings for a list of texts. Uses Azure if its env vars are present;
    otherwise uses the public OpenAI API.
    """
    if not texts:
        return []

    # ---- Azure path (if configured) ----
    azure_url = _azure_embed_url()
    if azure_url:
        headers = {"api-key": AZURE_API_KEY, "Content-Type": "application/json"}
        payload = {"input": texts}
        try:
            with httpx.Client(timeout=httpx.Timeout(30.0)) as c:
                r = c.post(azure_url, headers=headers, json=payload)
                r.raise_for_status()
                data = r.json()
                return [item["embedding"] for item in data["data"]]
        except httpx.ConnectError as e:
            raise RuntimeError(f"Embedding backend failed (connect): {e}") from e
        except httpx.HTTPStatusError as e:
            raise RuntimeError(f"Embedding backend HTTP {e.response.status_code}: {e.response.text}") from e
        except Exception as e:
            raise RuntimeError(f"Embedding backend failed: {e}") from e

    # ---- Normal OpenAI path ----
    payload = {"model": EMBEDDING_MODEL, "input": texts}
    try:
        with _openai_client() as c:
            r = c.post("/embeddings", json=payload)
            r.raise_for_status()
            data = r.json()
            return [item["embedding"] for item in data["data"]]
    except httpx.ConnectError as e:
        # This is the classic “[Errno 111] Connection refused” case
        raise RuntimeError(f"Embedding backend failed (connect): {e}") from e
    except httpx.HTTPStatusError as e:
        raise RuntimeError(f"Embedding backend HTTP {e.response.status_code}: {e.response.text}") from e
    except Exception as e:
        raise RuntimeError(f"Embedding backend failed: {e}") from e

def embed_query(text: str) -> list[float]:
    embs = embed_texts([text])
    return embs[0] if embs else []
