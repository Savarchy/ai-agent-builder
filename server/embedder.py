# server/embedder.py
import os
import httpx
from typing import List, Union

OLLAMA_BASE = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")

def _ensure_2d(v: Union[List[float], List[List[float]]]) -> List[List[float]]:
    # Turn [f,f,...] into [[f,f,...]] if needed
    if not v:
        return []
    return v if isinstance(v[0], list) else [v]  # type: ignore[index]

def _parse_embed_response(data) -> List[List[float]]:
    """
    Normalize possible Ollama responses into List[List[float]].
    Supports:
      - /api/embed => { model, embeddings: [[...], ...] }
      - /api/embeddings (new) => { data: [{embedding:[...]}] }
      - /api/embeddings (old) => { embedding:[...] }  (single)
    """
    if isinstance(data, dict):
        if "embeddings" in data:                    # /api/embed
            return _ensure_2d(data["embeddings"])
        if "data" in data and isinstance(data["data"], list):  # new /api/embeddings
            return [row.get("embedding", []) for row in data["data"]]
        if "embedding" in data:                     # old /api/embeddings single
            return _ensure_2d(data["embedding"])
    return []

def embed_texts(texts: List[str]) -> List[List[float]]:
    if not texts:
        return []
    # 1) Try /api/embed (the one you already verified works)
    try:
        with httpx.Client(timeout=60) as client:
            r = client.post(
                f"{OLLAMA_BASE}/api/embed",
                json={"model": EMBED_MODEL, "input": texts},
            )
        if r.status_code == 200:
            vecs = _parse_embed_response(r.json())
            if vecs and all(isinstance(v, list) and v for v in vecs):
                return vecs
            else:
                # fallthrough to try /api/embeddings
                pass
        else:
            # fall through and try /api/embeddings as a second attempt
            pass
    except Exception:
        # try fallback
        pass

    # 2) Fallback: /api/embeddings (supports "input", also "prompt" for single)
    try:
        payload = {"model": EMBED_MODEL, "input": texts}
        with httpx.Client(timeout=60) as client:
            r = client.post(f"{OLLAMA_BASE}/api/embeddings", json=payload)
        if r.status_code == 200:
            vecs = _parse_embed_response(r.json())
            if vecs and all(isinstance(v, list) and v for v in vecs):
                return vecs
            # As a last resort, embed one by one with "prompt"
            out: List[List[float]] = []
            with httpx.Client(timeout=60) as client:
                for t in texts:
                    rr = client.post(
                        f"{OLLAMA_BASE}/api/embeddings",
                        json={"model": EMBED_MODEL, "prompt": t},
                    )
                    if rr.status_code != 200:
                        raise RuntimeError(f"Ollama embeddings error: {rr.text}")
                    vv = _parse_embed_response(rr.json())
                    if not vv:
                        raise RuntimeError("No embedding returned for a chunk")
                    out.append(vv[0])
            return out
        else:
            raise RuntimeError(f"Ollama embeddings HTTP {r.status_code}: {r.text}")
    except Exception as e:
        raise RuntimeError(f"Embedding backend failed: {e}") from e
# --- single-query convenience wrapper ---
from typing import List

def embed_query(text: str) -> List[float]:
    """
    Return a single embedding vector for one query string.
    Uses embed_texts() under the hood and returns the first vector.
    """
    vecs = embed_texts([text])
    if not vecs or not vecs[0]:
        raise RuntimeError("No embedding returned for query")
    return vecs[0]

