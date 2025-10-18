# server/retriever.py
import os
from typing import List, Sequence, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import text as sql_text

# Keep this in sync with your embedder (o4-mini: 1536, text-embedding-3-small: 1536, etc.)
EMBED_DIM = int(os.getenv("EMBED_DIM", "1536"))

def _to_float_list(val) -> List[float]:
    """
    Normalize whatever Postgres returns for pgvector into a List[float].
    Handles: text like "[0.1, -0.2, ...]", bytes, memoryview, arrays, tuples.
    """
    if val is None:
        return []

    # Already a list/tuple of numbers?
    if isinstance(val, (list, tuple)):
        return [float(x) for x in val]

    # memoryview -> bytes -> str
    if isinstance(val, memoryview):
        val = val.tobytes()

    # bytes -> str
    if isinstance(val, (bytes, bytearray)):
        val = val.decode("utf-8", errors="ignore")

    # str like "[0.1, -0.2, ...]"
    if isinstance(val, str):
        s = val.strip()
        if s.startswith("[") and s.endswith("]"):
            s = s[1:-1]
        if not s:
            return []
        # remove stray whitespace/newlines and split
        parts = s.replace("\n", " ").split(",")
        out = []
        for p in parts:
            p = p.strip()
            if p:
                out.append(float(p))
        return out

    # Fallback (unexpected type)
    try:
        return [float(x) for x in val]  # may raise
    except Exception:
        # Last resort: stringify and parse like a vector text
        s = str(val).strip()
        if s.startswith("[") and s.endswith("]"):
            s = s[1:-1]
        if not s:
            return []
        return [float(p.strip()) for p in s.split(",") if p.strip()]

def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    # guard for empty inputs
    if not a or not b:
        return 0.0
    # truncate/pad to same length (defensive)
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    a = a[:n]
    b = b[:n]
    num = sum(x * y for x, y in zip(a, b))
    den_a = sum(x * x for x in a) ** 0.5
    den_b = sum(y * y for y in b) ** 0.5
    if den_a == 0 or den_b == 0:
        return 0.0
    return num / (den_a * den_b)

def _mmr(query_vec: List[float], cand_vecs: List[List[float]], k: int = 6, lambda_mult: float = 0.7) -> List[int]:
    """Return indices of the selected candidates using Maximal Marginal Relevance."""
    if not cand_vecs:
        return []
    k = max(1, min(k, len(cand_vecs)))

    # similarities to the query
    sims_to_q = [_cosine(query_vec, v) for v in cand_vecs]

    selected: List[int] = []
    remaining = set(range(len(cand_vecs)))

    # seed with the highest similarity to the query
    best_first = max(remaining, key=lambda i: sims_to_q[i])
    selected.append(best_first)
    remaining.remove(best_first)

    while len(selected) < k and remaining:
        def mmr_score(i: int) -> float:
            # diversity against already selected
            div = 0.0
            if selected:
                div = max(_cosine(cand_vecs[i], cand_vecs[j]) for j in selected)
            return lambda_mult * sims_to_q[i] - (1 - lambda_mult) * div

        next_i = max(remaining, key=mmr_score)
        selected.append(next_i)
        remaining.remove(next_i)

    return selected

def search_mmr(db: Session, qvec: List[float], k: int = 6, pool: int = 48, lambda_mult: float = 0.7) -> List[Tuple[str, str, str, List[float]]]:
    """
    Returns list of (text, title, url, embedding_vec) selected with MMR.
    Two modes:
      - DB_ANN=1 → let Postgres order with <=> then fetch & MMR (fast when ivfflat exists).
      - otherwise → fetch a pool (embedding::text) and MMR in Python (safe on tiny/free plans).
    """
    use_db_ann = os.getenv("DB_ANN", "0") == "1"

    if use_db_ann:
        # Requires pgvector operator and dimension match; fastest when the IVFFLAT index exists.
        sql = sql_text(f"""
            SELECT
              c.text,
              d.title,
              d.url,
              c.embedding::text AS emb_text
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            WHERE c.embedding IS NOT NULL
            ORDER BY c.embedding <=> (:qvec)::vector({EMBED_DIM})
            LIMIT :pool;
        """)
        rows = db.execute(sql, {"qvec": f"[{','.join(str(float(x)) for x in qvec)}]", "pool": int(pool)}).fetchall()
    else:
        # Portable path: no <=> ordering; take a pool (recent first) and compute MMR in Python.
        sql = sql_text("""
            SELECT
              c.text,
              d.title,
              d.url,
              c.embedding::text AS emb_text
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            WHERE c.embedding IS NOT NULL
            ORDER BY c.created_at DESC
            LIMIT :pool;
        """)
        rows = db.execute(sql, {"pool": int(pool)}).fetchall()

    if not rows:
        return []

    # Normalize embeddings
    texts: List[str] = []
    titles: List[str] = []
    urls: List[str] = []
    cand_vecs: List[List[float]] = []

    for r in rows:
        # r = (text, title, url, emb_text)
        texts.append(r[0])
        titles.append(r[1])
        urls.append(r[2])
        cand_vecs.append(_to_float_list(r[3]))

    keep_idx = _mmr(qvec, cand_vecs, k=k, lambda_mult=lambda_mult)

    out: List[Tuple[str, str, str, List[float]]] = []
    for i in keep_idx:
        out.append((texts[i], titles[i], urls[i], cand_vecs[i]))
    return out
