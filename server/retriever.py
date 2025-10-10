# server/retriever.py
from typing import List, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import text as sql_text
import numpy as np
import json

# -------------------------------------------------------------
# Ensure pgvector indexes (safe to call on startup)
# -------------------------------------------------------------
def ensure_indexes(db: Session):
    db.execute(sql_text("""
        CREATE INDEX IF NOT EXISTS ix_chunks_embedding_cosine
        ON chunks USING ivfflat (embedding vector_cosine_ops);
    """))
    db.commit()


# -------------------------------------------------------------
# Helpers
# -------------------------------------------------------------
def _to_vector_literal(vec: List[float]) -> str:
    """Convert a Python list of floats to a pgvector text literal: [0.1,0.2,...]."""
    return "[" + ",".join(f"{x:.8f}" for x in vec) + "]"

def _coerce_embedding(e):
    """Coerce DB value to list[float], regardless of how the driver returns it."""
    if e is None:
        return None
    if isinstance(e, (list, tuple)):
        return [float(x) for x in e]
    if isinstance(e, str):
        s = e.strip()
        # try JSON-like "[...]" first
        if s.startswith("[") and s.endswith("]"):
            try:
                return [float(x) for x in json.loads(s)]
            except Exception:
                pass
        # fallback: split on commas
        try:
            return [float(x) for x in s.strip("[]").split(",") if x.strip()]
        except Exception:
            return None
    if hasattr(e, "tolist"):
        return [float(x) for x in e.tolist()]
    return None

def _cos(a: np.ndarray, b: np.ndarray) -> float:
    a_n = a / (np.linalg.norm(a) + 1e-12)
    b_n = b / (np.linalg.norm(b) + 1e-12)
    return float(np.dot(a_n, b_n))


# -------------------------------------------------------------
# MMR re-ranking
# -------------------------------------------------------------
def mmr(rows: List[Tuple[str, str, str, List[float]]],
        query_embedding: List[float],
        k: int = 8,
        lambda_mult: float = 0.5):
    """Maximal Marginal Relevance (reduces redundancy)."""
    if not rows:
        return []

    E = np.stack([np.array(r[3], dtype=float) for r in rows])  # (n, d)
    q = np.array(query_embedding, dtype=float)                 # (d,)

    # relevance (cosine to query)
    rel = (E @ q) / ((np.linalg.norm(E, axis=1) * np.linalg.norm(q)) + 1e-12)

    picked: List[int] = []
    candidates = list(range(len(rows)))

    while candidates and len(picked) < min(k, len(rows)):
        if not picked:
            next_i = int(np.argmax(rel))
        else:
            # diversity: max cosine to already picked
            E_picked = E[picked]  # (m, d)
            # cosine(E[i], E_picked) -> take max
            def max_div(i: int) -> float:
                v = E[i]
                v_n = v / (np.linalg.norm(v) + 1e-12)
                Pn = E_picked / (np.linalg.norm(E_picked, axis=1, keepdims=True) + 1e-12)
                return float(np.max(Pn @ v_n))
            mmr_scores = [
                (lambda_mult * rel[i]) - ((1 - lambda_mult) * max_div(i))
                for i in candidates
            ]
            next_i = candidates[int(np.argmax(mmr_scores))]
        picked.append(next_i)
        candidates.remove(next_i)

    return [rows[i] for i in picked]


# -------------------------------------------------------------
# Vector search + robust fallback + MMR
# -------------------------------------------------------------
def search_mmr(db: Session,
               query_embedding: List[float],
               k: int = 8,
               pool: int = 24,
               lambda_mult: float = 0.5):
    """
    1) Try pgvector ANN: ORDER BY embedding <=> q
    2) If it returns 0 rows (for any reason), fall back to pulling up to 1000 rows
       and computing cosine in Python, then pick top `pool`.
    3) Apply MMR to get `k`.
    Returns: [(text, title, url, embedding_list)]
    """
    if not query_embedding:
        return []

    # --- Attempt 1: pgvector nearest neighbors
    qvec = _to_vector_literal(query_embedding)
    sql = sql_text("""
        SELECT
          c.text,
          d.title,
          d.url,
          c.embedding
        FROM chunks c
        JOIN documents d ON d.id = c.document_id
        WHERE c.embedding IS NOT NULL
        ORDER BY c.embedding <=> (:qvec)::vector(768)
        LIMIT :pool;
    """)
    rows = db.execute(sql, {"qvec": qvec, "pool": int(pool)}).fetchall()

    if not rows:
        # --- Attempt 2: Python scoring fallback (no ORDER BY in SQL)
        fallback_sql = sql_text("""
            SELECT c.text, d.title, d.url, c.embedding
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            WHERE c.embedding IS NOT NULL
            LIMIT 1000;
        """)
        raw = db.execute(fallback_sql).fetchall()
        q = np.array(query_embedding, dtype=float)
        scored = []
        for t, title, url, emb in raw:
            e = _coerce_embedding(emb)
            if e is None:
                continue
            e_np = np.array(e, dtype=float)
            score = _cos(e_np, q)
            scored.append((score, (t, title or "", url or "", e)))
        scored.sort(key=lambda x: x[0], reverse=True)  # high cosine first
        rows = [r for _, r in scored[:pool]]

    else:
        # Coerce embeddings for pgvector path
        clean = []
        for t, title, url, emb in rows:
            e = _coerce_embedding(emb)
            if e is None:
                continue
            clean.append((t, title or "", url or "", e))
        rows = clean

    if not rows:
        return []

    # --- MMR selection on the candidate pool
    return mmr(rows, query_embedding, k=k, lambda_mult=lambda_mult)
