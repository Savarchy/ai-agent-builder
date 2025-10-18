# server/retriever.py
import os
from typing import List, Tuple
from sqlalchemy import text
from sqlalchemy.orm import Session

# Use the same dimension as your embedder / DB column
EMBED_DIM = int(os.getenv("EMBED_DIM", "1536"))

def _to_pgvector_literal(vec: List[float]) -> str:
    """
    Convert a Python list of floats to a pgvector literal, e.g.:
    [0.1, 0.2] -> "[0.1,0.2]"
    """
    # Keep the numbers as standard floats; pgvector parses them fine.
    return "[" + ",".join(str(float(x)) for x in vec) + "]"

def _cosine(a: List[float], b: List[float]) -> float:
    """Simple cosine similarity for MMR (fallback if needed)."""
    import math
    num = sum(x * y for x, y in zip(a, b))
    da = math.sqrt(sum(x * x for x in a))
    db = math.sqrt(sum(y * y for y in b))
    if da == 0 or db == 0:
        return 0.0
    return num / (da * db)

def _mmr(
    query_vec: List[float],
    cand_vecs: List[List[float]],
    k: int = 6,
    lambda_mult: float = 0.7,
) -> List[int]:
    """
    Maximal Marginal Relevance (indices into cand_vecs).
    """
    if not cand_vecs:
        return []

    selected = []
    remaining = list(range(len(cand_vecs)))

    # precompute sims to query
    sims_to_q = [ _cosine(query_vec, v) for v in cand_vecs ]

    while remaining and len(selected) < k:
        best_idx = None
        best_score = -1e9
        for i in remaining:
            # diversity term: max sim to already selected
            if selected:
                max_sim_sel = max(_cosine(cand_vecs[i], cand_vecs[j]) for j in selected)
            else:
                max_sim_sel = 0.0
            score = lambda_mult * sims_to_q[i] - (1.0 - lambda_mult) * max_sim_sel
            if score > best_score:
                best_score = score
                best_idx = i

        selected.append(best_idx)
        remaining.remove(best_idx)

    return selected

def search_mmr(
    db: Session,
    qvec: List[float],
    k: int = 6,
    pool: int = 48,
    lambda_mult: float = 0.7,
) -> List[Tuple[str, str, str, List[float]]]:
    """
    1) Pull top-N (pool) by ANN distance using pgvector
    2) Rerank with MMR
    Returns list of tuples: (chunk_text, doc_title, doc_url, chunk_embedding)
    """
    # Convert query vector to pgvector literal and bind with :qvec
    qvec_lit = _to_pgvector_literal(qvec)

    sql = text(f"""
        SELECT
          c.text,
          d.title,
          d.url,
          c.embedding
        FROM chunks c
        JOIN documents d ON d.id = c.document_id
        WHERE c.embedding IS NOT NULL
        ORDER BY c.embedding <=> (:qvec)::vector({EMBED_DIM})
        LIMIT :pool
    """)

    rows = db.execute(sql, {"qvec": qvec_lit, "pool": int(pool)}).fetchall()
    if not rows:
        return []

    # rows: [(text, title, url, embedding), ...]
    cand_vecs = [r[3] for r in rows]  # embeddings
    keep_idx = _mmr(qvec, cand_vecs, k=k, lambda_mult=lambda_mult)

    # preserve original order of selected indices
    result = [rows[i] for i in keep_idx]
    return result
