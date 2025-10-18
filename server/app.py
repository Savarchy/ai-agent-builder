# --- imports ---
from dotenv import load_dotenv
from pathlib import Path
# Force-load the project root .env and override any OS env vars
from uuid import uuid4
import os, io, time, jwt, requests
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from sqlalchemy.orm import Session
from sqlalchemy import create_engine, text
from sqlalchemy import text as sql_text
from bs4 import BeautifulSoup
from .llm import stream_chat                       # keep your current LLM streamer

# local imports (your existing modules)
from .db import Base, engine, get_db
from .models import Document, Chunk
from .schemas import IngestTextIn, IngestURLIn, AskIn, AskOut, Citation
from .chunker import split_text
from .embedder import embed_texts, embed_query     # keep your current embedder
from .retriever import search_mmr                  # uses pgvector + mmr
from .prompting import build_prompt
from .db import engine

print("DB USER AT RUNTIME:", engine.url.username)  # TEMP: remove after verifying


# --- FastAPI app ---
app = FastAPI(
    title="AI Agent Builder — Quickstart",
    debug=True,
    swagger_ui_parameters={"persistAuthorization": True},
)

# --- OpenAPI with API key button (for /docs try-it-out) ---
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title=app.title,
        version="0.1.0",
        description="AI Agent Builder — Quickstart",
        routes=app.routes,
    )
    components = openapi_schema.setdefault("components", {})
    security_schemes = components.setdefault("securitySchemes", {})
    security_schemes["ApiKeyAuth"] = {
        "type": "apiKey",
        "in": "header",
        "name": "x-api-key",
        "description": "Enter your API key (e.g., dev-secret-123)",
    }
    for path_item in openapi_schema.get("paths", {}).values():
        for operation in list(path_item.values()):
            if isinstance(operation, dict):
                operation.setdefault("security", [{"ApiKeyAuth": []}])
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

# --- CORS ---
ALLOWED_ORIGINS = [
    "http://127.0.0.1:5500",
    "http://localhost:5500",
    "http://127.0.0.1:8000",
    "http://localhost:8000",
]
WIDGET_ORIGIN = os.getenv("WIDGET_ORIGIN")
if WIDGET_ORIGIN and WIDGET_ORIGIN not in ALLOWED_ORIGINS:
    ALLOWED_ORIGINS.append(WIDGET_ORIGIN)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS if ALLOWED_ORIGINS else ["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Type", "Cache-Control", "X-Requested-With"],
)

# --- Auth config ---
API_KEY = os.getenv("API_KEY", "dev-secret-123")
SITE_JWT_SECRET = os.getenv("SITE_JWT_SECRET", "change_me")

def issue_site_token(bot_id: str, domain: str | None = None, ttl_minutes: int = 1440):
    now = int(time.time())
    payload = {"sub": bot_id, "typ": "site", "iat": now, "exp": now + ttl_minutes * 60}
    if domain:
        payload["dom"] = domain
    return jwt.encode(payload, SITE_JWT_SECRET, algorithm="HS256")

def verify_site_bearer_header(auth_header: str | None):
    """Return decoded payload if valid 'Bearer <jwt>' for site tokens, else None/raise."""
    if not auth_header or not auth_header.startswith("Bearer "):
        return None
    token = auth_header.split(" ", 1)[1].strip()
    try:
        payload = jwt.decode(token, SITE_JWT_SECRET, algorithms=["HS256"])
        if payload.get("typ") != "site":
            raise HTTPException(status_code=401, detail="Invalid token type")
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token")

# --- Gate: allow either valid Bearer (site) OR x-api-key ---
@app.middleware("http")
async def auth_gate(request: Request, call_next):
    # Always allow CORS preflight & health/docs/openapi
    if request.method == "OPTIONS":
        return await call_next(request)
    if request.url.path.startswith(("/docs", "/openapi.json", "/health")):
        return await call_next(request)

    # 1) Try Bearer site token
    auth_header = request.headers.get("authorization") or request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        _payload = verify_site_bearer_header(auth_header)  # raises on invalid
        # valid → let it through
        return await call_next(request)

    # 2) Fallback to x-api-key (admin/dev)
    key = request.headers.get("x-api-key")
    if API_KEY and key == API_KEY:
        return await call_next(request)

    # Neither provided/valid
    return JSONResponse(status_code=401, content={"detail": "Unauthorized"})

# --- Simple logging (one clean middleware) ---
@app.middleware("http")
async def log_requests(request: Request, call_next):
    print(f">>> {request.method} {request.url.path}")
    try:
        resp = await call_next(request)
        print(f"<<< {resp.status_code} {request.url.path}")
        return resp
    except Exception as e:
        print(f"!!! {request.method} {request.url.path} -> {e}")
        raise

# --- Optional debug exception JSON ---
if os.getenv("APP_DEBUG", "1") == "1":
    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        import traceback
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        print("TRACEBACK:\n", tb)
        return JSONResponse(status_code=500, content={"detail": str(exc), "trace": tb})

# -----------------------------------------------------------------------------
# STARTUP: ensure pgvector extension, then tables & index
# -----------------------------------------------------------------------------
@app.on_event("startup")
def _startup():
    # 1) ensure extension exists
    with engine.connect() as conn:
        conn.exec_driver_sql("CREATE EXTENSION IF NOT EXISTS vector;")

    # 2) create tables now that 'vector' type exists
    Base.metadata.create_all(bind=engine)

    # 3) ensure ANN index (ivfflat) on the vector column
    with engine.begin() as conn:
        try:
            conn.exec_driver_sql("""
                CREATE INDEX IF NOT EXISTS ix_chunks_embedding_cosine
                ON chunks USING ivfflat (embedding vector_cosine_ops);
            """)
        except Exception as e:
            print("Index ensure error:", e)
        

# --- Health ---
@app.get("/health/db")
def health_db():
    with engine.connect() as c:
        c.execute(text("select 1"))
    return {"ok": True}

# --- Dev-only clear (guarded by APP_DEBUG) ---
if os.getenv("APP_DEBUG", "1") == "1":
    @app.post("/_dev/clear")
    def dev_clear(db: Session = Depends(get_db)):
        db.execute(sql_text("DELETE FROM chunks;"))
        db.execute(sql_text("DELETE FROM documents;"))
        db.commit()
        return {"ok": True}

# --- Bot profiles & /bots ---
BOT_PROFILES = {
    "default": {"name": "General", "system_prefix": ""},
    "friendly": {"name": "Friendly", "system_prefix": "You are a warm, upbeat assistant for a retail brand."},
    "terse": {"name": "Terse", "system_prefix": "Answer concisely. No fluff."},
    "Aussie mate": {"name": "Aussie", "system_prefix": "Answer concisely. Australian English."},
}

@app.get("/bots")
def list_bots():
    return [{"id": k, "name": v["name"]} for k, v in BOT_PROFILES.items()]

# --- Mint site token (admin/dev via x-api-key) ---
@app.post("/bots/{bot_id}/site-token")
def mint_site_token(bot_id: str, body: dict, request: Request):
    # The auth_gate already validated, but explicitly restrict to x-api-key here if you want:
    # (If you want Bearer callers to be able to mint, remove this check.)
    if request.headers.get("x-api-key") != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")
    domain = (body or {}).get("domain")
    ttl = int((body or {}).get("ttl_minutes", 60))
    return {"token": issue_site_token(bot_id, domain, ttl)}

# --- INGEST: TEXT ---
@app.post("/ingest/text")
def ingest_text(payload: IngestTextIn, db: Session = Depends(get_db)):
    try:
        text_in = (payload.text or "").strip()
        if not text_in:
            raise HTTPException(status_code=400, detail="No text provided.")
        url_str = str(payload.url) if getattr(payload, "url", None) else None

        # 1) Document (let DB generate UUID; note 'text' not 'content')
        doc = Document(title=payload.title, url=url_str, text=text_in)
        db.add(doc)
        db.commit()
        db.refresh(doc)  # now doc.id is a real UUID from DB

        # 2) Chunking
        parts = split_text(text_in)
        if parts is None:
            parts = [text_in]
        elif isinstance(parts, tuple):
            parts = parts[0] or [text_in]
        if not parts:
            parts = [text_in]

        # 3) Embeddings
        embs = embed_texts(parts)
        if embs is None or len(embs) != len(parts):
            embs = []
            for p in parts:
                one = embed_texts([p])
                if not one or len(one) != 1:
                    raise RuntimeError("Embedding failed for a chunk")
                embs.append(one[0])

        # 4) Write chunks (also let DB generate UUIDs)
        for i, (p, emb) in enumerate(zip(parts, embs)):
            db.add(Chunk(
                document_id=doc.id,
                ord=i,
                text=p,
                embedding=emb,
            ))
        db.commit()
        return {"document_id": str(doc.id), "chunks": len(parts)}

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"/ingest/text failed: {e}")


# --- INGEST: URL ---
# -----------------------------------------------------------------------------
# INGEST: URL (with fetch & text size guard)
# -----------------------------------------------------------------------------
MAX_FETCH_MB = float(os.getenv("MAX_FETCH_MB", "3"))
MAX_FETCH_CHARS = int(MAX_FETCH_MB * 1024 * 1024)  # approx chars ~= bytes
REQUEST_TIMEOUT_SECONDS = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "30"))

@app.post("/ingest/url")
def ingest_url(payload: IngestURLIn, db: Session = Depends(get_db)):
    # Fetch page
    try:
        r = requests.get(
            str(payload.url),
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"
            },
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        r.raise_for_status()
    except requests.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Fetch failed: {e}")

    # Extract text
    soup = BeautifulSoup(r.text, "html.parser")
    for s in soup(["script", "style", "noscript", "svg"]):
        s.decompose()
    text = soup.get_text(separator="\n", strip=True)
    if not text:
        raise HTTPException(status_code=400, detail="No extractable text.")

    # Guard: cap the size of text we process
    if len(text) > MAX_FETCH_CHARS:
        text = text[:MAX_FETCH_CHARS]
        text += f"\n\n[Note: content truncated at ~{MAX_FETCH_MB} MB for processing.]"

    title = payload.title or (soup.title.get_text(strip=True) if soup.title else str(payload.url))
    return ingest_text(IngestTextIn(title=title, text=text, url=str(payload.url)), db)

# --- INGEST: PDF ---
# -----------------------------------------------------------------------------
# INGEST: PDF (with size & pages limits)
# -----------------------------------------------------------------------------
from fastapi import UploadFile, File, Form

MAX_PDF_MB = int(os.getenv("MAX_PDF_MB", "10"))
MAX_PDF_BYTES = MAX_PDF_MB * 1024 * 1024
MAX_PDF_PAGES = int(os.getenv("MAX_PDF_PAGES", "200"))

@app.post("/ingest/pdf")
async def ingest_pdf(
    file: UploadFile = File(...),
    title: str | None = Form(None),
    url: str | None = Form(None),
    db: Session = Depends(get_db),
):
    # Basic content-type/extension check
    fname = (file.filename or "").lower()
    ctype = (file.content_type or "").lower()
    if not (fname.endswith(".pdf") or "pdf" in ctype):
        raise HTTPException(status_code=400, detail="Please upload a PDF file.")

    # Read with size cap (+1 byte to detect overflow)
    content = await file.read(MAX_PDF_BYTES + 1)
    if len(content) > MAX_PDF_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"PDF too large: limit is {MAX_PDF_MB} MB."
        )

    # Parse PDF text (cap pages)
    import pdfplumber
    text_all = []
    try:
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            page_limit = min(len(pdf.pages), MAX_PDF_PAGES)
            for i in range(page_limit):
                p = pdf.pages[i]
                t = p.extract_text() or ""
                if t.strip():
                    text_all.append(t)
            if len(pdf.pages) > MAX_PDF_PAGES:
                text_all.append(
                    f"\n\n[Note: PDF truncated at {MAX_PDF_PAGES} pages for processing.]"
                )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"PDF parse failed: {e}")

    full_text = "\n\n".join(text_all).strip()
    if not full_text:
        raise HTTPException(status_code=400, detail="No text found in PDF.")

    return ingest_text(IngestTextIn(title=title or file.filename, text=full_text, url=url), db)

# --- ASK (non-stream) ---
@app.post("/ask", response_model=AskOut)
async def ask(payload: AskIn, db: Session = Depends(get_db), request: Request = None):
    # Embed the question
    q_emb = embed_query(payload.question)

    # Retrieve candidate chunks
    rows = search_mmr(db, q_emb, k=payload.k or 6, pool=48, lambda_mult=0.7)

    if not rows:
        return AskOut(
            answer="I couldn’t find any matching chunks yet. Try ingesting a source first, or broaden your question.",
            citations=[],
        )

    # Build contexts & citations
    contexts, citations = [], []
    for (text, title, url, _emb) in rows:
        contexts.append((text[:1000], url, 0.0))
        citations.append(Citation(
            title=title or "",
            url=url or "",
            snippet=text[:240].replace("\n", " "),
            score=0.0,
        ))

    # Persona
    persona_key = getattr(payload, "bot_id", None) or "default"
    persona = BOT_PROFILES.get(persona_key)
    persona_prefix = persona["system_prefix"] if persona else ""

    # Prompt
    system, user = build_prompt(payload.question, contexts)
    if persona_prefix:
        system = persona_prefix + "\n\n" + system

    # Consume the async generator into a single string
    chunks = []
    async for tok in stream_chat(system, user):
        chunks.append(tok)
    answer = "".join(chunks)

    return AskOut(answer=answer, citations=citations[:5])


# --- ASK (streaming SSE) ---
@app.post("/ask/stream")
async def ask_stream(payload: AskIn, request: Request, db: Session = Depends(get_db)):
    try:
        # If a Bearer token is present, validate it (dev with x-api-key still passes middleware)
        _ = request.headers.get("authorization") or request.headers.get("Authorization")
        if _:
            _ = verify_site_bearer_header(_)

        # Embed the question
        q_emb = embed_query(payload.question)

        # Retrieve candidate chunks
        rows = search_mmr(db, q_emb, k=payload.k or 6, pool=48, lambda_mult=0.7)
        print("[ask/stream] candidates:", [{"title": t, "url": u} for (_txt, t, u, _emb) in rows])

        if not rows:
            async def err():
                yield "event: start\ndata: {}\n\n"
                yield "data: I couldn’t find any matching chunks yet. Try ingesting a source first, or broaden your question.\n\n"
                yield "event: end\ndata: {}\n\n"
            return StreamingResponse(err(), media_type="text/event-stream")

        # Build contexts & persona
        contexts = [(text[:1000], url, 0.0) for (text, title, url, _emb) in rows]
        persona_key = getattr(payload, "bot_id", None) or "default"
        persona = BOT_PROFILES.get(persona_key)
        persona_prefix = persona["system_prefix"] if persona else ""

        system, user = build_prompt(payload.question, contexts)
        if persona_prefix:
            system = persona_prefix + "\n\n" + system

        async def gen():
            yield "event: start\ndata: {}\n\n"
            # IMPORTANT: async generator -> use async for
            async for tok in stream_chat(system, user):
                yield f"data: {tok}\n\n"
            yield "event: end\ndata: {}\n\n"

        return StreamingResponse(
            gen(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    except Exception as e:
        import traceback
        print("ask_stream error:", e)
        print(traceback.format_exc())

        async def err():
            yield "event: error\ndata: {}\n\n"

        return StreamingResponse(err(), media_type="text/event-stream")
