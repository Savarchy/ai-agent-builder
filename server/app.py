# server/app.py

# --- stdlib / third-party imports ---
import os, io, time, jwt, requests
from uuid import uuid4
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from sqlalchemy.orm import Session
from sqlalchemy import text as sql_text, text

# --- local imports ---
from .db import Base, engine, get_db
from .models import Document, Chunk
from .schemas import IngestTextIn, IngestURLIn, AskIn, AskOut, Citation
from .chunker import split_text
from .embedder import embed_texts, embed_query
from .retriever import search_mmr
from .prompting import build_prompt
from .llm import stream_chat

# Load .env early
load_dotenv()


# -----------------------------------------------------------------------------
# PgVector bootstrap that works on small DB plans
# -----------------------------------------------------------------------------
def ensure_vector_schema(engine):
    """
    Makes schema safe for pgvector on small plans.
    - CREATE EXTENSION in AUTOCOMMIT
    - ALTER COLUMN -> vector(1536)
    - Try to create small IVFFLAT index, but never crash if it fails
    """
    # 1) CREATE EXTENSION must be outside a transaction.
    try:
        with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
            conn.exec_driver_sql("CREATE EXTENSION IF NOT EXISTS vector;")
            print("pgvector extension ensured")
    except Exception as e:
        # Do not crash on extension error; just log it.
        print(f"Extension ensure error (ignored): {e}")

    # 2) Ensure column is correct type/dimensions.
    try:
        with engine.begin() as conn:
            conn.exec_driver_sql("""
                ALTER TABLE chunks
                ALTER COLUMN embedding TYPE vector(1536);
            """)
            print("chunks.embedding is vector(1536)")
    except Exception as e:
        # It's OK if table/index doesn't exist yet or already correct.
        print(f"Column alter error (ignored): {e}")

    # 3) Optional: ANN index. Safe default is to skip on tiny plans.
    if os.getenv("SKIP_IVFFLAT", "1") == "1":
        print("SKIP_IVFFLAT=1 -> not creating IVFFLAT index (using full-scan)")
        return

    lists = int(os.getenv("IVFFLAT_LISTS", "4"))  # keep small on free tiers
    try:
        with engine.begin() as conn:
            conn.exec_driver_sql("DROP INDEX IF EXISTS ix_chunks_embedding_cosine;")
            conn.exec_driver_sql(f"""
                CREATE INDEX IF NOT EXISTS ix_chunks_embedding_cosine
                ON chunks USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = {lists});
            """)
            print(f"IVFFLAT index ensured with lists={lists}")
    except Exception as e:
        # Just log and continue; the app must still start.
        print(f"Index ensure error (ignored): {e}")

# -----------------------------------------------------------------------------
# FastAPI app
# -----------------------------------------------------------------------------
app = FastAPI(
    title="AI Agent Builder — Quickstart",
    debug=True,
    swagger_ui_parameters={"persistAuthorization": True},
)
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from pathlib import Path

# Serve /ui/* from server/static, and redirect "/" -> /ui/
app.mount("/ui", StaticFiles(directory=Path(__file__).parent / "static", html=True), name="ui")

@app.get("/")
def _home():
    return RedirectResponse("/ui/")
    
@app.get("/favicon.ico")
def favicon():
    # 1x1 transparent png
    import base64
    png = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAEElEQVR4nGP4////fwYGBgYGAAApPwKQ8O7D8wAAAABJRU5ErkJggg=="
    )
    from fastapi.responses import Response
    return Response(png, media_type="image/png")
    
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
    """Return decoded payload if valid 'Bearer <jwt>' for site tokens, else raise 401."""
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
PUBLIC_EXACT = {"/", "/openapi.json", "/health", "/health/db", "/favicon.ico"}
PUBLIC_PREFIXES = ("/ui", "/docs", "/static")  # everything under these is public

@app.middleware("http")
async def auth_gate(request: Request, call_next):
    # Always allow CORS preflight
    if request.method == "OPTIONS":
        return await call_next(request)

    path = request.url.path or "/"
    if path in PUBLIC_EXACT or any(path.startswith(p) for p in PUBLIC_PREFIXES):
        return await call_next(request)

    # 1) Try Bearer site token
    auth_header = request.headers.get("authorization") or request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        _ = verify_site_bearer_header(auth_header)  # raises on invalid
        return await call_next(request)

    # 2) Fallback to x-api-key (admin/dev)
    key = request.headers.get("x-api-key")
    if API_KEY and key == API_KEY:
        return await call_next(request)

    return JSONResponse(status_code=401, content={"detail": "Unauthorized"})

    if request.url.path.startswith(("/docs", "/openapi.json", "/health")):
        return await call_next(request)

    # 1) Try Bearer site token
    auth_header = request.headers.get("authorization") or request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        _ = verify_site_bearer_header(auth_header)  # raises on invalid
        return await call_next(request)

    # 2) Fallback to x-api-key (admin/dev)
    key = request.headers.get("x-api-key")
    if API_KEY and key == API_KEY:
        return await call_next(request)

    # Neither provided/valid
    return JSONResponse(status_code=401, content={"detail": "Unauthorized"})


# --- Simple logging ---
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
# STARTUP: ensure pgvector extension, then tables & (optional) index
# -----------------------------------------------------------------------------
@app.on_event("startup")
def _startup() -> None:
    # IMPORTANT: pass the ENGINE (not a Connection/Session/Transaction)
    ensure_vector_schema(engine)

    # Create tables after pgvector/type/index bootstrap
    Base.metadata.create_all(bind=engine)

    print("Startup bootstrap complete")


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
    if request.headers.get("x-api-key") != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")
    domain = (body or {}).get("domain")
    ttl = int((body or {}).get("ttl_minutes", 60))
    return {"token": issue_site_token(bot_id, domain, ttl)}


# -----------------------------------------------------------------------------
# INGEST: TEXT
# -----------------------------------------------------------------------------
@app.post("/ingest/text")
def ingest_text(payload: IngestTextIn, db: Session = Depends(get_db)):
    try:
        text_in = (payload.text or "").strip()
        if not text_in:
            raise HTTPException(status_code=400, detail="No text provided.")
        url_str = str(payload.url) if getattr(payload, "url", None) else None

        # 1) Document (let DB generate UUID; NOTE: uses Document.text)
        doc = Document(title=payload.title, url=url_str, text=text_in)  # <-- uses Document.text
        db.add(doc)
        db.commit()
        db.refresh(doc)  # doc.id is now set

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

        # 4) Write chunks (let DB generate UUIDs)
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


# -----------------------------------------------------------------------------
# INGEST: URL (with strong headers + optional Jina Reader fallback)
# -----------------------------------------------------------------------------
from urllib.parse import urlparse, urlunparse

MAX_FETCH_MB = float(os.getenv("MAX_FETCH_MB", "3"))
MAX_FETCH_CHARS = int(MAX_FETCH_MB * 1024 * 1024)  # approx chars ~= bytes
REQUEST_TIMEOUT_SECONDS = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "30"))
JINA_READER_FALLBACK = os.getenv("JINA_READER_FALLBACK", "0") == "1"

def _normalize_http_url(u: str) -> str:
    u = (u or "").strip()
    if not u:
        raise HTTPException(status_code=400, detail="URL is required.")
    if not u.lower().startswith(("http://", "https://")):
        u = "https://" + u
    p = urlparse(u)
    if not p.netloc:
        raise HTTPException(status_code=400, detail="Invalid URL.")
    return urlunparse(p)

def _nice_fetch(url: str) -> requests.Response:
    # Beefier headers to look like a real browser
    hdrs = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": (
            "text/html,application/xhtml+xml,application/xml;"
            "q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8"
        ),
        "Accept-Language": "en-US,en;q=0.9",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
        "Referer": url,
    }
    return requests.get(
        url,
        headers=hdrs,
        timeout=REQUEST_TIMEOUT_SECONDS,
        allow_redirects=True,
    )

def _jina_reader_url(orig: str) -> str:
    # Jina Reader pattern: https://r.jina.ai/http://{host}{path}?{query}
    # We always pass http://host...; most sites redirect to https server-side.
    p = urlparse(orig)
    path = p.path or "/"
    if p.query:
        path = f"{path}?{p.query}"
    return f"https://r.jina.ai/http://{p.netloc}{path}"

@app.post("/ingest/url")
def ingest_url(payload: IngestURLIn, db: Session = Depends(get_db)):
    # 0) Normalize URL
    target_url = _normalize_http_url(str(payload.url))

    # 1) Try direct fetch
    try:
        r = _nice_fetch(target_url)
        if r.status_code >= 400:
            # If blocked, optionally try fallback
            if r.status_code in (401, 403, 451) and JINA_READER_FALLBACK:
                try:
                    jurl = _jina_reader_url(target_url)
                    jr = _nice_fetch(jurl)
                    jr.raise_for_status()
                    text_body = jr.text or ""
                    text_body = text_body.strip()
                    if not text_body:
                        raise HTTPException(
                            status_code=400,
                            detail=f"Reader fallback returned no content for url: {target_url}",
                        )
                    # Guard & pass to ingest_text
                    if len(text_body) > MAX_FETCH_CHARS:
                        text_body = text_body[:MAX_FETCH_CHARS] + (
                            f"\n\n[Note: content truncated at ~{MAX_FETCH_MB} MB for processing.]"
                        )
                    title = payload.title or target_url
                    return ingest_text(
                        IngestTextIn(title=title, text=text_body, url=target_url),
                        db,
                    )
                except requests.RequestException as e:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Blocked by site (status {r.status_code}). Fallback failed: {e}",
                    )
            # No fallback, bubble a clear error
            raise HTTPException(
                status_code=400,
                detail=f"Fetch failed: {r.status_code} Client Error: "
                       f"{r.reason or 'Error'} for url: {target_url}",
            )
        # OK
        html = r.text or ""
    except requests.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Fetch failed: {e}")

    # 2) Extract text (basic HTML to text)
    soup = BeautifulSoup(html, "html.parser")
    for s in soup(["script", "style", "noscript", "svg"]):
        s.decompose()
    text_body = soup.get_text(separator="\n", strip=True)
    if not text_body:
        raise HTTPException(status_code=400, detail="No extractable text.")

    # 3) Guard: cap the size
    if len(text_body) > MAX_FETCH_CHARS:
        text_body = text_body[:MAX_FETCH_CHARS]
        text_body += f"\n\n[Note: content truncated at ~{MAX_FETCH_MB} MB for processing.]"

    # 4) Title
    title = payload.title or (soup.title.get_text(strip=True) if soup.title else target_url)

    # 5) Hand off to text ingest (chunk + embed)
    return ingest_text(IngestTextIn(title=title, text=text_body, url=target_url), db)



# -----------------------------------------------------------------------------
# INGEST: PDF (robust: size cap, page cap, magic header, dual parser)
# -----------------------------------------------------------------------------
import io as _io
from urllib.parse import urlparse

# -----------------------------------------------------------------------------
# INGEST: PDF (defensive, clearer errors, bigger size allowed)
# -----------------------------------------------------------------------------
from fastapi import UploadFile, File, Form

# Allow up to 50 MB uploads (Render/Cloudflare hard limit is typically 100 MB)
MAX_PDF_MB = int(os.getenv("MAX_PDF_MB", "50"))
MAX_PDF_BYTES = MAX_PDF_MB * 1024 * 1024

# Cap pages we actually parse (prevents super-long runs)
MAX_PDF_PAGES = int(os.getenv("MAX_PDF_PAGES", "200"))

# Even if the PDF is big, cap text we send to chunk/embeddings (MB of raw text)
# ~1 char ≈ 1 byte (roughly), so 2 MB of text is already a LOT of tokens.
MAX_PDF_TEXT_MB = float(os.getenv("MAX_PDF_TEXT_MB", "2.5"))
MAX_PDF_TEXT_CHARS = int(MAX_PDF_TEXT_MB * 1024 * 1024)

@app.post("/ingest/pdf")
async def ingest_pdf(
    file: UploadFile = File(...),
    title: str | None = Form(None),
    url: str | None = Form(None),
    db: Session = Depends(get_db),
):
    """
    Accept a PDF (multipart/form-data), extract text safely, trim it,
    then reuse /ingest/text. Returns 4xx on user issues instead of 502 crashes.
    """
    try:
        # --- 0) Basic validation
        fname = (file.filename or "").lower()
        ctype = (file.content_type or "").lower()
        if not (fname.endswith(".pdf") or "pdf" in ctype):
            raise HTTPException(status_code=400, detail="Please upload a PDF file (.pdf).")

        # --- 1) Size guard (prevent OOM)
        blob = await file.read(MAX_PDF_BYTES + 1)  # +1 lets us detect overflow
        if len(blob) > MAX_PDF_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"PDF too large. Limit is {MAX_PDF_MB} MB."
            )

        # --- 2) Parse text with page cap
        import pdfplumber, io
        pages_text: list[str] = []
        try:
            with pdfplumber.open(io.BytesIO(blob)) as pdf:
                page_limit = min(len(pdf.pages), MAX_PDF_PAGES)
                for i in range(page_limit):
                    p = pdf.pages[i]
                    t = (p.extract_text() or "").strip()
                    if t:
                        pages_text.append(t)
                if len(pdf.pages) > MAX_PDF_PAGES:
                    pages_text.append(
                        f"[Note: PDF truncated at {MAX_PDF_PAGES} pages for processing.]"
                    )
        except Exception as e:
            # Parsing issues (encrypted/corrupt) → 400
            raise HTTPException(status_code=400, detail=f"PDF parse failed: {e}")

        full_text = "\n\n".join(pages_text).strip()
        if not full_text:
            raise HTTPException(status_code=400, detail="No selectable text found in PDF.")

        # --- 3) Text-size cap before chunking/embeddings (prevents huge bills/timeouts)
        if len(full_text) > MAX_PDF_TEXT_CHARS:
            full_text = full_text[:MAX_PDF_TEXT_CHARS] + \
                f"\n\n[Note: content truncated at ~{MAX_PDF_TEXT_MB} MB of text for processing.]"

        # --- 4) Reuse /ingest/text path (this will chunk + embed)
        safe_title = title or (file.filename or "PDF")
        return ingest_text(IngestTextIn(title=safe_title, text=full_text, url=url), db)

    except HTTPException:
        # pass through explicit 4xx
        raise
    except Exception as e:
        # Anything unexpected → 500 with a clear message (and your APP_DEBUG logs the traceback)
        raise HTTPException(status_code=500, detail=f"/ingest/pdf failed: {e}")


# -----------------------------------------------------------------------------
# ASK (non-stream)
# -----------------------------------------------------------------------------
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
    for (text_val, title, url, _emb) in rows:
        contexts.append((text_val[:1000], url, 0.0))
        citations.append(Citation(
            title=title or "",
            url=url or "",
            snippet=text_val[:240].replace("\n", " "),
            score=0.0,
        ))

    # Persona
    persona_key = getattr(payload, "bot_id", None) or "default"
    persona = BOT_PROFILES.get(persona_key)
    persona_prefix = persona["system_prefix"] if persona else ""

    # Prompt
    ENQUIRY_URL = os.getenv("ENQUIRY_URL", "")  # e.g., https://tally.so/r/your-form-id
    ...
    system, user = build_prompt(payload.question, contexts, cta_url=ENQUIRY_URL or None)
    if persona_prefix:
        system = persona_prefix + "\n\n" + system

    # Consume the async generator into a single string
    chunks = []
    async for tok in stream_chat(system, user):
        chunks.append(tok)
    answer = "".join(chunks)

    return AskOut(answer=answer, citations=citations[:5])


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

        # Build contexts & persona
        contexts = [(text[:1000], url, 0.0) for (text, title, url, _emb) in rows]
        persona_key = getattr(payload, "bot_id", None) or "default"
        persona = BOT_PROFILES.get(persona_key)
        persona_prefix = persona["system_prefix"] if persona else ""

        system, user = build_prompt(payload.question, contexts)
        if persona_prefix:
            system = persona_prefix + "\n\n" + system

        # Prepare citation links we will stream at the end
        link_list = []
        for (_text, title, url, _emb) in rows:
            if url:
                link_list.append({"title": title or url, "url": url})

        async def gen():
            # handshake
            yield "event: start\ndata: {}\n\n"

            # stream the model text (no links here)
            async for tok in stream_chat(system, user):
                # standard 'message' event (no explicit 'event:' line)
                yield f"data: {tok}\n\n"

            # now send citations separately so UI can render nice clickable links
            if link_list:
                yield "event: citations\n"
                yield f"data: {json.dumps(link_list)}\n\n"

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
