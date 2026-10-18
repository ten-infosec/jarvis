# ===== Jarvis RAG API =====
# Local:  uvicorn app.main:app                      (private data, e5 + Ollama, no limits)
# Cloud:  $env:JARVIS_MODE="cloud"; uvicorn app.main:app   (public arXiv + Gemini, rate limited)


import json
import os
import threading
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field


# ===== Engine selection =====
MODE = os.getenv("JARVIS_MODE", "local")
if MODE == "cloud":
    from app.rag_cloud import CloudRAG as Engine
else:
    from app.rag import RAG as Engine

# In cloud mode, hide the auto-generated API docs (/docs, /openapi.json) from the public
_docs_off = {"docs_url": None, "redoc_url": None, "openapi_url": None} if MODE == "cloud" else {}
app = FastAPI(title=f"Jarvis RAG ({MODE})", **_docs_off)
rag = Engine()
STATIC_DIR = Path(__file__).parent / "static"
KST = timezone(timedelta(hours=9))

# ===== Book library engine (local mode only) =====
# Created on first use: the server starts fast, and the e5 model of the main engine is shared, not loaded twice
_library = None
_library_lock = threading.Lock()


def get_library():
    global _library
    if MODE == "cloud":
        # Private books never exist in the cloud: behave as if the feature is absent
        raise HTTPException(status_code=404, detail="Not found")
    with _library_lock:
        if _library is None:
            from app.library import LibraryRAG, SharedE5
            model = getattr(rag, "model", None)
            try:
                _library = LibraryRAG(embedder=SharedE5(model) if model is not None else None)
            except RuntimeError as e:
                # e.g. the keyword index was not built yet: a clear message instead of a stack trace
                raise HTTPException(status_code=503, detail=str(e))
        return _library

# ===== Request log =====
LOG_FILE = Path(os.getenv("JARVIS_DATA_DIR", r"C:\instagram_export")) / "logs" / "requests.jsonl"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
_log_lock = threading.Lock()


def write_log(record):
    with _log_lock:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    # Cloud Run keeps whatever is printed to stdout in Cloud Logging (the /tmp file is lost when the server scales to 0).
    # A one-line JSON becomes a structured log: each field can be searched, and "severity" sets the log level.
    if MODE == "cloud":
        status = record.get("status")
        severity = "ERROR" if status == "error" else ("WARNING" if status == "rejected" else "INFO")
        print(json.dumps({"severity": severity, "message": f"ask {status}", **record}, ensure_ascii=False), flush=True)

# ===== Rate limits (cloud mode only) =====
# Values can be changed with environment variables at deploy time
RATE_PER_MIN = int(os.getenv("RATE_PER_MIN", "5"))
DAILY_CAP = int(os.getenv("DAILY_CAP", "200"))
# How many trusted proxies append to X-Forwarded-For AFTER the client (0 = direct *.run.app URL)
# max(..., 0): a negative value from a typo cannot break the index math
TRUSTED_PROXY_HOPS = max(int(os.getenv("TRUSTED_PROXY_HOPS", "0")), 0)
# Upper bound on how many IPs are tracked in memory (protects the single instance from memory blow-up)
MAX_TRACKED_IPS = 10_000
# ip -> times of that IP's recent requests (deque = list that is fast to trim from the front)
_recent = defaultdict(deque)
_daily = {"date": None, "count": 0}
_limit_lock = threading.Lock()


def client_ip(request: Request):
    # Each proxy APPENDS to X-Forwarded-For, so a client can only forge entries on the LEFT.
    # The entry added by Google's front end is the rightmost one (minus any extra trusted proxies).
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        parts = [p.strip() for p in forwarded.split(",") if p.strip()]
        # A header made only of commas/spaces gives an empty list -> fall through to the socket address
        if parts:
            index = max(len(parts) - 1 - TRUSTED_PROXY_HOPS, 0)
            return parts[index]
    return request.client.host if request.client else "unknown"


def check_limits(ip):
    now = time.time()
    with _limit_lock:
        # 1) Whole service first: once the daily cap is hit, nothing else is stored in memory
        today = datetime.now(KST).date().isoformat()
        if _daily["date"] != today:
            _daily["date"], _daily["count"] = today, 0
        if _daily["count"] >= DAILY_CAP:
            raise HTTPException(status_code=429, detail="Daily limit reached for this demo. Please try again tomorrow.")

        # 2) Per-IP: look up WITHOUT creating an entry, forget requests older than 60 seconds
        times = _recent.get(ip)
        if times is not None:
            while times and now - times[0] > 60:
                times.popleft()
            if len(times) >= RATE_PER_MIN:
                raise HTTPException(status_code=429, detail=f"Too many requests: max {RATE_PER_MIN} per minute. Please wait.")

        # 3) Keep the IP table bounded: drop idle IPs, and if still too big, drop the oldest ones
        if times is None and len(_recent) >= MAX_TRACKED_IPS:
            for old_ip in [k for k, v in _recent.items() if not v or now - v[-1] > 60]:
                del _recent[old_ip]
            while len(_recent) >= MAX_TRACKED_IPS:
                del _recent[next(iter(_recent))]

        _recent[ip].append(now)
        _daily["count"] += 1


# ===== Request format (3, 4: size limits checked automatically by pydantic -> HTTP 422) =====
class AskRequest(BaseModel):
    question: str = Field(min_length=2, max_length=500)
    # Only the known local sources are accepted (cloud ignores it); blocks huge strings in logs
    source: Optional[Literal["saved", "my_post", "thread", "subtitle", "library"]] = None
    mode: Literal["hybrid", "vector"] = "hybrid"
    retrieve_only: bool = False
    top_n: int = Field(default=5, ge=1, le=10)


def pick_engine(source):
    # "library" = the book library engine (local mode only); every other source belongs to the main engine
    return get_library() if source == "library" else rag


# ===== Endpoints =====
@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health():
    return {"status": "ok", "mode": MODE, "docs": len(rag.docs)}


@app.post("/ask")
def ask(req: AskRequest, request: Request):
    t0 = time.time()
    record = {
        "ts": datetime.now(KST).isoformat(timespec="seconds"),
        "mode": MODE, "question": req.question, "source": req.source,
        "retrieve_only": req.retrieve_only,
    }

    # Limits first, before any paid / quota-limited API call
    if MODE == "cloud":
        # Number of X-Forwarded-For entries only (no IPs) -> lets us verify the header shape in Cloud Logging
        record["xff_hops"] = len([p for p in request.headers.get("x-forwarded-for", "").split(",") if p.strip()])
        try:
            check_limits(client_ip(request))
        except HTTPException as e:
            record.update({"status": "rejected", "error": e.detail})
            write_log(record)
            raise

    engine = pick_engine(req.source)
    try:
        query, hits = engine.retrieve(req.question, req.source, req.mode, req.top_n)
        t1 = time.time()
        answer = None
        if not req.retrieve_only:
            answer = engine.generate(req.question, hits)
        t2 = time.time()
    except Exception as e:
        record.update({"status": "error", "error": f"{type(e).__name__}: {e}",
                       "total_sec": round(time.time() - t0, 2)})
        write_log(record)
        # Details stay in the log; the public cloud demo only sees a generic message
        detail = "Internal error. Please try again later." if MODE == "cloud" else f"{type(e).__name__}: {e}"
        raise HTTPException(status_code=500, detail=detail)

    record.update({
        "status": "ok", "search_query": query,
        "doc_ids": [h["doc_id"] for h in hits],
        "answer_chars": len(answer) if answer else 0,
        "retrieve_sec": round(t1 - t0, 2), "generate_sec": round(t2 - t1, 2),
        "total_sec": round(t2 - t0, 2),
    })
    write_log(record)

    return {
        "question": req.question, "search_query": query, "answer": answer,
        "sources": _public_sources(hits),
        "timing_sec": {"retrieve": record["retrieve_sec"], "generate": record["generate_sec"]},
    }


# ===== Streaming answers (/ask/stream) =====
# Format: NDJSON (one JSON object per line, media type application/x-ndjson).
# Why not SSE: the browser's EventSource cannot send a POST body, so the page reads the stream with fetch() anyway,
# and "split on newline + JSON.parse" is simpler than parsing SSE's "event:/data:" fields, with no extra library.
# Event order:  {"type":"sources",...}  ->  {"type":"chunk","text":...} x N  ->  {"type":"done",...}
# If the engine fails after streaming has started: {"type":"error","detail":...} is the last line instead of "done".
GENERIC_ERROR = "Internal error. Please try again later."


def _event(obj):
    # One NDJSON line; ensure_ascii=False keeps Korean readable and short on the wire
    return json.dumps(obj, ensure_ascii=False) + "\n"


# Extra fields only the book library adds (title, book id, page numbers, OCR confidence)
_BOOK_FIELDS = ("title", "book_id", "page", "pages", "ocr", "ocr_low")


def _public_sources(hits):
    # Source fields /ask and /ask/stream return (no full text, only a 120-char preview)
    out = []
    for h in hits:
        item = {"n": h["n"], "source": h["source"], "date": h["date"], "url": h["url"],
                "via": h["via"], "preview": h["text"][:120]}
        item.update({k: h[k] for k in _BOOK_FIELDS if k in h})
        out.append(item)
    return out


def _answer_chunks(question, hits, engine=None):
    # Engines with generate_stream() give text piece by piece;
    # an engine without it still works: its whole generate() answer is sent as a single chunk
    engine = engine or rag
    stream = getattr(engine, "generate_stream", None)
    if stream is None:
        yield engine.generate(question, hits)
        return
    yield from stream(question, hits)


@app.post("/ask/stream")
def ask_stream(req: AskRequest, request: Request):
    # Validation (AskRequest -> 422) has already happened before this line, exactly like /ask
    t0 = time.time()
    record = {
        "ts": datetime.now(KST).isoformat(timespec="seconds"),
        "mode": MODE, "question": req.question, "source": req.source,
        "retrieve_only": req.retrieve_only, "stream": True,
    }

    # Limits first (counted once per request), before any paid / quota-limited API call -> normal 429 JSON response
    if MODE == "cloud":
        record["xff_hops"] = len([p for p in request.headers.get("x-forwarded-for", "").split(",") if p.strip()])
        try:
            check_limits(client_ip(request))
        except HTTPException as e:
            record.update({"status": "rejected", "error": e.detail})
            write_log(record)
            raise

    # Retrieval runs BEFORE the stream starts, so a search failure is still a plain HTTP 500 (same as /ask)
    engine = pick_engine(req.source)
    try:
        query, hits = engine.retrieve(req.question, req.source, req.mode, req.top_n)
    except Exception as e:
        record.update({"status": "error", "error": f"{type(e).__name__}: {e}",
                       "total_sec": round(time.time() - t0, 2)})
        write_log(record)
        detail = GENERIC_ERROR if MODE == "cloud" else f"{type(e).__name__}: {e}"
        raise HTTPException(status_code=500, detail=detail)
    t1 = time.time()

    def events():
        # Runs in a worker thread (StreamingResponse iterates sync generators in the threadpool),
        # so the blocking LLM calls do not freeze the server's event loop
        answer_chars = 0
        logged = False
        try:
            # 1) Sources first: the page can show them while the LLM is still writing
            yield _event({"type": "sources", "question": req.question, "search_query": query,
                          "sources": _public_sources(hits)})

            # 2) Answer text, piece by piece (skipped in search-only mode)
            if not req.retrieve_only:
                for piece in _answer_chunks(req.question, hits, engine):
                    if piece:
                        answer_chars += len(piece)
                        yield _event({"type": "chunk", "text": piece})
            t2 = time.time()

            # 3) One log record at the very end (same fields as /ask), written BEFORE "done" is sent
            record.update({
                "status": "ok", "search_query": query,
                "doc_ids": [h["doc_id"] for h in hits],
                "answer_chars": answer_chars,
                "retrieve_sec": round(t1 - t0, 2), "generate_sec": round(t2 - t1, 2),
                "total_sec": round(t2 - t0, 2),
            })
            write_log(record)
            logged = True
            yield _event({"type": "done", "answer_chars": answer_chars,
                          "timing_sec": {"retrieve": record["retrieve_sec"], "generate": record["generate_sec"]}})
        except GeneratorExit:
            # The browser closed the connection mid-answer: log it once, then let the generator close
            if not logged:
                record.update({"status": "error", "error": "client disconnected", "search_query": query,
                               "answer_chars": answer_chars, "total_sec": round(time.time() - t0, 2)})
                write_log(record)
            raise
        except Exception as e:
            # HTTP 200 + headers are already sent, so the error travels as the last event instead of a 500
            record.update({"status": "error", "error": f"{type(e).__name__}: {e}", "search_query": query,
                           "answer_chars": answer_chars, "total_sec": round(time.time() - t0, 2)})
            write_log(record)
            # Details stay in the log; the public cloud demo only sees a generic message
            detail = GENERIC_ERROR if MODE == "cloud" else f"{type(e).__name__}: {e}"
            yield _event({"type": "error", "detail": detail})

    # no-cache / X-Accel-Buffering: ask proxies not to hold the chunks back until the end
    headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    return StreamingResponse(events(), media_type="application/x-ndjson", headers=headers)


# ===== Original book PDF (local mode only) =====
# The page shows figures and graphs that OCR cannot read. Only a manifest book id is accepted (never a client path),
# and the engine checks that the file is a PDF inside the library folder.
@app.get("/library/pdf/{book_id}")
def library_pdf(book_id: str):
    path = get_library().pdf_path(book_id)
    if path is None:
        raise HTTPException(status_code=404, detail="Not found")
    # inline: the browser's own PDF viewer opens it; the page adds #page=N to jump to the right page
    return FileResponse(path, media_type="application/pdf", headers={"Content-Disposition": "inline"})


# → 클라우드 모드에서는 API를 부르기 "전에" 사람별(1분 5회)·하루 전체(200회) 제한을 먼저 검사하고, 넘으면 429로 거절하겠다.
#   질문 길이(500자)와 출처 개수(1~10)는 요청 형식 단계에서 자동으로 걸러진다.
#   거절된 요청도 "rejected"로 로그에 남겨서, 나중에 얼마나 막혔는지 운영 지표로 볼 수 있게 한다.
# → /ask/stream은 같은 검사·제한·로그를 그대로 거친 뒤, 출처(sources) → 답변 조각(chunk) → 완료(done) 순서로
#   한 줄에 JSON 하나씩(NDJSON) 흘려보내서, 답변 전체를 기다리지 않고 글자가 바로바로 보이게 하겠다.
#   스트리밍 도중 에러가 나면 이미 200을 보낸 뒤라서, 마지막 줄을 error 이벤트(클라우드는 일반 메시지만)로 보낸다.