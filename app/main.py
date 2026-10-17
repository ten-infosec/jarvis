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
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field


# ===== Engine selection =====
MODE = os.getenv("JARVIS_MODE", "local")
if MODE == "cloud":
    from app.rag_cloud import CloudRAG as Engine
else:
    from app.rag import RAG as Engine

app = FastAPI(title=f"Jarvis RAG ({MODE})")
rag = Engine()
STATIC_DIR = Path(__file__).parent / "static"
KST = timezone(timedelta(hours=9))

# ===== Request log =====
LOG_FILE = Path(os.getenv("JARVIS_DATA_DIR", r"C:\instagram_export")) / "logs" / "requests.jsonl"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
_log_lock = threading.Lock()


def write_log(record):
    with _log_lock:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ===== Rate limits (cloud mode only) =====
# Values can be changed with environment variables at deploy time
RATE_PER_MIN = int(os.getenv("RATE_PER_MIN", "5"))
DAILY_CAP = int(os.getenv("DAILY_CAP", "200"))
# ip -> times of that IP's recent requests (deque = list that is fast to trim from the front)
_recent = defaultdict(deque)
_daily = {"date": None, "count": 0}
_limit_lock = threading.Lock()


def client_ip(request: Request):
    # Behind a cloud load balancer, the original address arrives in X-Forwarded-For
    # (it can be faked by the client, which is why the daily cap below is the real safety net)
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def check_limits(ip):
    now = time.time()
    with _limit_lock:
        # 1) Per-IP: forget requests older than 60 seconds, then count what is left
        times = _recent[ip]
        while times and now - times[0] > 60:
            times.popleft()
        if len(times) >= RATE_PER_MIN:
            raise HTTPException(status_code=429, detail=f"Too many requests: max {RATE_PER_MIN} per minute. Please wait.")

        # 2) Whole service: reset the counter when the KST date changes
        today = datetime.now(KST).date().isoformat()
        if _daily["date"] != today:
            _daily["date"], _daily["count"] = today, 0
        if _daily["count"] >= DAILY_CAP:
            raise HTTPException(status_code=429, detail="Daily limit reached for this demo. Please try again tomorrow.")

        times.append(now)
        _daily["count"] += 1


# ===== Request format (3, 4: size limits checked automatically by pydantic -> HTTP 422) =====
class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500)
    source: Optional[str] = None
    mode: Literal["hybrid", "vector"] = "hybrid"
    retrieve_only: bool = False
    top_n: int = Field(default=5, ge=1, le=10)


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
        try:
            check_limits(client_ip(request))
        except HTTPException as e:
            record.update({"status": "rejected", "error": e.detail})
            write_log(record)
            raise

    try:
        query, hits = rag.retrieve(req.question, req.source, req.mode, req.top_n)
        t1 = time.time()
        answer = None
        if not req.retrieve_only:
            answer = rag.generate(req.question, hits)
        t2 = time.time()
    except Exception as e:
        record.update({"status": "error", "error": f"{type(e).__name__}: {e}",
                       "total_sec": round(time.time() - t0, 2)})
        write_log(record)
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")

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
        "sources": [
            {"n": h["n"], "source": h["source"], "date": h["date"], "url": h["url"],
             "via": h["via"], "preview": h["text"][:120]}
            for h in hits
        ],
        "timing_sec": {"retrieve": record["retrieve_sec"], "generate": record["generate_sec"]},
    }


# → 클라우드 모드에서는 API를 부르기 "전에" 사람별(1분 5회)·하루 전체(200회) 제한을 먼저 검사하고, 넘으면 429로 거절하겠다.
#   질문 길이(500자)와 출처 개수(1~10)는 요청 형식 단계에서 자동으로 걸러진다.
#   거절된 요청도 "rejected"로 로그에 남겨서, 나중에 얼마나 막혔는지 운영 지표로 볼 수 있게 한다.