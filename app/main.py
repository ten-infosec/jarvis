# ===== Jarvis RAG API (local only) =====
# Run:  uvicorn app.main:app
# Chat: http://127.0.0.1:8000/
# Docs: http://127.0.0.1:8000/docs


import json
import os
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

 


app = FastAPI(title="Jarvis RAG (local)")
# JARVIS_MODE=cloud -> public arXiv + Gemini, anything else -> local private data + e5 + Ollama
MODE = os.getenv("JARVIS_MODE", "local")
if MODE == "cloud":
    from app.rag_cloud import CloudRAG as Engine
else:
    from app.rag import RAG as Engine
rag = Engine()
STATIC_DIR = Path(__file__).parent / "static"

# Korea Standard Time (UTC+9, no daylight saving). Containers default to UTC, so set it explicitly
KST = timezone(timedelta(hours=9))

# ===== Request log =====
# Stored in the private data folder (never in the git repo) because it contains my questions
LOG_FILE = Path(os.getenv("JARVIS_DATA_DIR", r"C:\instagram_export")) / "logs" / "requests.jsonl"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
# Only one request may write to the file at a time, so lines never get mixed
_log_lock = threading.Lock()


def write_log(record):
    with _log_lock:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            # ensure_ascii=False keeps Korean readable in the file instead of \uXXXX codes
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


class AskRequest(BaseModel):
    question: str
    source: Optional[str] = None
    mode: Literal["hybrid", "vector"] = "hybrid"
    retrieve_only: bool = False
    top_n: int = 5


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health():
    return {"status": "ok", "docs": len(rag.docs)}


@app.post("/ask")
def ask(req: AskRequest):
    t0 = time.time()
    # Fields known before any work starts
    record = {
                "ts": datetime.now(KST).isoformat(timespec="seconds"),
        "question": req.question, "source": req.source,
        "mode": req.mode, "retrieve_only": req.retrieve_only,
    }

    try:
        query, hits = rag.retrieve(req.question, req.source, req.mode, req.top_n)
        t1 = time.time()
        answer = None
        if not req.retrieve_only:
            answer = rag.generate(req.question, hits)
        t2 = time.time()
    except Exception as e:
        # Failures are logged too (e.g. Ollama turned off), then reported to the browser as HTTP 500
        record.update({
            "status": "error",
            "error": f"{type(e).__name__}: {e}",
            "total_sec": round(time.time() - t0, 2),
        })
        write_log(record)
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")

    record.update({
        "status": "ok",
        "search_query": query,
        "doc_ids": [h["doc_id"] for h in hits],
        "answer_chars": len(answer) if answer else 0,
        "retrieve_sec": round(t1 - t0, 2),
        "generate_sec": round(t2 - t1, 2),
        "total_sec": round(t2 - t0, 2),
    })
    write_log(record)

    return {
        "question": req.question,
        "search_query": query,
        "answer": answer,
        "sources": [
            {"n": h["n"], "source": h["source"], "date": h["date"], "url": h["url"],
             "via": h["via"], "preview": h["text"][:120]}
            for h in hits
        ],
        "timing_sec": {"retrieve": record["retrieve_sec"], "generate": record["generate_sec"]},
    }


# → 질문이 들어올 때마다 시각, 질문, 검색어, 찾은 문서, 답 길이, 단계별 시간을 한 줄씩 requests.jsonl에 기록하겠다.
#   에러가 나도 원인을 기록한 뒤 브라우저에 500 에러로 알려 준다.
#   로그 파일은 내 질문이 담겨 있으니 저장소 밖의 개인 데이터 폴더에 둔다.