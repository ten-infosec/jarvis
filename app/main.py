# ===== Jarvis RAG API (local only) =====
# Run:  uvicorn app.main:app
# Chat: http://127.0.0.1:8000/
# Docs: http://127.0.0.1:8000/docs


import time
from pathlib import Path
from typing import Literal, Optional

from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.rag import RAG


app = FastAPI(title="Jarvis RAG (local)")
rag = RAG()
# Folder that holds index.html (next to this file)
STATIC_DIR = Path(__file__).parent / "static"


class AskRequest(BaseModel):
    question: str
    source: Optional[str] = None
    mode: Literal["hybrid", "vector"] = "hybrid"
    retrieve_only: bool = False
    top_n: int = 5


# ===== Chat page =====
# GET /: send the chat screen file as it is
@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health():
    return {"status": "ok", "docs": len(rag.docs)}


@app.post("/ask")
def ask(req: AskRequest):
    t0 = time.time()
    query, hits = rag.retrieve(req.question, req.source, req.mode, req.top_n)
    t1 = time.time()
    answer = None
    if not req.retrieve_only:
        answer = rag.generate(req.question, hits)
    t2 = time.time()
    return {
        "question": req.question,
        "search_query": query,
        "answer": answer,
        "sources": [
            {"n": h["n"], "source": h["source"], "date": h["date"], "url": h["url"],
             "via": h["via"], "preview": h["text"][:120]}
            for h in hits
        ],
        "timing_sec": {"retrieve": round(t1 - t0, 2), "generate": round(t2 - t1, 2)},
    }


# → 기존 /ask, /health는 그대로 두고, 첫 주소(/)로 들어오면 static 폴더의 index.html(채팅 화면)을 보내 주겠다.