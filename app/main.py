# ===== Jarvis RAG API (local only) =====
# Run:  uvicorn app.main:app
# Docs: http://127.0.0.1:8000/docs


import time
from typing import Literal, Optional

from fastapi import FastAPI
from pydantic import BaseModel

from app.rag import RAG


app = FastAPI(title="Jarvis RAG (local)")
# Created once when the server starts -> models stay in memory between requests
rag = RAG()


# ===== Request format =====
# BaseModel: describes what JSON the client must send; FastAPI checks it automatically
class AskRequest(BaseModel):
    question: str
    # Optional[str] = None: may be left out (search all sources)
    source: Optional[str] = None
    # Literal: only these two values are accepted
    mode: Literal["hybrid", "vector"] = "hybrid"
    retrieve_only: bool = False
    top_n: int = 5


# ===== Endpoints =====
# GET /health: quick "is the server alive?" check (used later for monitoring)
@app.get("/health")
def health():
    return {"status": "ok", "docs": len(rag.docs)}


# POST /ask: receive a question, return answer + sources + timing
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
        # Timing per step: the first operations metric (stage 4)
        "timing_sec": {"retrieve": round(t1 - t0, 2), "generate": round(t2 - t1, 2)},
    }


# → 서버가 켜질 때 RAG를 한 번 만들어 두고, /health(상태 확인)와 /ask(질문) 두 창구를 열겠다.
#   /ask는 질문을 받아 검색어 정리 → 하이브리드 검색 → LLM 답변을 실행하고,
#   답과 출처, 단계별 걸린 시간을 JSON으로 돌려준다.