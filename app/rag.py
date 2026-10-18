# ===== RAG core: load models and data once, answer many questions =====


import json
import os
import re
from collections import defaultdict
from pathlib import Path

import chromadb
import requests
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


# ===== Settings (same values as ask.py, chosen in E3/E5) =====
# JARVIS_DATA_DIR: set to /data inside Docker; falls back to the laptop folder when not set
DATA_DIR = Path(os.getenv("JARVIS_DATA_DIR", r"C:\instagram_export"))
CORPUS = DATA_DIR / "output" / "corpus.jsonl"
INDEX_DIR = str(DATA_DIR / "index" / "chroma")
MODEL_NAME = "intfloat/multilingual-e5-small"
COLLECTION = "insta_v0"
# Inside Docker, "localhost" is the container itself, so the host's Ollama is reached via host.docker.internal
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat")
LLM = "qwen2.5:3b"
CANDIDATES = 50
RRF_K = 60
W_VEC = 1.5
W_BM25 = 1.0
CONTEXT_CHARS = 600

STOPWORDS = {
    "내가", "나", "제가", "내", "저장한", "저장했던", "저장된", "저장",
    "관련", "관련된", "관한", "대한", "게시물", "게시글", "포스트", "글",
    "알려줘", "찾아줘", "보여줘", "정리해줘", "추천해줘",
    "뭐", "뭐야", "어떤", "있어", "있는", "좀", "것", "거",
}

SYSTEM_PROMPT = (
    "너는 사용자의 개인 기록을 바탕으로 답하는 비서야. "
    "반드시 아래 [자료]에 있는 내용만 사용해서 한국어로 답해. "
    "문장마다 근거가 된 자료 번호를 [1]처럼 붙여. "
    "질문과 관계없는 자료는 언급하지 마. "
    "[자료]에 답이 없으면 '기록에서 찾지 못했어요'라고 말해."
)


# ===== Text helpers =====
def tokenize(text):
    text = text.lower()
    words = re.findall(r"[가-힣]+|[a-z0-9]+", text)
    tokens = []
    for w in words:
        tokens.append(w)
        if re.match(r"[가-힣]", w) and len(w) > 2:
            tokens += [w[i:i + 2] for i in range(len(w) - 1)]
    return tokens


def clean_query(question):
    words = re.findall(r"[가-힣]+|[a-zA-Z0-9]+", question)
    kept = [w for w in words if w.lower() not in STOPWORDS]
    return " ".join(kept) if kept else question


# ===== RAG class =====
class RAG:
    def __init__(self):
        # Everything heavy is loaded here, once, when the server starts
        self.docs = {}
        with open(CORPUS, encoding="utf-8") as f:
            for line in f:
                doc = json.loads(line)
                self.docs[doc["id"]] = doc
        self.model = SentenceTransformer(MODEL_NAME)
        self.col = chromadb.PersistentClient(path=INDEX_DIR).get_collection(COLLECTION)
        # BM25 indexes are built on first use per source and then reused (cache)
        self._bm25 = {}

    def _get_bm25(self, source):
        key = source or "all"
        if key not in self._bm25:
            ids = [d for d, doc in self.docs.items() if not source or doc["source"] == source]
            index = BM25Okapi([tokenize(self.docs[d]["text"]) for d in ids])
            self._bm25[key] = (index, ids)
        return self._bm25[key]

    def retrieve(self, question, source=None, mode="hybrid", top_n=5):
        query = clean_query(question)

        # --- Vector search, one entry per post ---
        q_vec = self.model.encode(["query: " + query], normalize_embeddings=True).tolist()
        where = {"source": source} if source else None
        res = self.col.query(query_embeddings=q_vec, n_results=CANDIDATES * 3, where=where)
        vec_rank, best_chunk = [], {}
        for text, meta in zip(res["documents"][0], res["metadatas"][0]):
            d = meta["doc_id"]
            if d in self.docs and d not in best_chunk:
                best_chunk[d] = text
                vec_rank.append(d)
        vec_rank = vec_rank[:CANDIDATES]

        # --- BM25 + weighted RRF ---
        bm_rank = []
        if mode == "hybrid":
            bm25, ids = self._get_bm25(source)
            scores = bm25.get_scores(tokenize(query))
            order = sorted(range(len(scores)), key=lambda i: -scores[i])
            bm_rank = [ids[i] for i in order[:CANDIDATES] if scores[i] > 0]
            fused = defaultdict(float)
            for ranked, w in [(vec_rank, W_VEC), (bm_rank, W_BM25)]:
                for rank, d in enumerate(ranked, start=1):
                    fused[d] += w / (RRF_K + rank)
            final = sorted(fused, key=lambda d: -fused[d])[:top_n]
        else:
            final = vec_rank[:top_n]

        # --- Package results as plain dictionaries (easy to send as JSON) ---
        hits = []
        for i, d in enumerate(final, start=1):
            doc = self.docs[d]
            if d in best_chunk and d in bm_rank:
                via = "vec+bm25"
            elif d in best_chunk:
                via = "vec"
            else:
                via = "bm25"
            hits.append({
                "n": i, "doc_id": d, "source": doc["source"], "date": doc.get("date", ""),
                "url": doc.get("meta", {}).get("url", ""), "via": via,
                "text": best_chunk.get(d) or doc["text"][:CONTEXT_CHARS],
            })
        return query, hits

    def generate(self, question, hits):
        context = "\n\n".join(f"[{h['n']}] ({h['source']}, {h['date']})\n{h['text']}" for h in hits)
        r = requests.post(OLLAMA_URL, json={
            "model": LLM,
            "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                         {"role": "user", "content": f"[자료]\n{context}\n\n[질문]\n{question}"}],
            "stream": False,
        }, timeout=600)
        r.raise_for_status()
        return r.json()["message"]["content"]

    def generate_stream(self, question, hits):
        # Same prompt as generate(), but with "stream": True Ollama sends one JSON object per line (NDJSON)
        context = "\n\n".join(f"[{h['n']}] ({h['source']}, {h['date']})\n{h['text']}" for h in hits)
        # stream=True (requests): read the body as it arrives instead of waiting for the whole answer
        with requests.post(OLLAMA_URL, json={
            "model": LLM,
            "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                         {"role": "user", "content": f"[자료]\n{context}\n\n[질문]\n{question}"}],
            "stream": True,
        }, timeout=600, stream=True) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                # Blank keep-alive lines carry nothing
                if not line:
                    continue
                data = json.loads(line)
                # Ollama reports problems that happen mid-answer as {"error": "..."} on its own line
                if data.get("error"):
                    raise RuntimeError(f"Ollama error: {data['error']}")
                piece = data.get("message", {}).get("content", "")
                if piece:
                    yield piece
                # The last line has "done": true (plus timing stats) and no more text
                if data.get("done"):
                    break


# → 문서, 임베딩 모델, 벡터 DB를 서버가 켜질 때 한 번만 불러 두고,
#   질문이 올 때마다 retrieve(검색)와 generate(LLM 답변)만 실행하겠다.
#   BM25는 출처별로 처음 쓸 때 한 번 만들어 저장해 두고 재사용한다.
#   LLM 지시문에 "관계없는 자료는 언급하지 마"를 추가해서 아까 [4] 같은 엉뚱한 요약을 줄여 본다.
#   generate_stream은 Ollama에 stream=true로 요청해, 한 줄씩 오는 JSON에서 글자 조각만 꺼내 바로 넘겨준다.