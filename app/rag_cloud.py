# ===== Cloud RAG engine: public arXiv papers + Gemini (embedding + answer) =====
# Same interface as app/rag.py (retrieve, generate), but no torch / no local model / no private data


import json
import os
import re
from collections import defaultdict
from pathlib import Path

import chromadb
import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types
from rank_bm25 import BM25Okapi


# ===== Settings =====
load_dotenv()
ROOT = Path(__file__).resolve().parent.parent
# Public paper data folder (can be changed by environment variable inside Docker)
ARXIV_DIR = Path(os.getenv("ARXIV_DIR", str(ROOT / "data" / "arxiv")))
PAPERS = ARXIV_DIR / "papers.jsonl"
INDEX_DIR = str(ARXIV_DIR / "chroma")
COLLECTION = "arxiv_gemini"
EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-001")
LLM_MODEL = os.getenv("GEMINI_LLM_MODEL", "gemini-flash-latest")
# Must match the index (768 dimensions)
DIM = 768
CANDIDATES = 50
# Small k makes the top ranks of each list dominate, so a keyword-only hit at rank 1 is not pushed out by vector hits
# (50-question eval, English Hit@10: 0.71 at k=60 -> 0.96 at k=3)
RRF_K = 3
W_VEC = 1.5
W_BM25 = 1.0
CONTEXT_CHARS = 1500

# Request words in Korean and English that say HOW to answer, not WHAT to find
STOPWORDS = {
    "관련", "관련된", "논문", "최신", "최근", "알려줘", "찾아줘", "보여줘", "정리해줘",
    "뭐", "뭐야", "어떤", "있어", "있는", "좀", "대한", "관한",
    "what", "which", "paper", "papers", "about", "on", "the", "a", "an", "of", "for",
    "in", "is", "are", "show", "me", "find", "tell", "related", "recent", "latest",
}

SYSTEM_PROMPT = (
    "You are a research assistant for medical AI papers. "
    "Answer ONLY from the [Sources] below and cite them like [1] after each claim. "
    "Do not mention sources that are unrelated to the question. "
    "If the sources do not contain the answer, say so. "
    "Answer in the same language as the question."
)


# ===== Text helpers (same rules as the local engine) =====
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


# ===== Engine =====
class CloudRAG:
    def __init__(self):
        key = os.getenv("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY is not set")
        self.client = genai.Client(api_key=key)

        # All collected papers (BM25 can search all of them, even ones not embedded yet)
        self.docs = {}
        with open(PAPERS, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    self.docs[d["id"]] = d
        self.ids = list(self.docs)
        self.bm25 = BM25Okapi([tokenize(self.docs[i]["text"]) for i in self.ids])
        self.col = chromadb.PersistentClient(path=INDEX_DIR).get_collection(COLLECTION)

    def _embed_query(self, query):
        res = self.client.models.embed_content(
            model=EMBED_MODEL,
            contents=[query],
            config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY", output_dimensionality=DIM),
        )
        v = np.array(res.embeddings[0].values, dtype=np.float32)
        return (v / np.linalg.norm(v)).tolist()

    def retrieve(self, question, source=None, mode="hybrid", top_n=5):
        query = clean_query(question)

        # --- Vector search, with graceful degradation ---
        vec_rank = []
        try:
            out = self.col.query(query_embeddings=[self._embed_query(query)], n_results=CANDIDATES)
            vec_rank = [i for i in out["ids"][0] if i in self.docs]
        except Exception as e:
            # Embedding API down or over quota -> keep serving with keyword search only
            print(f"[warn] vector search unavailable -> BM25 only ({type(e).__name__}: {str(e)[:400]})")

        # --- BM25 (always used in hybrid mode, and as the fallback when vectors failed) ---
        bm_rank = []
        if mode == "hybrid" or not vec_rank:
            scores = self.bm25.get_scores(tokenize(query))
            order = sorted(range(len(scores)), key=lambda i: -scores[i])
            bm_rank = [self.ids[i] for i in order[:CANDIDATES] if scores[i] > 0]

        # --- Weighted RRF (same weights chosen in E3) ---
        fused = defaultdict(float)
        for ranked, w in [(vec_rank, W_VEC), (bm_rank, W_BM25)]:
            for rank, d in enumerate(ranked, start=1):
                fused[d] += w / (RRF_K + rank)
        final = sorted(fused, key=lambda d: -fused[d])[:top_n]

        hits = []
        for i, d in enumerate(final, start=1):
            doc = self.docs[d]
            if d in vec_rank and d in bm_rank:
                via = "vec+bm25"
            elif d in vec_rank:
                via = "vec"
            else:
                via = "bm25"
            hits.append({
                "n": i, "doc_id": d, "source": "arxiv", "date": doc["date"],
                "url": doc["meta"]["url"], "via": via, "text": doc["text"][:CONTEXT_CHARS],
            })
        return query, hits

    def generate(self, question, hits):
        if not hits:
            return "관련 논문을 찾지 못했어요. (No matching papers found.)"
        context = "\n\n".join(f"[{h['n']}] ({h['date']}) {h['text']}" for h in hits)
        out = self.client.models.generate_content(
            model=LLM_MODEL,
            contents=f"[Sources]\n{context}\n\n[Question]\n{question}",
            # system_instruction: rules the model must follow; low temperature = less random, more consistent answers
            config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT, temperature=0.2),
        )
        return out.text

    def generate_stream(self, question, hits):
        # Same prompt and config as generate(), but yields the answer piece by piece as Gemini writes it
        if not hits:
            yield "관련 논문을 찾지 못했어요. (No matching papers found.)"
            return
        context = "\n\n".join(f"[{h['n']}] ({h['date']}) {h['text']}" for h in hits)
        stream = self.client.models.generate_content_stream(
            model=LLM_MODEL,
            contents=f"[Sources]\n{context}\n\n[Question]\n{question}",
            config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT, temperature=0.2),
        )
        for chunk in stream:
            # Some chunks carry only metadata (e.g. the final usage info) and have no text -> skip them
            if chunk.text:
                yield chunk.text


# → 공개 논문만 다루는 클라우드 엔진을 만들겠다. 질문을 Gemini로 임베딩해 벡터 검색하고, BM25 결과와 1.5 : 1로 합친다.
#   임베딩 API가 한도 초과나 장애로 실패하면 멈추지 않고 BM25만으로 검색을 계속한다(우아한 성능 저하).
#   답변은 Gemini가 출처 번호를 붙여, 질문과 같은 언어로 쓴다.
#   generate_stream은 같은 설정으로 generate_content_stream을 불러, 답변을 조각조각 바로 넘겨준다.