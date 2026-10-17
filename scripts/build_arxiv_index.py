# ===== Build the arXiv vector index with Gemini embeddings (PUBLIC data only) =====
# Usage: python scripts/build_arxiv_index.py


import json
import os
import time
from pathlib import Path

import chromadb
import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types


# ===== Settings =====
load_dotenv()
ROOT = Path(__file__).resolve().parent.parent
PAPERS = ROOT / "data" / "arxiv" / "papers.jsonl"
# Separate folder from the private Instagram index, so the two can never mix
INDEX_DIR = ROOT / "data" / "arxiv" / "chroma"
COLLECTION = "arxiv_gemini"
EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-001")
# 3072 -> 768 dimensions: 4x smaller storage, small quality loss for this model
DIM = 768
BATCH = 50
SLEEP_SEC = 1.0
MAX_CHARS = 2000

key = os.getenv("GEMINI_API_KEY")
if not key:
    raise SystemExit("GEMINI_API_KEY not found in .env")
client = genai.Client(api_key=key)


def embed(texts):
    for attempt in range(6):
        try:
            res = client.models.embed_content(
                model=EMBED_MODEL,
                contents=texts,
                # RETRIEVAL_DOCUMENT: "these are documents to be searched" (questions use RETRIEVAL_QUERY)
                config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT", output_dimensionality=DIM),
            )
            vectors = np.array([e.values for e in res.embeddings], dtype=np.float32)
            # Reduced-size vectors are not unit length, so normalize them
            return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        except Exception as e:
            msg = str(e)
            # Daily quota: waiting minutes will not help -> stop cleanly and resume tomorrow
            if "PerDay" in msg:
                raise SystemExit("Daily free quota used up. Run again tomorrow (indexed papers are skipped).")
            # Per-minute quota resets every minute -> wait a full minute
            if "429" in msg:
                print(f"  rate limit (per minute) -> waiting 60s  [{attempt + 1}/6]")
                time.sleep(60)
            else:
                print(f"  error -> waiting 10s  ({type(e).__name__}: {msg[:120]})")
                time.sleep(10)
    raise RuntimeError("Embedding failed 6 times in a row")


# ===== 1) Load papers and skip the ones already indexed =====
docs = []
with open(PAPERS, encoding="utf-8") as f:
    for line in f:
        if line.strip():
            docs.append(json.loads(line))

INDEX_DIR.mkdir(parents=True, exist_ok=True)
col = chromadb.PersistentClient(path=str(INDEX_DIR)).get_or_create_collection(
    COLLECTION, metadata={"hnsw:space": "cosine"})
existing = set(col.get(include=[])["ids"])
todo = [d for d in docs if d["id"] not in existing]
print(f"Papers: {len(docs):,}  already indexed: {len(existing):,}  to embed: {len(todo):,}")

# ===== 2) Embed and store batch by batch =====
start = time.time()
for b in range(0, len(todo), BATCH):
    batch = todo[b:b + BATCH]
    vectors = embed([d["text"][:MAX_CHARS] for d in batch])
    col.add(
        ids=[d["id"] for d in batch],
        documents=[d["text"] for d in batch],
        metadatas=[{
            "source": "arxiv",
            "date": d["date"],
            "url": d["meta"]["url"],
            "title": d["meta"]["title"][:300],
            "categories": " | ".join(d["meta"]["categories"]),
        } for d in batch],
        embeddings=vectors.tolist(),
    )
    print(f"  {b + len(batch):,}/{len(todo):,}  ({time.time() - start:.0f}s)")
    time.sleep(SLEEP_SEC)

print(f"Done. Collection size: {col.count():,}  -> {INDEX_DIR}")


# → 공개 논문을 50편씩 Gemini 임베딩으로 768차원 벡터로 바꿔 arxiv_gemini 컬렉션에 저장하겠다.
#   1분 한도에 걸리면 60초 기다렸다 다시 보내고, 하루 한도면 안내 후 멈춘다. 이미 저장된 논문은 건너뛴다.