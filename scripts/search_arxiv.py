# ===== Quick search test on the arXiv index (Gemini query embedding) =====
# Usage: python scripts/search_arxiv.py "chest X-ray pneumonia detection"


import os
import sys
from pathlib import Path

import chromadb
import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types


load_dotenv()
ROOT = Path(__file__).resolve().parent.parent
INDEX_DIR = ROOT / "data" / "arxiv" / "chroma"
EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-001")
# Must match the dimension used when the index was built
DIM = 768

# sys.argv: words typed after the script name; join them into one question
query = " ".join(sys.argv[1:]) or "chest X-ray pneumonia detection"
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# RETRIEVAL_QUERY: tells the model this text is a question, not a stored document
res = client.models.embed_content(
    model=EMBED_MODEL,
    contents=[query],
    config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY", output_dimensionality=DIM),
)
vec = np.array(res.embeddings[0].values, dtype=np.float32)
# Normalize the same way as the stored vectors
vec = vec / np.linalg.norm(vec)

col = chromadb.PersistentClient(path=str(INDEX_DIR)).get_collection("arxiv_gemini")
out = col.query(query_embeddings=[vec.tolist()], n_results=5)

print(f"Query: {query}\n")
for i, (meta, dist) in enumerate(zip(out["metadatas"][0], out["distances"][0]), start=1):
    print(f"[{i}] dist={dist:.3f}  {meta['date']}  {meta['title'][:90]}")
    print(f"     {meta['url']}")


# → 질문을 Gemini로 "질문용" 768차원 벡터로 바꾸고 크기를 1로 맞춘 뒤,
#   arXiv 인덱스에서 가장 가까운 논문 5편의 제목, 날짜, 거리, 링크를 출력하겠다.