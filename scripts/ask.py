# ===== Ask the second brain: search ChromaDB, then answer with Ollama =====
# Usage: python scripts/ask.py "질문"
#        python scripts/ask.py "질문" --source saved   (search only one source)


import argparse
# requests: send HTTP requests to the local Ollama server
import requests
import chromadb
from sentence_transformers import SentenceTransformer


# ===== Settings =====
INDEX_DIR = r"C:\instagram_export\index\chroma"
COLLECTION = "insta_v0"
MODEL_NAME = "intfloat/multilingual-e5-small"
# Ollama runs as a local server on this address
OLLAMA_URL = "http://localhost:11434/api/chat"
LLM = "qwen2.5:3b"
# TOP_K: how many chunks to hand to the LLM
TOP_K = 5


# ===== 1) Read the question =====
parser = argparse.ArgumentParser()
parser.add_argument("question")
# --source: saved / my_post / thread / subtitle (optional filter)
parser.add_argument("--source", default=None)
args = parser.parse_args()

# ===== 2) Search: question -> vector -> nearest chunks =====
model = SentenceTransformer(MODEL_NAME)
# e5 rule: questions start with "query: "
q_vec = model.encode(["query: " + args.question], normalize_embeddings=True).tolist()

col = chromadb.PersistentClient(path=INDEX_DIR).get_collection(COLLECTION)
# where: metadata filter, used only when --source is given
where = {"source": args.source} if args.source else None
res = col.query(query_embeddings=q_vec, n_results=TOP_K, where=where)

# Chroma returns lists of lists (one list per question) -> take [0]
docs = res["documents"][0]
metas = res["metadatas"][0]
dists = res["distances"][0]

# ===== 3) Build the prompt with numbered sources =====
context = "\n\n".join(f"[{i + 1}] ({m['source']}, {m['date']})\n{d}"
                      for i, (d, m) in enumerate(zip(docs, metas)))
system = ("너는 사용자의 개인 기록을 바탕으로 답하는 비서야. "
          "반드시 아래 [자료] 안의 내용만 근거로 한국어로 답해. "
          "근거로 쓴 자료 번호를 [1]처럼 문장 끝에 붙여. "
          "자료에 없는 내용이면 '기록에서 찾지 못했어요'라고 말해.")
user = f"[자료]\n{context}\n\n[질문]\n{args.question}"

# ===== 4) Ask Ollama =====
# stream=False: wait for the full answer at once
resp = requests.post(OLLAMA_URL, json={
    "model": LLM,
    "messages": [{"role": "system", "content": system},
                 {"role": "user", "content": user}],
    "stream": False,
}, timeout=600)
answer = resp.json()["message"]["content"]

# ===== 5) Print answer and sources =====
print("\n=== 답변 ===")
print(answer)
print("\n=== 근거 ===")
for i, (m, dist) in enumerate(zip(metas, dists)):
    # cosine distance: 0 = same meaning, bigger = less similar
    link = m.get("url") or m.get("doc_id")
    print(f"[{i + 1}] {m['source']:9} {m['date']:16} dist={dist:.3f}  {link}")


# → 질문을 벡터로 바꿔 ChromaDB에서 가장 비슷한 조각 5개를 찾고,
#   그 조각들에 번호를 붙여 "이 자료만 근거로 답하라"는 지시와 함께 로컬 Ollama에 보내겠다.
#   답변과 함께 각 근거의 출처·날짜·거리·링크를 출력한다.