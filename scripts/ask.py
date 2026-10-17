# ===== Ask a question to the second brain (hybrid retrieval + local LLM) =====
# Usage: python scripts/ask.py "질문" --source saved
#        python scripts/ask.py "질문" --source saved --mode vector       (old way, for comparison)
#        python scripts/ask.py "질문" --source saved --retrieve-only     (show sources only, no LLM)


import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import chromadb
import requests
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


# ===== Settings =====
CORPUS = Path(r"C:\instagram_export\output\corpus.jsonl")
INDEX_DIR = r"C:\instagram_export\index\chroma"
MODEL_NAME = "intfloat/multilingual-e5-small"
COLLECTION = "insta_v0"
# Ollama runs on this computer only (localhost) -> private data never leaves the laptop
OLLAMA_URL = "http://localhost:11434/api/chat"
LLM = "qwen2.5:3b"
# How many posts go into the LLM prompt
TOP_N = 5
# How many candidates each search method brings before fusion
CANDIDATES = 50
RRF_K = 60
# Weights chosen in E3 (vector 1.5 : BM25 1)
W_VEC = 1.5
W_BM25 = 1.0
# For posts found only by BM25 (no matching chunk), use the first 600 characters
CONTEXT_CHARS = 600

parser = argparse.ArgumentParser()
# The question itself (positional = no "--" name needed)
parser.add_argument("question")
parser.add_argument("--source", default=None)
# hybrid (default) or vector (old behavior)
parser.add_argument("--mode", choices=["hybrid", "vector"], default="hybrid")
# store_true: the option is a switch; writing it means True
parser.add_argument("--retrieve-only", action="store_true")
args = parser.parse_args()


# ===== Tokenizer (same as eval_hybrid.py) =====
def tokenize(text):
    text = text.lower()
    words = re.findall(r"[가-힣]+|[a-z0-9]+", text)
    tokens = []
    for w in words:
        tokens.append(w)
        if re.match(r"[가-힣]", w) and len(w) > 2:
            tokens += [w[i:i + 2] for i in range(len(w) - 1)]
    return tokens

# ===== Query cleaning (stopwords) =====
# Request words that say HOW to answer, not WHAT to find.
# They match unrelated captions (e.g. "저장하고 보세요"), so remove them before searching
STOPWORDS = {
    "내가", "나", "제가", "내", "저장한", "저장했던", "저장된", "저장",
    "관련", "관련된", "관한", "대한", "게시물", "게시글", "포스트", "글",
    "알려줘", "찾아줘", "보여줘", "정리해줘", "추천해줘",
    "뭐", "뭐야", "어떤", "있어", "있는", "좀", "것", "거",
}


def clean_query(question):
    # Split into Korean / English / number words, the same way as the tokenizer
    words = re.findall(r"[가-힣]+|[a-zA-Z0-9]+", question)
    # Keep only words that are not in the stopword list (lowercase so "ML" == "ml")
    kept = [w for w in words if w.lower() not in STOPWORDS]
    # If everything was removed, fall back to the original question
    return " ".join(kept) if kept else question


query = clean_query(args.question)
print(f"Search query: {query}")

# ===== 1) Load documents (only the chosen source) =====
# docs: doc_id -> full document, so any search result can be looked up by id
docs = {}
with open(CORPUS, encoding="utf-8") as f:
    for line in f:
        doc = json.loads(line)
        if args.source and doc["source"] != args.source:
            continue
        docs[doc["id"]] = doc
doc_ids = list(docs)

# ===== 2) Vector search, collapsed to one entry per document =====
model = SentenceTransformer(MODEL_NAME)
col = chromadb.PersistentClient(path=INDEX_DIR).get_collection(COLLECTION)
q_vec = model.encode(["query: " + query], normalize_embeddings=True).tolist()
where = {"source": args.source} if args.source else None
# Ask for extra chunks because one post can have several chunks
res = col.query(query_embeddings=q_vec, n_results=CANDIDATES * 3, where=where)

vec_rank = []
# best_chunk: the closest chunk of each post (used as LLM context)
best_chunk = {}
for text, meta in zip(res["documents"][0], res["metadatas"][0]):
    d = meta["doc_id"]
    # Results are sorted closest-first, so the first chunk seen for a post is its best one.
    # Skipping later chunks of the same post fixes the [2]/[4] duplicate problem
    if d in docs and d not in best_chunk:
        best_chunk[d] = text
        vec_rank.append(d)
vec_rank = vec_rank[:CANDIDATES]

# ===== 3) BM25 search + weighted RRF fusion =====
bm_rank = []
if args.mode == "hybrid":
    bm25 = BM25Okapi([tokenize(docs[d]["text"]) for d in doc_ids])
    scores = bm25.get_scores(tokenize(query))
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    bm_rank = [doc_ids[i] for i in order[:CANDIDATES] if scores[i] > 0]

    fused = defaultdict(float)
    for ranked, w in [(vec_rank, W_VEC), (bm_rank, W_BM25)]:
        for rank, d in enumerate(ranked, start=1):
            fused[d] += w / (RRF_K + rank)
    final = sorted(fused, key=lambda d: -fused[d])[:TOP_N]
else:
    final = vec_rank[:TOP_N]

# ===== 4) Build numbered context =====
blocks, sources = [], []
for i, d in enumerate(final, start=1):
    doc = docs[d]
    text = best_chunk.get(d) or doc["text"][:CONTEXT_CHARS]
    # Which search found this post (useful for checking how hybrid helps)
    if d in best_chunk and d in bm_rank:
        via = "vec+bm25"
    elif d in best_chunk:
        via = "vec"
    else:
        via = "bm25"
    blocks.append(f"[{i}] ({doc['source']}, {doc.get('date', '')})\n{text}")
    sources.append((i, doc, via))
context = "\n\n".join(blocks)

# ===== 5) Ask the local LLM (skipped with --retrieve-only) =====
if not args.retrieve_only:
    system = (
        "너는 사용자의 개인 기록을 바탕으로 답하는 비서야. "
        "반드시 아래 [자료]에 있는 내용만 사용해서 한국어로 답해. "
        "문장마다 근거가 된 자료 번호를 [1]처럼 붙여. "
        "[자료]에 답이 없으면 '기록에서 찾지 못했어요'라고 말해."
    )
    user = f"[자료]\n{context}\n\n[질문]\n{args.question}"
    r = requests.post(OLLAMA_URL, json={
        "model": LLM,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "stream": False,
    }, timeout=600)
    r.raise_for_status()
    print("\n===== Answer =====")
    print(r.json()["message"]["content"])

# ===== 6) Show sources =====
print(f"\n===== Sources ({args.mode}) =====")
for i, doc, via in sources:
    url = doc.get("meta", {}).get("url", "")
    preview = doc["text"][:60].replace("\n", " ")
    print(f"[{i}] {doc['source']:8} {doc.get('date', '')[:10]}  {via:8}  {url}")
    print(f"     {preview}")


# → 질문을 벡터 검색과 BM25 검색으로 각각 50개씩 찾은 뒤, E3에서 고른 1.5 : 1 무게로 합쳐 상위 5개 게시물을 고르겠다.
#   같은 게시물의 조각은 가장 가까운 것 하나만 남겨 중복을 없애고, 그 조각을 근거 자료로 로컬 LLM(Ollama)에 넘겨 답을 받는다.
#   --mode vector로 예전 방식과 비교할 수 있고, --retrieve-only면 LLM 없이 검색 결과만 바로 보여 준다.