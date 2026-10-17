# ===== E2/E3/E6: vector / BM25 / weighted hybrid / + reranker =====
# Usage: python scripts/eval_hybrid.py --tag E3 --weights 1,1.5,2,3
#        python scripts/eval_hybrid.py --tag E6 --rerank


import argparse
import csv
import json
import re
import time
from collections import defaultdict
from pathlib import Path

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


# ===== Settings =====
CORPUS = Path(r"C:\instagram_export\output\corpus.jsonl")
INDEX_DIR = r"C:\instagram_export\index\chroma"
COLLECTIONS_CSV = Path(r"C:\instagram_export\output\collections.csv")
RESULTS_CSV = Path(r"C:\instagram_export\output\eval_results.csv")
MIN_SIZE = 9
RRF_K = 60
# Text given to the reranker per post (captions put the key info first)
RERANK_CHARS = 512

parser = argparse.ArgumentParser()
parser.add_argument("--k", type=int, default=10)
parser.add_argument("--tag", default="E6")
parser.add_argument("--model", default="intfloat/multilingual-e5-small")
parser.add_argument("--collection", default="insta_v0")
# Default is now the weight chosen in E3
parser.add_argument("--weights", default="1.5")
# --rerank: also test "hybrid + reranker"
parser.add_argument("--rerank", action="store_true")
parser.add_argument("--rerank-model", default="BAAI/bge-reranker-v2-m3")
# How many top hybrid results the reranker re-judges
parser.add_argument("--rerank-n", type=int, default=30)
args = parser.parse_args()

Q_PREFIX = "query: " if "e5" in args.model else ""
WEIGHTS = [float(w) for w in args.weights.split(",")]


# ===== Tokenizer =====
def tokenize(text):
    text = text.lower()
    words = re.findall(r"[가-힣]+|[a-z0-9]+", text)
    tokens = []
    for w in words:
        tokens.append(w)
        if re.match(r"[가-힣]", w) and len(w) > 2:
            tokens += [w[i:i + 2] for i in range(len(w) - 1)]
    return tokens


# ===== 1) BM25 index over saved posts (+ text lookup for the reranker) =====
urls, tokenized_docs = [], []
# url_text: post URL -> first 512 characters, so the reranker can read each candidate
url_text = {}
with open(CORPUS, encoding="utf-8") as f:
    for line in f:
        doc = json.loads(line)
        if doc["source"] != "saved":
            continue
        url = doc.get("meta", {}).get("url", "")
        if not url:
            continue
        urls.append(url)
        tokenized_docs.append(tokenize(doc["text"]))
        url_text[url] = doc["text"][:RERANK_CHARS]
bm25 = BM25Okapi(tokenized_docs)
print(f"BM25 documents: {len(urls):,}")

# ===== 2) Vector search =====
model = SentenceTransformer(args.model)
col = chromadb.PersistentClient(path=INDEX_DIR).get_collection(args.collection)

# ===== 3) Reranker (only loaded when --rerank is given) =====
reranker = None
if args.rerank:
    # CrossEncoder reads (question, text) together and returns one relevance score
    from sentence_transformers import CrossEncoder
    reranker = CrossEncoder(args.rerank_model, max_length=512)

# ===== 4) Ground truth =====
truth = defaultdict(set)
with open(COLLECTIONS_CSV, encoding="utf-8-sig") as f:
    for r in csv.DictReader(f):
        truth[r["collection"].strip()].add(r["url"])
tests = {name: urls_ for name, urls_ in truth.items() if len(urls_) >= MIN_SIZE}


# ===== Ranking functions =====
def bm25_rank(query, n):
    scores = bm25.get_scores(tokenize(query))
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    ranked = []
    for i in order:
        if scores[i] <= 0:
            break
        if urls[i] not in ranked:
            ranked.append(urls[i])
        if len(ranked) >= n:
            break
    return ranked


def vector_rank(query, n):
    q_vec = model.encode([Q_PREFIX + query], normalize_embeddings=True).tolist()
    res = col.query(query_embeddings=q_vec, n_results=n * 5, where={"source": "saved"})
    ranked = []
    for m in res["metadatas"][0]:
        if m["url"] and m["url"] not in ranked:
            ranked.append(m["url"])
    return ranked[:n]


def rrf(ranked_lists, weights):
    score = defaultdict(float)
    for ranked, w in zip(ranked_lists, weights):
        for rank, url in enumerate(ranked, start=1):
            score[url] += w / (RRF_K + rank)
    return sorted(score, key=lambda u: -score[u])


def rerank(query, ranked):
    # Only the top N candidates are re-judged (the reranker is slow)
    cands = ranked[:args.rerank_n]
    # One (question, post text) pair per candidate
    pairs = [(query, url_text.get(u, "")) for u in cands]
    scores = reranker.predict(pairs)
    # Sort candidates by reranker score, highest first
    order = sorted(range(len(cands)), key=lambda i: -scores[i])
    return [cands[i] for i in order]


def metrics(top, gold, k):
    hits = [u in gold for u in top]
    first = next((i + 1 for i, h in enumerate(hits) if h), None)
    return {
        "hit": int(any(hits)),
        "precision": round(sum(hits) / k, 3),
        "recall": round(sum(hits) / len(gold), 3),
        "rr": round(1 / first, 3) if first else 0,
    }


# ===== 5) Evaluate =====
MODES = ["vector", "bm25"] + [f"w{w:g}" for w in WEIGHTS]
if args.rerank:
    # "_rr" = the same hybrid list, re-ordered by the reranker
    MODES += [f"w{w:g}_rr" for w in WEIGHTS]

rows, table = [], []
start = time.time()
for name, gold in tests.items():
    v = vector_rank(name, 50)
    b = bm25_rank(name, 50)
    results = {"vector": v, "bm25": b}
    for w in WEIGHTS:
        fused = rrf([v, b], [w, 1.0])
        results[f"w{w:g}"] = fused
        if args.rerank:
            results[f"w{w:g}_rr"] = rerank(name, fused)
    line = {"collection": name, "size": len(gold)}
    for mode in MODES:
        m = metrics(results[mode][:args.k], gold, args.k)
        line[mode] = m
        rows.append({"tag": f"{args.tag}_{mode}", "k": args.k, "collection": name,
                     "size": len(gold), **m})
    table.append(line)
    # Progress line, because the reranker takes a while on CPU
    print(f"  done: {name}  ({time.time() - start:.0f}s)")

# ===== 6) Print per-collection RR, then averages =====
header = "".join(f"{m:>9}" for m in MODES)
print(f"\nRR per collection (1.00 = first result is correct)")
print(f"{'collection':24} {'size':>4}{header}")
for t in table:
    cells = "".join(f"{t[m]['rr']:9.2f}" for m in MODES)
    print(f"{t['collection'][:24]:24} {t['size']:4d}{cells}")

print()
for mode in MODES:
    sub = [t[mode] for t in table]
    n = len(sub)
    avg = {key: sum(s[key] for s in sub) / n for key in ["hit", "precision", "recall", "rr"]}
    print(f"[{args.tag}_{mode:8}] k={args.k}  Hit={avg['hit']:.3f}  P@k={avg['precision']:.3f}  "
          f"R@k={avg['recall']:.3f}  MRR={avg['rr']:.3f}")

# ===== 7) Append to results file =====
with open(RESULTS_CSV, "a", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writerows(rows)


# → E3에서 고른 하이브리드(w1.5) 결과 상위 30개를, 질문과 글을 함께 읽는 리랭커로 다시 순위를 매기겠다.
#   리랭커 전(w1.5)과 후(w1.5_rr)를 같은 15개 질문으로 비교해서, 2차 심사가 실제로 정답을 위로 올리는지 확인한다.
#   리랭커는 CPU에서 느리니 컬렉션마다 진행 시간을 출력한다.