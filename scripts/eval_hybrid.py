# ===== E2: compare vector / BM25 / hybrid retrieval =====
# Usage: python scripts/eval_hybrid.py --tag E2


import argparse
import csv
import json
import re
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
# Only collections with at least 9 posts are used as test questions (same as v0)
MIN_SIZE = 9
# RRF constant: 1 / (60 + rank). 60 is the common default from the original paper
RRF_K = 60

parser = argparse.ArgumentParser()
parser.add_argument("--k", type=int, default=10)
parser.add_argument("--tag", default="E2")
parser.add_argument("--model", default="intfloat/multilingual-e5-small")
parser.add_argument("--collection", default="insta_v0")
args = parser.parse_args()

Q_PREFIX = "query: " if "e5" in args.model else ""


# ===== Tokenizer =====
def tokenize(text):
    # Lowercase so "ML" and "ml" count as the same word
    text = text.lower()
    # Pull out Korean words ([가-힣]+) and English/number words ([a-z0-9]+); drop symbols and emoji
    words = re.findall(r"[가-힣]+|[a-z0-9]+", text)
    tokens = []
    for w in words:
        tokens.append(w)
        # Korean words carry particles ("의학공부를", "의학은"), so exact matching fails.
        # Add 2-character pieces ("의학", "학공", "공부", "부를") so partial matches still score
        if re.match(r"[가-힣]", w) and len(w) > 2:
            tokens += [w[i:i + 2] for i in range(len(w) - 1)]
    return tokens


# ===== 1) BM25 index over saved posts (document level) =====
urls, tokenized_docs = [], []
with open(CORPUS, encoding="utf-8") as f:
    for line in f:
        doc = json.loads(line)
        # Same scope as the evaluation: saved posts only
        if doc["source"] != "saved":
            continue
        url = doc.get("meta", {}).get("url", "")
        if not url:
            continue
        urls.append(url)
        tokenized_docs.append(tokenize(doc["text"]))
# BM25Okapi builds word statistics (how often, how rare) from all documents at once
bm25 = BM25Okapi(tokenized_docs)
print(f"BM25 documents: {len(urls):,}")

# ===== 2) Vector search (existing index) =====
model = SentenceTransformer(args.model)
col = chromadb.PersistentClient(path=INDEX_DIR).get_collection(args.collection)

# ===== 3) Ground truth: collection name -> set of URLs =====
truth = defaultdict(set)
with open(COLLECTIONS_CSV, encoding="utf-8-sig") as f:
    for r in csv.DictReader(f):
        truth[r["collection"].strip()].add(r["url"])
tests = {name: urls_ for name, urls_ in truth.items() if len(urls_) >= MIN_SIZE}


# ===== Ranking functions =====
def bm25_rank(query, n):
    # One score per document; 0 means no shared word at all
    scores = bm25.get_scores(tokenize(query))
    # Document indexes sorted from highest score to lowest
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
    # Ask for n*5 chunks because several chunks can belong to the same post
    res = col.query(query_embeddings=q_vec, n_results=n * 5, where={"source": "saved"})
    ranked = []
    for m in res["metadatas"][0]:
        if m["url"] and m["url"] not in ranked:
            ranked.append(m["url"])
    return ranked[:n]


def rrf(*ranked_lists):
    # Add 1 / (RRF_K + rank) from every list; posts ranked high in both lists win
    score = defaultdict(float)
    for ranked in ranked_lists:
        for rank, url in enumerate(ranked, start=1):
            score[url] += 1 / (RRF_K + rank)
    return sorted(score, key=lambda u: -score[u])


def metrics(top, gold, k):
    hits = [u in gold for u in top]
    first = next((i + 1 for i, h in enumerate(hits) if h), None)
    return {
        "hit": int(any(hits)),
        "precision": round(sum(hits) / k, 3),
        "recall": round(sum(hits) / len(gold), 3),
        "rr": round(1 / first, 3) if first else 0,
    }


# ===== 4) Evaluate three modes on the same questions =====
MODES = ["vector", "bm25", "hybrid"]
rows = []
table = []
for name, gold in tests.items():
    # Take 50 candidates from each side so the fusion has enough to work with
    v = vector_rank(name, 50)
    b = bm25_rank(name, 50)
    h = rrf(v, b)
    line = {"collection": name, "size": len(gold)}
    for mode, ranked in zip(MODES, [v, b, h]):
        m = metrics(ranked[:args.k], gold, args.k)
        line[mode] = m
        rows.append({"tag": f"{args.tag}_{mode}", "k": args.k, "collection": name,
                     "size": len(gold), **m})
    table.append(line)

# ===== 5) Print P@k and RR per collection, then averages =====
print(f"\n{'collection':24} {'size':>4}   {'P vec':>6} {'P bm25':>6} {'P hyb':>6}   {'RR vec':>6} {'RR bm25':>7} {'RR hyb':>6}")
for t in sorted(table, key=lambda t: -t["hybrid"]["precision"]):
    print(f"{t['collection'][:24]:24} {t['size']:4d}   "
          f"{t['vector']['precision']:6.2f} {t['bm25']['precision']:6.2f} {t['hybrid']['precision']:6.2f}   "
          f"{t['vector']['rr']:6.2f} {t['bm25']['rr']:7.2f} {t['hybrid']['rr']:6.2f}")

print()
for mode in MODES:
    sub = [t[mode] for t in table]
    n = len(sub)
    avg = {key: sum(s[key] for s in sub) / n for key in ["hit", "precision", "recall", "rr"]}
    print(f"[{args.tag}_{mode:6}] k={args.k}  Hit={avg['hit']:.3f}  P@k={avg['precision']:.3f}  "
          f"R@k={avg['recall']:.3f}  MRR={avg['rr']:.3f}")

# ===== 6) Append to the same results file as v0 / E1 =====
with open(RESULTS_CSV, "a", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writerows(rows)


# → 저장 게시물로 BM25(단어 일치) 검색기를 만들고, 기존 벡터 검색 결과와 RRF로 합쳐서
#   같은 15개 질문에 대해 vector / bm25 / hybrid 세 방식의 점수를 한 번에 비교하겠다.
#   한국어는 조사 때문에 단어가 정확히 맞지 않으니 두 글자 조각을 같이 넣어서 부분 일치도 잡는다.
#   결과는 E2_vector, E2_bm25, E2_hybrid 이름으로 eval_results.csv에 누적한다.