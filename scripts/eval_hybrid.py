# ===== E2/E3: compare vector / BM25 / weighted hybrid retrieval =====
# Usage: python scripts/eval_hybrid.py --tag E3
#        python scripts/eval_hybrid.py --tag E3 --weights 1,1.5,2,3


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
MIN_SIZE = 9
RRF_K = 60

parser = argparse.ArgumentParser()
parser.add_argument("--k", type=int, default=10)
parser.add_argument("--tag", default="E3")
parser.add_argument("--model", default="intfloat/multilingual-e5-small")
parser.add_argument("--collection", default="insta_v0")
# --weights: vector weights to try (BM25 weight is always 1). "1" reproduces E2_hybrid
parser.add_argument("--weights", default="1,1.5,2,3")
args = parser.parse_args()

Q_PREFIX = "query: " if "e5" in args.model else ""
# "1,1.5,2,3" -> [1.0, 1.5, 2.0, 3.0]
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


# ===== 1) BM25 index over saved posts =====
urls, tokenized_docs = [], []
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
bm25 = BM25Okapi(tokenized_docs)
print(f"BM25 documents: {len(urls):,}")

# ===== 2) Vector search =====
model = SentenceTransformer(args.model)
col = chromadb.PersistentClient(path=INDEX_DIR).get_collection(args.collection)

# ===== 3) Ground truth =====
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
    # Each list adds weight / (RRF_K + rank); a bigger weight makes that list count more
    score = defaultdict(float)
    for ranked, w in zip(ranked_lists, weights):
        for rank, url in enumerate(ranked, start=1):
            score[url] += w / (RRF_K + rank)
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


# ===== 4) Evaluate every mode on the same questions =====
# Mode names: vector, bm25, w1, w1.5, w2, w3 ... (":g" drops trailing ".0")
MODES = ["vector", "bm25"] + [f"w{w:g}" for w in WEIGHTS]
rows, table = [], []
for name, gold in tests.items():
    v = vector_rank(name, 50)
    b = bm25_rank(name, 50)
    results = {"vector": v, "bm25": b}
    # One fused ranking per vector weight (BM25 weight fixed at 1)
    for w in WEIGHTS:
        results[f"w{w:g}"] = rrf([v, b], [w, 1.0])
    line = {"collection": name, "size": len(gold)}
    for mode in MODES:
        m = metrics(results[mode][:args.k], gold, args.k)
        line[mode] = m
        rows.append({"tag": f"{args.tag}_{mode}", "k": args.k, "collection": name,
                     "size": len(gold), **m})
    table.append(line)

# ===== 5) Print per-collection RR (rank of first hit), then averages =====
header = "".join(f"{m:>8}" for m in MODES)
print(f"\nRR per collection (1.00 = first result is correct)")
print(f"{'collection':24} {'size':>4}{header}")
for t in table:
    cells = "".join(f"{t[m]['rr']:8.2f}" for m in MODES)
    print(f"{t['collection'][:24]:24} {t['size']:4d}{cells}")

print()
for mode in MODES:
    sub = [t[mode] for t in table]
    n = len(sub)
    avg = {key: sum(s[key] for s in sub) / n for key in ["hit", "precision", "recall", "rr"]}
    print(f"[{args.tag}_{mode:6}] k={args.k}  Hit={avg['hit']:.3f}  P@k={avg['precision']:.3f}  "
          f"R@k={avg['recall']:.3f}  MRR={avg['rr']:.3f}")

# ===== 6) Append to results file =====
with open(RESULTS_CSV, "a", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writerows(rows)


# → 벡터 쪽 무게를 1, 1.5, 2, 3으로 바꿔 가며 RRF로 합친 결과를 한 번에 비교하겠다.
#   w1은 E2_hybrid와 같은 값이 나와야 정상이고(검증용),
#   Hit(놓친 컬렉션 수)와 MRR(첫 정답 순위)을 함께 보면서 가장 균형 좋은 무게를 고른다.