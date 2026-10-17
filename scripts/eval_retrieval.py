# ===== Evaluate retrieval with saved collections as ground truth =====
# Usage: python scripts/eval_retrieval.py --tag v0
#        python scripts/eval_retrieval.py --tag E1_bge_m3 --model BAAI/bge-m3 --collection saved_bge_m3


import argparse
import csv
from collections import defaultdict
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


INDEX_DIR = r"C:\instagram_export\index\chroma"
COLLECTIONS_CSV = Path(r"C:\instagram_export\output\collections.csv")
RESULTS_CSV = Path(r"C:\instagram_export\output\eval_results.csv")
MIN_SIZE = 9

parser = argparse.ArgumentParser()
parser.add_argument("--k", type=int, default=10)
parser.add_argument("--tag", default="v0")
parser.add_argument("--model", default="intfloat/multilingual-e5-small")
parser.add_argument("--collection", default="insta_v0")
args = parser.parse_args()

# e5 models need "query: " in front of questions; other models do not
Q_PREFIX = "query: " if "e5" in args.model else ""

# ===== Ground truth: collection name -> set of post URLs =====
truth = defaultdict(set)
with open(COLLECTIONS_CSV, encoding="utf-8-sig") as f:
    for r in csv.DictReader(f):
        truth[r["collection"].strip()].add(r["url"])
tests = {name: urls for name, urls in truth.items() if len(urls) >= MIN_SIZE}

model = SentenceTransformer(args.model)
col = chromadb.PersistentClient(path=INDEX_DIR).get_collection(args.collection)

rows = []
for name, gold in tests.items():
    q_vec = model.encode([Q_PREFIX + name], normalize_embeddings=True).tolist()
    res = col.query(query_embeddings=q_vec, n_results=args.k * 5, where={"source": "saved"})
    ranked = []
    for m in res["metadatas"][0]:
        if m["url"] and m["url"] not in ranked:
            ranked.append(m["url"])
    top = ranked[:args.k]

    hits = [u in gold for u in top]
    first = next((i + 1 for i, h in enumerate(hits) if h), None)
    rows.append({
        "tag": args.tag, "k": args.k, "collection": name, "size": len(gold),
        "hit": int(any(hits)),
        "precision": round(sum(hits) / args.k, 3),
        "recall": round(sum(hits) / len(gold), 3),
        "rr": round(1 / first, 3) if first else 0,
    })

print(f"{'collection':24} {'size':>4} {'hit':>4} {'P@k':>6} {'R@k':>6} {'RR':>6}")
for r in sorted(rows, key=lambda r: -r["precision"]):
    print(f"{r['collection'][:24]:24} {r['size']:4d} {r['hit']:4d} {r['precision']:6.2f} {r['recall']:6.2f} {r['rr']:6.2f}")
n = len(rows)
avg = {m: sum(r[m] for r in rows) / n for m in ["hit", "precision", "recall", "rr"]}
print(f"\n[{args.tag}] k={args.k}  Hit={avg['hit']:.3f}  P@k={avg['precision']:.3f}  "
      f"R@k={avg['recall']:.3f}  MRR={avg['rr']:.3f}  (collections={n})")

new_file = not RESULTS_CSV.exists()
with open(RESULTS_CSV, "a", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    if new_file:
        writer.writeheader()
    writer.writerows(rows)


# → 모델과 컬렉션을 옵션으로 받아, 컬렉션 이름을 질문으로 던졌을 때 상위 k개 중 정답 비율을 계산하겠다.
#   e5 모델일 때만 "query: "를 붙이고, 결과는 실험 이름과 함께 eval_results.csv에 누적한다.