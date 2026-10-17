# ===== Build the vector index: chunk -> embed -> store in ChromaDB =====
# Usage: python scripts/build_index.py
#        python scripts/build_index.py --model BAAI/bge-m3 --collection saved_bge_m3 --source saved


import argparse
import json
import time
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


# ===== Settings =====
CORPUS = Path(r"C:\instagram_export\output\corpus.jsonl")
INDEX_DIR = Path(r"C:\instagram_export\index\chroma")
CHUNK_SIZE = 400
OVERLAP = 50
BATCH = 64

parser = argparse.ArgumentParser()
# --limit N: only the first N documents (quick test)
parser.add_argument("--limit", type=int, default=0)
# --model: which embedding model to use
parser.add_argument("--model", default="intfloat/multilingual-e5-small")
# --collection: Chroma collection name (one per experiment)
parser.add_argument("--collection", default="insta_v0")
# --source: index only one source (saved / my_post / thread / subtitle)
parser.add_argument("--source", default=None)
args = parser.parse_args()

# e5 models need "passage: " in front of documents; other models do not
DOC_PREFIX = "passage: " if "e5" in args.model else ""


# ===== Chunking =====
def chunk_text(text, size=CHUNK_SIZE, overlap=OVERLAP):
    text = text.strip()
    if len(text) <= size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return chunks


# ===== 1) Read documents and cut into chunks =====
ids, texts, metas = [], [], []
with open(CORPUS, encoding="utf-8") as f:
    for n, line in enumerate(f):
        if args.limit and n >= args.limit:
            break
        doc = json.loads(line)
        # Skip other sources when --source is given
        if args.source and doc["source"] != args.source:
            continue
        for j, piece in enumerate(chunk_text(doc["text"])):
            ids.append(f"{doc['id']}#{j}")
            texts.append(piece)
            m = doc.get("meta", {})
            metas.append({
                "doc_id": doc["id"], "source": doc["source"], "date": doc.get("date", ""),
                "url": m.get("url", ""), "owner": m.get("owner", ""),
                "collections": " | ".join(m.get("collections", [])),
            })
print(f"Model: {args.model}  Collection: {args.collection}  Chunks to index: {len(texts):,}")

# ===== 2) Load model and open DB =====
model = SentenceTransformer(args.model)
INDEX_DIR.mkdir(parents=True, exist_ok=True)
client = chromadb.PersistentClient(path=str(INDEX_DIR))
col = client.get_or_create_collection(args.collection, metadata={"hnsw:space": "cosine"})

# ===== 3) Embed and store batch by batch (skips batches already stored) =====
start_time = time.time()
done = 0
for b in range(0, len(texts), BATCH):
    batch_ids = ids[b:b + BATCH]
    if col.get(ids=[batch_ids[-1]])["ids"]:
        continue
    vectors = model.encode([DOC_PREFIX + t for t in texts[b:b + BATCH]],
                           normalize_embeddings=True)
    col.add(ids=batch_ids, documents=texts[b:b + BATCH],
            metadatas=metas[b:b + BATCH], embeddings=vectors.tolist())
    done += len(batch_ids)
    if (b // BATCH) % 20 == 0:
        elapsed = time.time() - start_time
        speed = done / elapsed if elapsed else 0
        remain = (len(texts) - b - BATCH) / speed / 60 if speed else 0
        print(f"  {b + len(batch_ids):,}/{len(texts):,}  {speed:.1f} chunks/s  ~{remain:.0f} min left")

print(f"Done. Stored in collection: {col.count():,} chunks  ({(time.time() - start_time) / 60:.1f} min)")


# → 명령어 옵션으로 모델·컬렉션 이름·출처를 바꿀 수 있게 하고,
#   문서를 400자 조각으로 잘라 선택한 모델로 벡터를 만들어 ChromaDB의 지정한 컬렉션에 저장하겠다.
#   e5 모델일 때만 "passage: "를 붙이고, 이미 저장된 묶음은 건너뛴다.