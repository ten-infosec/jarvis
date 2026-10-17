# ===== Build the vector index: chunk -> embed -> store in ChromaDB =====
# Usage: python scripts/build_index.py            (all documents)
#        python scripts/build_index.py --limit 500 (quick speed test)


# argparse: read options like --limit from the command line
import argparse
# json: read corpus.jsonl line by line
import json
# time: measure how long embedding takes
import time
from pathlib import Path

# chromadb: the local vector database
import chromadb
# SentenceTransformer: loads the embedding model and turns text into vectors
from sentence_transformers import SentenceTransformer


# ===== Settings =====
CORPUS = Path(r"C:\instagram_export\output\corpus.jsonl")
# Index lives next to the data, never inside the code repo
INDEX_DIR = Path(r"C:\instagram_export\index\chroma")
MODEL_NAME = "intfloat/multilingual-e5-small"
COLLECTION = "insta_v0"
# CHUNK_SIZE: max characters per chunk / OVERLAP: characters shared with the next chunk
CHUNK_SIZE = 400
OVERLAP = 50
# BATCH: how many chunks to embed at once (bigger = faster but more memory)
BATCH = 64


# ===== Chunking =====
# chunk_text(): cut long text into overlapping pieces
def chunk_text(text, size=CHUNK_SIZE, overlap=OVERLAP):
    text = text.strip()
    # Short text: keep as one chunk
    if len(text) <= size:
        return [text]
    chunks = []
    start = 0
    # Move forward by (size - overlap) each time so neighbours share `overlap` chars
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return chunks


# ===== Main =====
parser = argparse.ArgumentParser()
# --limit N: only use the first N documents (for a quick test)
parser.add_argument("--limit", type=int, default=0)
args = parser.parse_args()

# 1) Read documents and cut them into chunks
ids, texts, metas = [], [], []
with open(CORPUS, encoding="utf-8") as f:
    for n, line in enumerate(f):
        if args.limit and n >= args.limit:
            break
        doc = json.loads(line)
        for j, piece in enumerate(chunk_text(doc["text"])):
            ids.append(f"{doc['id']}#{j}")
            texts.append(piece)
            # Chroma metadata accepts only str/int/float/bool -> join lists into one string
            m = doc.get("meta", {})
            metas.append({
                "doc_id": doc["id"],
                "source": doc["source"],
                "date": doc.get("date", ""),
                "url": m.get("url", ""),
                "owner": m.get("owner", ""),
                "collections": " | ".join(m.get("collections", [])),
            })
print(f"Chunks to index: {len(texts):,}")

# 2) Load the embedding model (downloads ~470MB the first time)
model = SentenceTransformer(MODEL_NAME)

# 3) Open (or create) the local Chroma database
INDEX_DIR.mkdir(parents=True, exist_ok=True)
client = chromadb.PersistentClient(path=str(INDEX_DIR))
# cosine: compare vectors by angle (standard for sentence embeddings)
col = client.get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})

# 4) Embed and store batch by batch; skip batches already stored (lets you resume)
start_time = time.time()
done = 0
for b in range(0, len(texts), BATCH):
    batch_ids = ids[b:b + BATCH]
    # Resume: if the last id of this batch is already stored, skip the batch
    if col.get(ids=[batch_ids[-1]])["ids"]:
        continue
    # e5 rule: documents must start with "passage: "
    vectors = model.encode(["passage: " + t for t in texts[b:b + BATCH]],
                           normalize_embeddings=True)
    col.add(ids=batch_ids, documents=texts[b:b + BATCH],
            metadatas=metas[b:b + BATCH], embeddings=vectors.tolist())
    done += len(batch_ids)
    # Progress line every ~20 batches
    if (b // BATCH) % 20 == 0:
        elapsed = time.time() - start_time
        speed = done / elapsed if elapsed else 0
        remain = (len(texts) - b - BATCH) / speed / 60 if speed else 0
        print(f"  {b + len(batch_ids):,}/{len(texts):,}  {speed:.1f} chunks/s  ~{remain:.0f} min left")

print(f"Done. Stored in collection: {col.count():,} chunks  ({(time.time() - start_time) / 60:.1f} min)")


# → corpus.jsonl의 문서를 400자 조각(앞뒤 50자 겹침)으로 자르고,
#   e5-small 모델로 조각마다 벡터를 만들어 출처 정보와 함께 ChromaDB에 64개씩 저장하겠다.
#   이미 저장된 묶음은 건너뛰어서 중간에 멈춰도 이어서 할 수 있고, 진행률과 남은 시간을 보여 준다.