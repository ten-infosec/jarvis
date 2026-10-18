# ===== Step 2: meaning (vector) index of the book library, built from the keyword index chunks =====
# Usage (PowerShell, C:\dev\jarvis, venv active), after build_library_fts.py:
#   python scripts\library\build_library_vectors.py --folders 08_Medicine --max-minutes 120
# Resumable: chunk ids already in the vector store are skipped, so stop (Ctrl+C) and re-run any time.
# The e5 model runs on this PC; no text leaves the computer.

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import library_common as lc


# ===== Options =====
ap = argparse.ArgumentParser()
# --folders: embed only these folders (do the most useful ones first)
ap.add_argument("--folders", nargs="*", default=[])
# --max-minutes: stop cleanly after this long (0 = no limit)
ap.add_argument("--max-minutes", type=float, default=0)
# --batch: chunks per embedding call (bigger = faster, more memory)
ap.add_argument("--batch", type=int, default=64)
args = ap.parse_args()

paths = lc.get_paths()
conn = lc.open_fts(paths["fts_db"], create=False)
if conn is None:
    sys.exit("Keyword index not found. Run build_library_fts.py first.")
col = lc.open_collection(paths["chroma_dir"], lc.FULL_COLLECTION)
have = lc.existing_ids(col)


# ===== Pick chunks still missing from the vector store =====
rows = conn.execute("SELECT c.chunk_id, c.doc_id, c.page_start, c.page_end, c.text, d.title, d.folder "
                    "FROM chunks c JOIN docs d ON d.doc_id = c.doc_id ORDER BY c.rid").fetchall()
todo = [r for r in rows if r[0] not in have
        and (not args.folders or any(lc.folder_matches(r[6], f) for f in args.folders))]
print(f"to embed: {len(todo)} chunks (already stored: {len(have)})", flush=True)
if not todo:
    sys.exit(0)


# ===== Embed in batches with progress, ETA and a time budget =====
emb = lc.load_embedder()
start, done = time.time(), 0
try:
    for i in range(0, len(todo), args.batch):
        if args.max_minutes and (time.time() - start) / 60 >= args.max_minutes:
            print("time budget reached; re-run to continue")
            break
        batch = todo[i:i + args.batch]
        vecs = emb.passages([r[4] for r in batch])
        col.add(ids=[r[0] for r in batch], embeddings=vecs, documents=[r[4] for r in batch],
                metadatas=[{"doc_id": r[1], "page_start": r[2], "page_end": r[3],
                            "title": r[5], "folder": r[6]} for r in batch])
        done += len(batch)
        if (i // args.batch) % 20 == 0 or done == len(todo):
            rate = done / max(time.time() - start, 1e-9)
            print(f"{done}/{len(todo)}  {rate:.1f}/s  ETA {lc.fmt_eta((len(todo) - done) / max(rate, 1e-9))}",
                  flush=True)
except KeyboardInterrupt:
    print("stopped by user; progress is saved")
print(f"embedded {done} chunks in {lc.fmt_eta(time.time() - start)}")
