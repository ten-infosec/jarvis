# ===== Step 1: keyword (BM25) index of the book library, stored in one SQLite file =====
# Usage (PowerShell, C:\dev\jarvis, venv active):
#   python scripts\library\build_library_fts.py                       # all folders
#   python scripts\library\build_library_fts.py --folders 08_Medicine # only these folders
# Resumable: books already in the index are skipped. Runs locally only; nothing is uploaded.

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from library_common import (chunk_pages, folder_matches, get_paths, insert_doc, iter_manifest,
                            open_fts, read_text, split_pages)


# ===== Options =====
ap = argparse.ArgumentParser()
# --folders: limit to these top folders (space separated); empty = every folder
ap.add_argument("--folders", nargs="*", default=[])
args = ap.parse_args()

paths = get_paths()
conn = open_fts(paths["fts_db"])
# Books already indexed (a book is inserted in one transaction, so "present" = complete)
done = {r[0] for r in conn.execute("SELECT doc_id FROM docs")}


# ===== Index every text book not yet indexed =====
start, n_books, n_chunks = time.time(), 0, 0
for m in iter_manifest(paths["manifest"]):
    # Scanned / failed PDFs have no usable text (they need OCR first)
    if m["status"] != "ok" or m["id"] in done:
        continue
    if args.folders and not any(folder_matches(m["folder"], f) for f in args.folders):
        continue
    text_file = paths["text_dir"] / f"{m['id']}.txt"
    if not text_file.exists():
        continue
    # Chunk with page numbers, then give each chunk a stable id "<book id>-<n>"
    chunks = chunk_pages(split_pages(read_text(text_file)))
    for c in chunks:
        c["id"] = f"{m['id']}-{c['n']}"
    meta = {"doc_id": m["id"], "source": "library", "title": m["title"],
            "folder": m["folder"], "path": m["path"], "pages": m["pages"]}
    insert_doc(conn, meta, chunks)
    n_books += 1
    n_chunks += len(chunks)
    if n_books % 20 == 0:
        print(f"{n_books} books, {n_chunks} chunks, {(time.time() - start) / 60:.1f} min", flush=True)

total = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
print(f"done: +{n_books} books, +{n_chunks} chunks (index total {total} chunks) -> {paths['fts_db']}")
