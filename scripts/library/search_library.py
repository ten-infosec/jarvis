# ===== Search the book library: keyword + meaning, merged with weighted RRF =====
# Usage (PowerShell, C:\dev\jarvis, venv active):
#   python scripts\library\search_library.py "심방세동 치료"
#   python scripts\library\search_library.py "beta blocker" --folder 08_Medicine --top 5
# Works with only the keyword index (vectors are used when they exist).

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import library_common as lc


def search(query, folder=None, top=5, embedder=None):
    paths = lc.get_paths()
    q = lc.clean_query(query)
    rows = {}
    ranked = []

    # ----- Keyword candidates (SQLite FTS5 BM25) -----
    conn = lc.open_fts(paths["fts_db"], create=False)
    if conn is not None:
        hits = lc.fts_search(conn, q, limit=lc.CANDIDATES, folder=folder)
        for r in hits:
            rows[r[0]] = {"title": r[5], "folder": r[6], "pages": (r[2], r[3]), "text": r[4]}
        ranked.append(([r[0] for r in hits], lc.W_BM25))

    # ----- Meaning candidates (Chroma), only if the vector index exists -----
    col = lc.open_collection(paths["chroma_dir"], lc.FULL_COLLECTION, create=False)
    if col is not None and col.count() > 0:
        emb = embedder or lc.load_embedder()
        res = col.query(query_embeddings=[emb.query(q)], n_results=lc.CANDIDATES,
                        include=["documents", "metadatas"])
        ids = res["ids"][0]
        for cid, doc, meta in zip(ids, res["documents"][0], res["metadatas"][0]):
            if folder and not lc.folder_matches(meta.get("folder"), folder):
                continue
            rows.setdefault(cid, {"title": meta["title"], "folder": meta["folder"],
                                  "pages": (meta["page_start"], meta["page_end"]), "text": doc})
        ranked.append(([i for i in ids if i in rows], lc.W_VEC))

    # ----- Fuse, then keep the best chunk per book -----
    out, seen = [], set()
    for cid, score in lc.weighted_rrf(ranked):
        book = cid.rsplit("-", 1)[0]
        if book in seen:
            continue
        seen.add(book)
        out.append({"chunk_id": cid, "score": round(score, 4), **rows[cid]})
        if len(out) >= top:
            break
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--folder", default=None)
    ap.add_argument("--top", type=int, default=5)
    a = ap.parse_args()
    results = search(a.query, a.folder, a.top)
    if not results:
        print("no results (is the index built?)")
    for i, r in enumerate(results, 1):
        p1, p2 = r["pages"]
        print(f"[{i}] {r['title']}  ({r['folder']}, p.{p1}-{p2})")
        print("    " + r["text"][:200].replace("\n", " ") + "...")
