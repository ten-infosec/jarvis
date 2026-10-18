# ===== Shared helpers for the book library indexes (paths, tokenizer, chunking, stores) =====
# Every library script imports this file, so the tokenizer / chunk rules / paths stay identical.
# Nothing here touches the network: models load from the local Hugging Face cache only.


import hashlib
import json
import os
import re
import sqlite3
from bisect import bisect_right
from collections import defaultdict
from pathlib import Path


# ===== Settings (same values as app/rag.py where they overlap) =====
MODEL_NAME = "intfloat/multilingual-e5-small"
# Chroma collection names: whole text chunks / one-entry-per-book overviews
FULL_COLLECTION = "library_v0"
BOOKS_COLLECTION = "library_books"
# 800-character chunks with 100 characters shared between neighbours
CHUNK_SIZE = 800
OVERLAP = 100
# Fusion settings copied from rag.py (chosen in E3: vector 1.5 : BM25 1)
CANDIDATES = 50
RRF_K = 60
W_VEC = 1.5
W_BM25 = 1.0
# The book-overview index is also a vector signal, but coarser, so it gets a smaller weight
W_BOOKS = 1.0

# Request words removed from questions (same list as rag.py) + a few English filler words
STOPWORDS = {
    "내가", "나", "제가", "내", "저장한", "저장했던", "저장된", "저장",
    "관련", "관련된", "관한", "대한", "게시물", "게시글", "포스트", "글",
    "알려줘", "찾아줘", "보여줘", "정리해줘", "추천해줘",
    "뭐", "뭐야", "어떤", "있어", "있는", "좀", "것", "거",
    "책", "어느", "어디", "나와", "나와있는", "설명", "설명한",
    "the", "a", "an", "of", "and", "or", "in", "on", "to", "is", "what", "which", "book",
}


# ===== Paths (read from environment variables at call time, Windows defaults) =====
def get_paths():
    # JARVIS_LIBRARY_DIR: where manifest.jsonl and text\<id>.txt were written by the extractor
    library_dir = Path(os.getenv("JARVIS_LIBRARY_DIR", r"C:\jarvis_data\library"))
    # JARVIS_LIBRARY_INDEX: where all indexes go (default: <library>\index)
    index_dir = Path(os.getenv("JARVIS_LIBRARY_INDEX", str(library_dir / "index")))
    # JARVIS_NOTES_OUT: where extract_notes.py writes the notes chunks
    notes_dir = Path(os.getenv("JARVIS_NOTES_OUT", r"C:\jarvis_data\notes"))
    return {
        "library_dir": library_dir,
        "manifest": library_dir / "manifest.jsonl",
        "text_dir": library_dir / "text",
        "index_dir": index_dir,
        "fts_db": index_dir / "library_fts.sqlite3",
        "chunks_jsonl": index_dir / "chunks.jsonl",
        "chroma_dir": index_dir / "chroma",
        "notes_dir": notes_dir,
        "notes_chunks": notes_dir / "notes_chunks.jsonl",
    }


# ===== Text helpers (tokenize / clean_query are the same rules as rag.py) =====
def tokenize(text):
    # Lowercase so "MRI" and "mri" are the same token
    text = text.lower()
    # Words = runs of Hangul syllables, or runs of latin letters / digits
    words = re.findall(r"[가-힣]+|[a-z0-9]+", text)
    tokens = []
    for w in words:
        tokens.append(w)
        # Korean words longer than 2 also add 2-grams, so "폐렴은" still matches "폐렴"
        if re.match(r"[가-힣]", w) and len(w) > 2:
            tokens += [w[i:i + 2] for i in range(len(w) - 1)]
    return tokens


def clean_query(question):
    # Split the question the same way as the tokenizer, keeping the original case for display
    words = re.findall(r"[가-힣]+|[a-zA-Z0-9]+", question)
    # Drop "how to answer" words that would match random pages
    kept = [w for w in words if w.lower() not in STOPWORDS]
    # If everything was a stopword, search with the raw question instead
    return " ".join(kept) if kept else question


def normalize_ws(text):
    # PDF text has many broken lines and repeated spaces; collapse every whitespace run to one space
    return re.sub(r"\s+", " ", text).strip()


def split_pages(raw):
    # The extractor separated pages with a form feed "\f"; page numbers are 1-based list positions + 1
    return raw.split("\f")


def read_text(path):
    # errors="replace": a single bad byte must not stop an overnight job
    with open(path, encoding="utf-8", errors="replace") as f:
        return f.read()


# ===== Chunking with page numbers =====
def chunk_pages(pages, size=CHUNK_SIZE, overlap=OVERLAP):
    """Cut a list of page strings into overlapping chunks.

    Returns a list of dicts: {"n", "page_start", "page_end", "text"} (pages are 1-based).
    Empty pages are kept in the numbering so page numbers match the PDF.
    """
    # Normalize every page first, so offsets below refer to the exact text we store
    clean = [normalize_ws(p) for p in pages]
    # starts[i] = character offset where page i begins inside the joined text
    starts, pos = [], 0
    for p in clean:
        starts.append(pos)
        # +1 for the "\n" separator placed between pages
        pos += len(p) + 1
    # One long string; "\n" marks page boundaries (whitespace, so it never starts a word)
    full = "\n".join(clean)
    if not full.strip():
        return []

    def page_of(offset):
        # bisect_right finds the last page whose start is <= offset (0-based -> +1 makes it 1-based)
        return bisect_right(starts, offset)

    chunks, start, step = [], 0, size - overlap
    while start < len(full):
        end = min(start + size, len(full))
        piece = full[start:end]
        stripped = piece.strip()
        if stripped:
            # Page of the first / last real character (ignores separators at the edges)
            first = start + (len(piece) - len(piece.lstrip()))
            last = start + len(piece.rstrip()) - 1
            chunks.append({"n": len(chunks), "page_start": page_of(first),
                           "page_end": page_of(last), "text": stripped})
        # Stop once a chunk reached the end, so no tiny duplicate tail chunk is made
        if end >= len(full):
            break
        start += step
    return chunks


def short_hash(text, n=10):
    # Content fingerprint used for notes chunk ids (changed text -> new id -> re-embedded)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:n]


# ===== Manifest =====
def iter_manifest(path):
    # Stream line by line: never holds the whole manifest in memory
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def folder_matches(folder, wanted):
    # "08_Medicine" matches "08_Medicine" and sub folders like "08_Medicine\Radiology"
    f = (folder or "").replace("\\", "/").lower()
    w = wanted.replace("\\", "/").lower().rstrip("/")
    return f == w or f.startswith(w + "/")


# ===== SQLite FTS5 store (BM25 on disk) =====
SCHEMA = """
CREATE TABLE IF NOT EXISTS docs(
    doc_id TEXT PRIMARY KEY, source TEXT, title TEXT, folder TEXT,
    path TEXT, pages INTEGER, n_chunks INTEGER);
CREATE TABLE IF NOT EXISTS chunks(
    rid INTEGER PRIMARY KEY, chunk_id TEXT UNIQUE, doc_id TEXT,
    page_start INTEGER, page_end INTEGER, text TEXT);
CREATE INDEX IF NOT EXISTS chunks_doc ON chunks(doc_id);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    tokens, content='', tokenize='unicode61 remove_diacritics 0');
"""


def open_fts(db_path, create=True):
    db_path = Path(db_path)
    if not create and not db_path.exists():
        return None
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    # WAL + NORMAL sync: much faster bulk inserts, still safe if the PC is switched off mid-run
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.executescript(SCHEMA)
    return conn


def insert_doc(conn, meta, chunks):
    """Insert one document and all its chunks in ONE transaction (all-or-nothing = resumable)."""
    with conn:
        conn.execute("INSERT OR REPLACE INTO docs VALUES (?,?,?,?,?,?,?)",
                     (meta["doc_id"], meta["source"], meta.get("title", ""), meta.get("folder", ""),
                      meta.get("path", ""), meta.get("pages", 0), len(chunks)))
        for c in chunks:
            cur = conn.execute(
                "INSERT INTO chunks(chunk_id, doc_id, page_start, page_end, text) VALUES (?,?,?,?,?)",
                (c["id"], meta["doc_id"], c["page_start"], c["page_end"], c["text"]))
            # The FTS row shares the rowid with the chunks row; it stores only the token index
            conn.execute("INSERT INTO chunks_fts(rowid, tokens) VALUES (?, ?)",
                         (cur.lastrowid, " ".join(tokenize(c["text"]))))


def remove_doc(conn, doc_id):
    """Delete a document. Contentless FTS5 needs the original tokens to delete, so recompute them."""
    with conn:
        for rid, text in conn.execute("SELECT rid, text FROM chunks WHERE doc_id=?", (doc_id,)).fetchall():
            conn.execute("INSERT INTO chunks_fts(chunks_fts, rowid, tokens) VALUES ('delete', ?, ?)",
                         (rid, " ".join(tokenize(text))))
        conn.execute("DELETE FROM chunks WHERE doc_id=?", (doc_id,))
        conn.execute("DELETE FROM docs WHERE doc_id=?", (doc_id,))


def fts_query_string(query):
    # Each token quoted (so FTS5 never reads it as an operator) and joined with OR, like BM25 bag-of-words
    toks = list(dict.fromkeys(t for t in tokenize(query) if t not in STOPWORDS))
    return " OR ".join(f'"{t}"' for t in toks)


def fts_search(conn, query, limit=CANDIDATES * 4, folder=None, source=None):
    """Return chunk rows best-first: (chunk_id, doc_id, page_start, page_end, text, title, folder, source)."""
    match = fts_query_string(query)
    if not match:
        return []
    # bm25() in FTS5 is Okapi BM25; smaller = better, so ORDER BY rank ascending
    rows = conn.execute(
        "SELECT c.chunk_id, c.doc_id, c.page_start, c.page_end, c.text, d.title, d.folder, d.source "
        "FROM chunks_fts JOIN chunks c ON c.rid = chunks_fts.rowid JOIN docs d ON d.doc_id = c.doc_id "
        "WHERE chunks_fts MATCH ? ORDER BY rank LIMIT ?",
        (match, limit * 4 if (folder or source) else limit)).fetchall()
    # Filters are applied after ranking (simple, and the over-fetch above keeps enough candidates)
    if folder:
        rows = [r for r in rows if folder_matches(r[6], folder)]
    if source:
        rows = [r for r in rows if r[7] == source]
    return rows[:limit]


# ===== Embedding model (loaded lazily; tests replace load_embedder with a fake) =====
class E5Embedder:
    def __init__(self, model_name=MODEL_NAME):
        # Imported here so BM25-only runs never need torch / sentence-transformers
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        # e5 models need "passage: " / "query: " prefixes; other models do not
        self.e5 = "e5" in model_name

    def passages(self, texts, batch_size=32):
        prefix = "passage: " if self.e5 else ""
        return self.model.encode([prefix + t for t in texts], batch_size=batch_size,
                                 normalize_embeddings=True).tolist()

    def query(self, text):
        prefix = "query: " if self.e5 else ""
        return self.model.encode([prefix + text], normalize_embeddings=True).tolist()[0]


def load_embedder(model_name=MODEL_NAME):
    return E5Embedder(model_name)


# ===== Chroma store =====
def chroma_client(chroma_dir, create=True):
    chroma_dir = Path(chroma_dir)
    if not create and not chroma_dir.exists():
        return None
    import chromadb
    from chromadb.config import Settings
    chroma_dir.mkdir(parents=True, exist_ok=True)
    # anonymized_telemetry=False: no usage pings leave the PC
    return chromadb.PersistentClient(path=str(chroma_dir), settings=Settings(anonymized_telemetry=False))


def open_collection(chroma_dir, name, create=True):
    """Return the collection, or None when it (or chromadb itself) is not available and create=False."""
    try:
        client = chroma_client(chroma_dir, create=create)
    except ImportError:
        if create:
            raise
        return None
    if client is None:
        return None
    if create:
        # cosine space + no built-in embedding function: we always pass our own vectors
        return client.get_or_create_collection(name, metadata={"hnsw:space": "cosine"},
                                               embedding_function=None)
    try:
        return client.get_collection(name, embedding_function=None)
    except Exception:
        return None


def existing_ids(col, page=5000):
    """All ids already stored, read in pages (one call per 5,000 ids instead of one per batch)."""
    ids, offset = set(), 0
    while True:
        got = col.get(include=[], limit=page, offset=offset)["ids"]
        ids.update(got)
        if len(got) < page:
            return ids
        offset += page


# ===== Weighted reciprocal rank fusion =====
def weighted_rrf(ranked_lists, k=RRF_K):
    """ranked_lists: [(list_of_ids_best_first, weight), ...] -> [(id, score)] best first."""
    fused = defaultdict(float)
    for ranked, w in ranked_lists:
        for rank, d in enumerate(ranked, start=1):
            fused[d] += w / (k + rank)
    # Ties broken by id so the output is stable between runs
    return sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))


def fmt_eta(seconds):
    seconds = int(max(seconds, 0))
    h, rem = divmod(seconds, 3600)
    return f"{h}h {rem // 60:02d}m" if h else f"{rem // 60}m {rem % 60:02d}s"
