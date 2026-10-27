"""Tests for the book-library routes in app/main.py: /ask and /ask/stream with source "library", /library/pdf/<id>.

Runs without models, network or private data (fake engines + a tiny temp library). Run from the repo root:
python -m pytest tests -v
"""
import importlib
import json
import sys
import types
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import library  # noqa: E402

BOOK = "a" * 16
MAIN_HIT = {"n": 1, "doc_id": "d1", "source": "saved", "date": "2026-01-01", "url": "https://x.example/1",
            "via": "bm25", "text": "main text"}
LIB_HIT = {"n": 1, "doc_id": BOOK, "source": "library", "date": "", "url": "", "via": "bm25", "text": "book text " * 30,
           "title": "책A", "book_id": BOOK, "page": 2, "pages": "2-3", "ocr": 88.5, "ocr_low": True,
           "secret": "must not leak"}


class FakeMain:
    # Main engine (Instagram / arXiv): records how it was called
    def __init__(self):
        self.docs = {"d1": "t"}
        self.calls = []

    def retrieve(self, question, source, mode, top_n):
        self.calls.append("retrieve")
        return "q", [dict(MAIN_HIT)]

    def generate(self, question, hits):
        self.calls.append("generate")
        return "main answer"


class FakeLibrary:
    def __init__(self, pdf=None):
        self.calls = []
        self.pdf = pdf

    def retrieve(self, question, source, mode, top_n):
        self.calls.append("retrieve")
        return "lq", [dict(LIB_HIT)]

    def generate(self, question, hits):
        return "library answer"

    def generate_stream(self, question, hits):
        yield "library "
        yield "answer"

    def pdf_path(self, book_id):
        return self.pdf if book_id == BOOK else None


def load_main(monkeypatch, tmp_path, mode):
    monkeypatch.setenv("JARVIS_DATA_DIR", str(tmp_path))
    if mode == "cloud":
        monkeypatch.setenv("JARVIS_MODE", "cloud")
        fake = types.ModuleType("app.rag_cloud")
        fake.CloudRAG = FakeMain
        monkeypatch.setitem(sys.modules, "app.rag_cloud", fake)
    else:
        monkeypatch.delenv("JARVIS_MODE", raising=False)
        fake = types.ModuleType("app.rag")
        fake.RAG = FakeMain
        monkeypatch.setitem(sys.modules, "app.rag", fake)
    sys.modules.pop("app.main", None)
    main = importlib.import_module("app.main")
    main._recent.clear()
    main._daily.update(date=None, count=0)
    return main


def ndjson(res):
    return [json.loads(line) for line in res.text.splitlines() if line.strip()]


# ===== 1. Local: source "library" goes to the library engine only, and book fields reach the page =====
def test_ask_library_uses_library_engine(monkeypatch, tmp_path):
    main = load_main(monkeypatch, tmp_path, "local")
    main._library = FakeLibrary()
    res = TestClient(main.app).post("/ask", json={"question": "심방세동 치료", "source": "library"})
    assert res.status_code == 200
    body = res.json()
    assert body["answer"] == "library answer" and main.rag.calls == []
    s = body["sources"][0]
    assert (s["title"], s["book_id"], s["page"], s["pages"], s["ocr"], s["ocr_low"]) == ("책A", BOOK, 2, "2-3", 88.5, True)
    assert len(s["preview"]) == 120
    assert "secret" not in s                       # only the known book fields pass through


# ===== 2. Other sources are untouched =====
def test_other_sources_use_main_engine(monkeypatch, tmp_path):
    main = load_main(monkeypatch, tmp_path, "local")
    main._library = FakeLibrary()
    res = TestClient(main.app).post("/ask", json={"question": "hello there", "source": "saved"})
    assert res.json()["answer"] == "main answer" and main._library.calls == []
    assert "title" not in res.json()["sources"][0]


# ===== 3. Streaming works for the library too =====
def test_stream_library(monkeypatch, tmp_path):
    main = load_main(monkeypatch, tmp_path, "local")
    main._library = FakeLibrary()
    res = TestClient(main.app).post("/ask/stream", json={"question": "심방세동 치료", "source": "library"})
    events = ndjson(res)
    assert [e["type"] for e in events] == ["sources", "chunk", "chunk", "done"]
    assert events[0]["sources"][0]["title"] == "책A"
    assert "".join(e["text"] for e in events if e["type"] == "chunk") == "library answer"
    assert main.rag.calls == []


# ===== 4. Cloud: the library does not exist =====
def test_cloud_hides_library(monkeypatch, tmp_path):
    main = load_main(monkeypatch, tmp_path, "cloud")
    c = TestClient(main.app)
    assert c.post("/ask", json={"question": "심방세동 치료", "source": "library"}).status_code == 404
    assert c.post("/ask/stream", json={"question": "심방세동 치료", "source": "library"}).status_code == 404
    assert c.get(f"/library/pdf/{BOOK}").status_code == 404
    assert main._library is None and main.rag.calls == []
    # the normal cloud flow still works
    assert c.post("/ask", json={"question": "hello there"}).status_code == 200


# ===== 5. A library whose index is missing gives a clear 503, not a stack trace =====
def test_missing_index_is_503(monkeypatch, tmp_path):
    main = load_main(monkeypatch, tmp_path, "local")

    def boom(embedder=None):
        raise RuntimeError("Library keyword index not found. Run build_library_fts.py first.")

    monkeypatch.setattr(library, "LibraryRAG", boom)
    res = TestClient(main.app).post("/ask", json={"question": "심방세동 치료", "source": "library"})
    assert res.status_code == 503 and "keyword index not found" in res.json()["detail"]


# ===== 6. /library/pdf: real engine, real path rules =====
@pytest.fixture
def real_library(monkeypatch, tmp_path):
    lib_dir = tmp_path / "lib"
    (lib_dir / "index").mkdir(parents=True)
    ten = tmp_path / "TEN"
    (ten / "x").mkdir(parents=True)
    monkeypatch.setenv("JARVIS_LIBRARY_DIR", str(lib_dir))
    monkeypatch.setenv("LIBRARY_SRC", str(ten))
    lc = library.load_common()
    conn = lc.open_fts(lc.get_paths()["fts_db"])
    chunks = lc.chunk_pages(["심방세동 치료 " * 40])
    for c in chunks:
        c["id"] = f"{BOOK}-{c['n']}"
    lc.insert_doc(conn, {"doc_id": BOOK, "source": "library", "title": "책A", "folder": "f", "path": "x/책A.pdf",
                         "pages": 1}, chunks)
    conn.close()
    (lib_dir / "manifest.jsonl").write_text(json.dumps(
        {"id": BOOK, "path": "x/책A.pdf", "folder": "f", "title": "책A", "pages": 1, "status": "ok"}, ensure_ascii=False) + "\n",
        encoding="utf-8")
    (ten / "x" / "책A.pdf").write_bytes(b"%PDF-1.4 hello")
    return tmp_path


def test_pdf_route(monkeypatch, real_library):
    main = load_main(monkeypatch, real_library, "local")

    class NoEmb:
        def query(self, text):
            return [0.0]

    main._library = library.LibraryRAG(embedder=NoEmb())
    c = TestClient(main.app)
    ok = c.get(f"/library/pdf/{BOOK}")
    assert ok.status_code == 200 and ok.headers["content-type"] == "application/pdf"
    assert ok.headers["content-disposition"].startswith("inline") and ok.content.startswith(b"%PDF")
    for bad in ["..", "e" * 16, "A" * 16, "a" * 15, "%2e%2e", "a" * 16 + "x"]:
        assert c.get(f"/library/pdf/{bad}").status_code == 404, bad
    # An end-to-end search on the real engine still returns book fields through /ask
    res = c.post("/ask", json={"question": "심방세동 치료", "source": "library", "retrieve_only": True})
    assert res.status_code == 200 and res.json()["sources"][0]["title"] == "책A"
