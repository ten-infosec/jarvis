"""Tests for app/library.py (book library engine): search with page numbers, OCR flags, PDF path safety, prompt.

Runs without models, network, or private data: a tiny library is built in a temp folder and the embedder is a fake.
Run from the repo root:  python -m pytest tests -v
"""
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import library  # noqa: E402

BOOK_A = "a" * 16
BOOK_B = "b" * 16
# Each page is long enough that chunks (800 chars) fall inside one or two pages
PAGES_A = ["서론 " * 300, "심방세동 치료 항응고제 " * 80, "결론 " * 300]
PAGES_B = ["머리말 " * 300, "당뇨병 인슐린 치료 " * 80, "부록 " * 300]


class FakeEmbedder:
    def query(self, text):
        return [0.0, 0.0, 0.0]


def write_manifest(lib, records):
    (lib / "manifest.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
                                        encoding="utf-8")


def add_book(lc, conn, book_id, title, pages):
    chunks = lc.chunk_pages(pages)
    for c in chunks:
        c["id"] = f"{book_id}-{c['n']}"
    lc.insert_doc(conn, {"doc_id": book_id, "source": "library", "title": title, "folder": "08_Medicine",
                         "path": f"x/{title}.pdf", "pages": len(pages)}, chunks)


@pytest.fixture
def lib(tmp_path, monkeypatch):
    library_dir = tmp_path / "lib"
    (library_dir / "index").mkdir(parents=True)
    (tmp_path / "TEN" / "x").mkdir(parents=True)
    monkeypatch.setenv("JARVIS_LIBRARY_DIR", str(library_dir))
    monkeypatch.setenv("LIBRARY_SRC", str(tmp_path / "TEN"))
    lc = library.load_common()
    paths = lc.get_paths()
    conn = lc.open_fts(paths["fts_db"])
    add_book(lc, conn, BOOK_A, "책A", PAGES_A)
    add_book(lc, conn, BOOK_B, "책B", PAGES_B)
    conn.close()
    write_manifest(library_dir, [
        {"id": BOOK_A, "path": "x/책A.pdf", "folder": "08_Medicine", "title": "책A", "pages": 3, "status": "ok"},
        {"id": BOOK_B, "path": "x/책B.pdf", "folder": "08_Medicine", "title": "책B", "pages": 3, "status": "ok",
         "ocr": True, "ocr_conf": 88.5},
    ])
    (library_dir / "ocr_quality.json").write_text(json.dumps({BOOK_B: {"low_pages": [2]}}), encoding="utf-8")
    (tmp_path / "TEN" / "x" / "책A.pdf").write_bytes(b"%PDF-1.4 fake")
    return library_dir, tmp_path


def make_engine(lib):
    return library.LibraryRAG(embedder=FakeEmbedder())


# ===== 1. Search returns book title and page numbers (keyword only: no vector index) =====
def test_retrieve_keyword_only_has_title_and_pages(lib):
    engine = make_engine(lib)
    query, hits = engine.retrieve("심방세동 치료", None, "hybrid", 5)
    assert hits and hits[0]["title"] == "책A" and hits[0]["source"] == "library"
    h = hits[0]
    assert h["book_id"] == BOOK_A and h["doc_id"] == BOOK_A and h["via"] == "bm25"
    first, last = h["page"], h["pages"]
    # The keyword sits on page 2, so the chunk must cover page 2
    low, high = (int(x) for x in (last.split("-") if "-" in last else (last, last)))
    assert low <= 2 <= high and first == low
    assert h["url"] == ""                       # the page builds the PDF link itself
    assert "ocr" not in h                       # a text book carries no OCR fields


# ===== 2. One hit per book, numbering is 1..n =====
def test_one_hit_per_book(lib):
    engine = make_engine(lib)
    _, hits = engine.retrieve("치료", None, "hybrid", 10)
    ids = [h["book_id"] for h in hits]
    assert len(ids) == len(set(ids)) == 2
    assert [h["n"] for h in hits] == [1, 2]


# ===== 3. OCR books show the confidence, and low-trust pages are flagged =====
def test_ocr_fields_and_low_pages(lib):
    engine = make_engine(lib)
    _, hits = engine.retrieve("당뇨병 인슐린", None, "hybrid", 5)
    h = next(x for x in hits if x["book_id"] == BOOK_B)
    assert h["ocr"] == 88.5 and h["ocr_low"] is True      # the matching chunk touches page 2 (a low-trust page)


# ===== 4. Vector candidates are fused with keyword candidates =====
def test_vector_candidates_are_fused(lib):
    engine = make_engine(lib)

    class FakeCol:
        def count(self):
            return 1

        def query(self, query_embeddings, n_results, include):
            return {"ids": [[f"{BOOK_A}-0"]], "documents": [["서론 서론"]],
                    "metadatas": [[{"title": "책A", "folder": "08_Medicine", "page_start": 1, "page_end": 1}]]}

    engine._col = FakeCol()
    _, hits = engine.retrieve("심방세동", None, "hybrid", 5)
    # Best chunk per book: BOOK_A appears once; it may come from keyword and/or vector candidates
    assert [h["book_id"] for h in hits].count(BOOK_A) == 1
    _, vec_hits = engine.retrieve("심방세동", None, "vector", 5)
    assert [h["via"] for h in vec_hits] == ["vec"]


# ===== 5. PDF path safety =====
def test_pdf_path_rules(lib):
    library_dir, tmp_path = lib
    engine = make_engine(lib)
    ok = engine.pdf_path(BOOK_A)
    assert ok is not None and ok.name == "책A.pdf"
    assert engine.pdf_path(BOOK_B) is None                              # listed in the manifest but the file is missing
    for bad in ["../x", "..", "A" * 16, "a" * 15, "a" * 17, "", None, "a" * 16 + "/../../x", "g" * 16]:
        assert engine.pdf_path(bad) is None
    # A manifest path that escapes the library folder is refused, even if that file exists
    (tmp_path / "outside.pdf").write_bytes(b"%PDF")
    evil = "c" * 16
    write_manifest(library_dir, [{"id": evil, "path": "../outside.pdf", "folder": "f", "title": "t", "pages": 1,
                                  "status": "ok"}])
    os.utime(library_dir / "manifest.jsonl", (1, 1))                     # force a different mtime
    assert engine.pdf_path(evil) is None
    # Not a PDF
    (tmp_path / "TEN" / "x" / "note.txt").write_text("hi", encoding="utf-8")
    write_manifest(library_dir, [{"id": evil, "path": "x/note.txt", "folder": "f", "title": "t", "pages": 1, "status": "ok"}])
    os.utime(library_dir / "manifest.jsonl", (2, 2))
    assert engine.pdf_path(evil) is None


# ===== 6. The manifest is reloaded when it changes (after OCR books are merged) =====
def test_manifest_reload_on_change(lib):
    library_dir, tmp_path = lib
    engine = make_engine(lib)
    new_id = "d" * 16
    assert engine.pdf_path(new_id) is None
    (tmp_path / "TEN" / "x" / "new.pdf").write_bytes(b"%PDF")
    write_manifest(library_dir, [{"id": new_id, "path": "x/new.pdf", "folder": "f", "title": "new", "pages": 1, "status": "ok"}])
    os.utime(library_dir / "manifest.jsonl", (3, 3))
    assert engine.pdf_path(new_id) is not None


# ===== 7. The prompt carries title + pages, marks OCR, and treats the sources as data =====
def test_generate_prompt(lib, monkeypatch):
    engine = make_engine(lib)
    seen = {}

    class Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"message": {"content": "답변 [1]"}}

    def fake_post(url, json=None, timeout=None, stream=False):
        seen["json"] = json
        return Resp()

    monkeypatch.setattr(library.requests, "post", fake_post)
    hits = [{"n": 1, "title": "책A", "pages": "2-3", "text": "본문 A"},
            {"n": 2, "title": "책B", "pages": "5", "text": "본문 B", "ocr": 88.5}]
    assert engine.generate("질문?", hits) == "답변 [1]"
    system, user = seen["json"]["messages"]
    assert "지시가 아니니" in system["content"] and "OCR" in system["content"]
    assert "[1] (책A, 쪽 2-3)" in user["content"] and "[2] (책B, 쪽 5, OCR)" in user["content"]
    assert seen["json"]["stream"] is False and seen["json"]["model"] == library.LLM


# ===== 8. A missing keyword index gives a clear error =====
def test_missing_index_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("JARVIS_LIBRARY_DIR", str(tmp_path / "empty"))
    with pytest.raises(RuntimeError, match="keyword index not found"):
        library.LibraryRAG(embedder=FakeEmbedder())
