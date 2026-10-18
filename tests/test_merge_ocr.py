"""Tests for scripts/library/merge_ocr.py: quality gate, page numbers, backups, dry run, idempotence.

Uses a tiny temp library with synthetic text only (no private books). Run from the repo root: python -m pytest tests -v
"""
import csv
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "library"))
import merge_ocr as mo  # noqa: E402
import library_common as lc  # noqa: E402

A, B, C, D, E = ("a" * 16, "b" * 16, "c" * 16, "d" * 16, "e" * 16)
BOOK_FIELDS = ["id", "folder", "langs", "pages_total", "pages_done", "ocr_pages", "layer_pages", "blank_pages",
               "mean_conf", "low_ratio", "sec_per_ocr_page", "wall_min", "verdict", "sample"]
PAGE_FIELDS = ["id", "idx", "kind", "conf", "words", "sec", "chars", "hangul_ratio", "odd_ratio"]


def book_row(bid, total, done=None, sample="False"):
    return {"id": bid, "folder": "07_Economics", "langs": "kor+eng", "pages_total": total,
            "pages_done": total if done is None else done, "ocr_pages": total, "layer_pages": 0, "blank_pages": 0,
            "mean_conf": 90, "low_ratio": 0, "sec_per_ocr_page": 3, "wall_min": 1, "verdict": "pass", "sample": sample}


def page_row(bid, idx, conf, chars, kind="ocr"):
    return {"id": bid, "idx": idx, "kind": kind, "conf": conf, "words": 5, "sec": 3, "chars": chars,
            "hangul_ratio": 0.9, "odd_ratio": 0}


def write_csv(path, fields, rows):
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


@pytest.fixture
def library(tmp_path, monkeypatch):
    lib = tmp_path / "lib"
    (lib / "text").mkdir(parents=True)
    ocr = lib / "text_ocr"
    ocr.mkdir()
    monkeypatch.setenv("JARVIS_LIBRARY_DIR", str(lib))

    manifest = [
        {"id": A, "path": "07/a.pdf", "folder": "07_Economics", "title": "A", "pages": 3, "chars": 0, "status": "scanned"},
        {"id": B, "path": "07/b.pdf", "folder": "07_Economics", "title": "B", "pages": 4, "chars": 0, "status": "scanned"},
        {"id": C, "path": "07/c.pdf", "folder": "07_Economics", "title": "C", "pages": 5, "chars": 0, "status": "scanned"},
        {"id": D, "path": "01/d.pdf", "folder": "01_Computer & IT", "title": "D", "pages": 2, "chars": 99, "status": "ok"},
        {"id": E, "path": "07/e.pdf", "folder": "07_Economics", "title": "E", "pages": 3, "chars": 0, "status": "scanned"},
    ]
    (lib / "manifest.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in manifest), encoding="utf-8")
    for bid in (A, B, C, D, E):
        (lib / "text" / f"{bid}.txt").write_text("\f" * (3 if bid == A else 1), encoding="utf-8")   # old, empty scan text

    # A: three good pages. B: a trusted page, a low-trust page, a low-confidence page, a too-short page.
    # C: only 1 of 5 pages readable (20%). E: text file has the wrong number of pages.
    # D: a TEXT book that wrongly appears in the OCR list: the guard must leave it alone.
    (ocr / f"{A}.txt").write_text("\f".join(["A 첫 쪽", "A 둘째 쪽", "A 셋째 쪽"]), encoding="utf-8")
    (ocr / f"{B}.txt").write_text("\f".join(["B 믿을 만한 쪽 " * 5, "B 낮은 신뢰 쪽 " * 5, "B 못 읽은 쪽 " * 5, "짧음"]), encoding="utf-8")
    (ocr / f"{C}.txt").write_text("\f".join(["C 하나 " * 10, "x", "x", "x", "x"]), encoding="utf-8")
    (ocr / f"{E}.txt").write_text("\f".join(["E 쪽 " * 10, "E 쪽 " * 10]), encoding="utf-8")
    write_csv(ocr / "ocr_books.csv", BOOK_FIELDS,
              [book_row(A, 3), book_row(B, 4), book_row(C, 5), book_row(D, 2), book_row(E, 3)])
    write_csv(ocr / "ocr_pages.csv", PAGE_FIELDS,
              [page_row(A, i, 92, 50) for i in range(3)]
              + [page_row(B, 0, 90, 80), page_row(B, 1, 70, 80), page_row(B, 2, 40, 80), page_row(B, 3, 95, 5)]
              + [page_row(C, 0, 90, 80)] + [page_row(C, i, 30, 80) for i in range(1, 5)]
              + [page_row(E, i, 90, 50) for i in range(3)])
    return lib


def snapshot(lib):
    # Every file under the library folder with its bytes, to prove a dry run changes nothing
    return {p.relative_to(lib).as_posix(): p.read_bytes() for p in lib.rglob("*") if p.is_file()}


def manifest(lib):
    return {r["id"]: r for r in (json.loads(x) for x in (lib / "manifest.jsonl").read_text(encoding="utf-8").splitlines())}


# ===== 1. The gate, at the exact boundaries =====
@pytest.mark.parametrize("conf, chars, expected", [
    (85.0, 20, "trusted"), (84.9, 20, "low"), (60.0, 20, "low"), (59.9, 20, "drop"),
    (95.0, 19, "drop"), (95.0, 0, "drop"), (0.0, 500, "drop"),
])
def test_classify_page_boundaries(conf, chars, expected):
    assert mo.classify_page(conf, chars) == expected


# ===== 2. A dry run writes nothing at all =====
def test_dry_run_changes_nothing(library, capsys):
    before = snapshot(library)
    result = mo.main([])
    assert snapshot(library) == before
    assert sorted(result["merged"]) == sorted([A, B])            # what WOULD be merged
    assert "DRY RUN" in capsys.readouterr().out


# ===== 3. Apply: gate, page numbers, manifest, sidecar =====
def test_apply_merges_with_the_gate(library):
    result = mo.main(["--apply"])
    assert sorted(result["merged"]) == sorted([A, B])
    m = manifest(library)
    assert m[A]["status"] == "ok" and m[A]["ocr"] is True and m[A]["ocr_conf"] == 92.0
    assert m[B]["status"] == "ok" and m[B]["ocr_pages_dropped"] == 2 and m[B]["ocr_pages_kept"] == 2
    # skipped books stay scanned; the text book is untouched
    assert m[C]["status"] == "scanned" and m[E]["status"] == "scanned"
    assert m[D]["status"] == "ok" and "ocr" not in m[D]
    assert "too few readable pages" in result["skipped"][C] and "page count mismatch" in result["skipped"][E]
    assert result["skipped"][D] == "already a text book"

    # Page numbers are kept: emptied pages stay as empty entries, so the book still has 4 pages
    pages = (library / "text" / f"{B}.txt").read_text(encoding="utf-8").split("\f")
    assert len(pages) == 4 and pages[0].startswith("B 믿을") and pages[1].startswith("B 낮은")
    assert pages[2] == "" and pages[3] == ""
    q = json.loads((library / "ocr_quality.json").read_text(encoding="utf-8"))
    assert q[B]["low_pages"] == [2] and q[B]["conf"] == 80.0       # 1-based page 2; mean of 90 and 70
    assert q[A]["low_pages"] == []


# ===== 4. Backups exist and keep the old content =====
def test_backups(library):
    old_manifest = (library / "manifest.jsonl").read_bytes()
    mo.main(["--apply"])
    assert (library / "text_backup_before_ocr" / f"{A}.txt").read_bytes() == b"\f\f\f"
    baks = list(library.glob("manifest.jsonl.bak-*"))
    assert len(baks) == 1 and baks[0].read_bytes() == old_manifest
    assert not list(library.rglob("*.tmp"))                          # atomic writes left no temp files


# ===== 5. Running again changes nothing (no new backup, same bytes) =====
def test_idempotent(library):
    mo.main(["--apply"])
    after_first = snapshot(library)
    second = mo.main(["--apply"])
    assert second["merged"] == [] and sorted(second["unchanged"]) == sorted([A, B])
    assert snapshot(library) == after_first


# ===== 6. A pilot sample or an unfinished OCR run is never merged =====
def test_sample_and_unfinished_are_skipped(library):
    write_csv(library / "text_ocr" / "ocr_books.csv", BOOK_FIELDS,
              [book_row(A, 3, sample="True"), book_row(B, 4, done=2)])
    result = mo.main(["--apply"])
    assert result["merged"] == []
    assert "pilot sample" in result["skipped"][A] and "not finished" in result["skipped"][B]
    assert manifest(library)[A]["status"] == "scanned"


# ===== 7. Folder filter =====
def test_folder_filter(library):
    assert mo.main(["--folders", "09_Math"])["merged"] == []
    assert sorted(mo.main(["--folders", "07_Economics"])["merged"]) == sorted([A, B])


# ===== 8. The merged text feeds the existing indexer: page numbers in chunks match the PDF =====
def test_merged_text_works_with_the_indexer(library):
    mo.main(["--apply"])
    raw = (library / "text" / f"{B}.txt").read_text(encoding="utf-8")
    chunks = lc.chunk_pages(lc.split_pages(raw))
    assert chunks and chunks[0]["page_start"] == 1 and max(c["page_end"] for c in chunks) == 2   # pages 3-4 are empty
