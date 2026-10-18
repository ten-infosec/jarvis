# ===== Personal library (TEN) PDF -> text, runs on the local PC only =====
# Usage (PowerShell, C:\dev\jarvis, venv active):
#   pip install pymupdf
#   python scripts\extract_library.py
# Resumable: PDFs already listed in manifest.jsonl are skipped, so it can be stopped (Ctrl+C) and re-run.
# Output stays on this PC (C:\jarvis_data\library). Never commit or upload it.

import hashlib
import json
import os
import sys
import time
from pathlib import Path

# PyMuPDF: fast PDF text extraction (imported as "fitz")
import fitz


# ===== Settings =====
# Source library folder (read only; nothing in it is changed)
SRC = Path(os.getenv("LIBRARY_SRC", str(Path.home() / "Desktop" / "TEN")))
# Output folder: outside the repo and outside OneDrive
OUT = Path(os.getenv("LIBRARY_OUT", r"C:\jarvis_data\library"))
TEXT_DIR = OUT / "text"
MANIFEST = OUT / "manifest.jsonl"


# ===== Already done (resume) =====
TEXT_DIR.mkdir(parents=True, exist_ok=True)
done = set()
if MANIFEST.exists():
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        if line.strip():
            done.add(json.loads(line)["path"])


# ===== Find PDFs =====
# Relative paths use "/" so ids match the earlier run made from Linux
pdfs = sorted(p.relative_to(SRC).as_posix() for p in SRC.rglob("*") if p.suffix.lower() == ".pdf")
todo = [r for r in pdfs if r not in done]
print(f"total {len(pdfs)}, done {len(done)}, to do {len(todo)}")


# ===== Extract =====
start = time.time()
with MANIFEST.open("a", encoding="utf-8") as man:
    for i, rel in enumerate(todo, 1):
        # Stable id from the relative path (same rule as the first batch)
        doc_id = hashlib.sha1(rel.encode("utf-8")).hexdigest()[:16]
        status = "ok"
        try:
            with fitz.open(SRC / rel) as doc:
                pages = doc.page_count
                # "\f" (form feed) separates pages, like pdftotext
                text = "\f".join(page.get_text() for page in doc)
        except Exception as e:
            pages, text, status = 0, "", f"error:{type(e).__name__}"

        chars = len(text.strip())
        # Very little text per page -> image-only (scanned) PDF, needs OCR later
        if status == "ok" and chars < 20 * max(pages, 1):
            status = "scanned"

        (TEXT_DIR / f"{doc_id}.txt").write_text(text, encoding="utf-8")
        man.write(json.dumps({
            "id": doc_id, "path": rel, "folder": rel.split("/")[0],
            "title": Path(rel).stem, "pages": pages, "chars": chars, "status": status,
        }, ensure_ascii=False) + "\n")
        man.flush()

        # Progress line every 10 books
        if i % 10 == 0 or i == len(todo):
            mins = (time.time() - start) / 60
            print(f"{i}/{len(todo)}  {mins:.1f} min")

print("finished")
