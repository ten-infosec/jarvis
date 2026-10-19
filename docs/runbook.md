# Jarvis runbook (local operations)

Where to run: PowerShell, repo root `C:\dev\jarvis`, venv active (`.venv\Scripts\Activate.ps1`).
Private data stays on this PC: nothing in this file sends library or Instagram data anywhere.

## 1. Start and check
| Goal | Command | What it does |
|---|---|---|
| Tests (before any commit) | `python -m pytest tests -q` | 82 tests with fake engines, no keys, no network |
| Local server | `uvicorn app.main:app` | http://127.0.0.1:8000 (local mode; library search needs the library index) |
| Cloud-mode check | `$env:JARVIS_MODE="cloud"; uvicorn app.main:app` | library routes return 404 in cloud mode on purpose |

## 2. Library pipeline (books, local only)
Data lives outside the repo: `C:\jarvis_data\library` (text, manifest.jsonl, index\, text_ocr\).
1. Extract text and index: `python scripts\library\build_library_fts.py` (keyword index) then
   `python scripts\library\build_library_vectors.py` (vector index; resumable, safe to stop and restart).
2. OCR for scanned books runs in the background and writes `text_ocr\` (not part of the repo).
3. Merge OCR text into the library (the only step that changes `text\` and `manifest.jsonl`):
   - `python scripts\library\merge_ocr.py` -> DRY RUN, prints what would change, writes nothing.
   - `python scripts\library\merge_ocr.py --apply` -> really merges. Needs the owner's go-ahead.
   - Quality gate: page confidence >= 85 trusted, 60-85 used but marked low trust, < 60 or < 20 chars emptied;
     a book is merged only if >= 30% of its OCR pages pass. Old text and the manifest are backed up first
     (`text_backup_before_ocr\`, `manifest.jsonl.bak-<time>`). Running again changes nothing.
4. After a merge, rebuild both indexes (step 1) and run 10 test questions in the web UI; check that the
   page links open the right PDF page.

## 3. Safety checks
- Library PDF route accepts only 16-hex book ids and only files inside the library folder.
- Web UI keeps escapeHtml/safeUrl; do not insert model or PDF text with innerHTML.
- Do not copy library text to the clipboard, a cloud model, or the repo. Demos use fake data.

## 4. Cloud (public arXiv only)
- Release 0.3 is live; 0.4 (rate-limit hardening) is committed but NOT deployed.
- Deploy: `.\scripts\deploy.ps1 -Version 0.4 -DryRun` first, real run only after the owner approves.
- Index remaining papers after the Gemini quota resets: `python scripts\build_arxiv_index.py`.
- Incidents and lessons: docs\deploy.md.

## 5. When something breaks
| Symptom | First check |
|---|---|
| `/ask` with source library returns 503 | Library keyword index missing: rebuild with build_library_fts.py |
| Library returns 404 | Cloud mode is on (`JARVIS_MODE=cloud`); library is local only |
| Vector build stopped | Run build_library_vectors.py again; it resumes |
| OCR text looks wrong | Check `ocr_quality.json` low-trust pages; open the original PDF page link |
| Tests fail after pulling | `pip install -r requirements.txt -r requirements-dev.txt` |
