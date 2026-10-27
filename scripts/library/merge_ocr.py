# ===== Step 3: merge the OCR text of scanned books into the library (quality gate, backups, safe to re-run) =====
# Usage (PowerShell, C:\dev\jarvis, venv active):
#   python scripts\library\merge_ocr.py                              # DRY RUN: only prints what would change
#   python scripts\library\merge_ocr.py --apply                      # really merge
#   python scripts\library\merge_ocr.py --apply --folders 07_Economics 06_Counterintelligence
# Then run build_library_fts.py and build_library_vectors.py (both resumable) to index the merged books.
# Input : <library>\text_ocr\<id>.txt + ocr_books.csv + ocr_pages.csv (written by the OCR script)
# Output: <library>\text\<id>.txt, manifest.jsonl (status "scanned" -> "ok", plus "ocr": true), ocr_quality.json
# Nothing is deleted: the old text file and the manifest are backed up first. Everything stays on this PC.

import argparse
import csv
import json
import os
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import library_common as lc


# ===== Quality gate (starting values from the eye review of the pilot books) =====
# A page at or above TRUSTED_CONF is used as is; between MIN_CONF and TRUSTED_CONF it is used but marked "low trust";
# below MIN_CONF, or with fewer than MIN_CHARS characters, the page is emptied (its page number is kept)
TRUSTED_CONF = 85.0
MIN_CONF = 60.0
MIN_CHARS = 20
# A book is merged only when at least this share of its OCR pages passed the gate
MIN_KEPT_RATIO = 0.3


def classify_page(conf, chars):
    if chars < MIN_CHARS or conf < MIN_CONF:
        return "drop"
    return "trusted" if conf >= TRUSTED_CONF else "low"


# ===== Planning one book (pure function: no files are touched) =====
def plan_book(row, page_rows, raw_text):
    """Return {"skip": reason} or {"text", "pages_kept", "pages_dropped", "low_pages", "mean_conf"}."""
    if row.get("sample") == "True":
        return {"skip": "pilot sample, not a full run"}
    total = int(row["pages_total"])
    if int(row["pages_done"]) != total:
        return {"skip": "OCR not finished for every page"}
    pages = raw_text.split("\f")
    if len(pages) != total:
        return {"skip": f"page count mismatch ({len(pages)} text pages vs {total})"}
    ocr = [r for r in page_rows if r["kind"] == "ocr"]
    if not ocr:
        return {"skip": "no OCR pages"}

    new, low_pages, confs, dropped = list(pages), [], [], 0
    for r in ocr:
        idx = int(r["idx"])
        conf = float(r["conf"] or 0)
        verdict = classify_page(conf, int(r["chars"] or 0))
        if verdict == "drop":
            # Keep the (empty) page so page numbers still match the PDF
            new[idx] = ""
            dropped += 1
            continue
        confs.append(conf)
        if verdict == "low":
            # 1-based page number, as used in chunks and in the PDF viewer
            low_pages.append(idx + 1)
    kept = len(ocr) - dropped
    if kept / len(ocr) < MIN_KEPT_RATIO:
        return {"skip": f"too few readable pages ({kept} of {len(ocr)})"}
    return {"text": "\f".join(new), "pages_kept": kept, "pages_dropped": dropped, "low_pages": low_pages,
            "mean_conf": round(sum(confs) / len(confs), 1)}


# ===== Small file helpers =====
def read_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_atomic(path, text):
    # Write next to the target, then replace: a power cut never leaves a half-written file behind
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8", newline="")
    os.replace(tmp, path)


# ===== Main =====
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="really write (default: dry run)")
    ap.add_argument("--folders", nargs="*", default=[], help="only these top folders")
    ap.add_argument("--ocr-dir", default=None, help="folder with the OCR output (default: <library>\\text_ocr)")
    args = ap.parse_args(argv)

    paths = lc.get_paths()
    lib = paths["library_dir"]
    ocr_dir = Path(args.ocr_dir) if args.ocr_dir else lib / "text_ocr"
    books = {r["id"]: r for r in read_csv(ocr_dir / "ocr_books.csv")} if (ocr_dir / "ocr_books.csv").exists() else {}
    pages_by_book = {}
    if (ocr_dir / "ocr_pages.csv").exists():
        for r in read_csv(ocr_dir / "ocr_pages.csv"):
            pages_by_book.setdefault(r["id"], []).append(r)

    manifest_lines = [json.loads(x) for x in paths["manifest"].read_text(encoding="utf-8").splitlines() if x.strip()]
    quality_path = lib / "ocr_quality.json"
    quality = json.loads(quality_path.read_text(encoding="utf-8")) if quality_path.exists() else {}
    backup_dir = lib / "text_backup_before_ocr"

    merged, unchanged, skipped, changed_manifest = [], [], {}, False
    stats = {"pages_kept": 0, "pages_dropped": 0, "pages_low": 0}
    for rec in manifest_lines:
        bid = rec["id"]
        if bid not in books:
            continue
        if args.folders and not any(lc.folder_matches(rec["folder"], f) for f in args.folders):
            continue
        if rec["status"] == "ok" and not rec.get("ocr"):
            skipped[bid] = "already a text book"
            continue
        raw_file = ocr_dir / f"{bid}.txt"
        if not raw_file.exists():
            skipped[bid] = "OCR text file not found"
            continue
        plan = plan_book(books[bid], pages_by_book.get(bid, []), raw_file.read_text(encoding="utf-8", errors="replace"))
        if "skip" in plan:
            skipped[bid] = plan["skip"]
            continue

        target = paths["text_dir"] / f"{bid}.txt"
        current = target.read_text(encoding="utf-8", errors="replace") if target.exists() else None
        already = rec["status"] == "ok" and rec.get("ocr") and current == plan["text"]
        if already:
            unchanged.append(bid)
            continue
        merged.append(bid)
        stats["pages_kept"] += plan["pages_kept"]
        stats["pages_dropped"] += plan["pages_dropped"]
        stats["pages_low"] += len(plan["low_pages"])
        if not args.apply:
            continue

        # 1) back up the old text once (never overwrite an existing backup), 2) write the new text atomically
        backup_dir.mkdir(exist_ok=True)
        if target.exists() and not (backup_dir / target.name).exists():
            shutil.copy2(target, backup_dir / target.name)
        write_atomic(target, plan["text"])
        rec.update({"status": "ok", "ocr": True, "ocr_conf": plan["mean_conf"], "chars": len(plan["text"].strip()),
                    "ocr_pages_kept": plan["pages_kept"], "ocr_pages_dropped": plan["pages_dropped"]})
        quality[bid] = {"conf": plan["mean_conf"], "low_pages": plan["low_pages"],
                        "kept_pages": plan["pages_kept"], "dropped_pages": plan["pages_dropped"]}
        changed_manifest = True

    # Manifest and sidecar are written ONCE at the end, after a timestamped backup of the manifest
    if args.apply and changed_manifest:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        shutil.copy2(paths["manifest"], paths["manifest"].with_name(f"manifest.jsonl.bak-{stamp}"))
        write_atomic(paths["manifest"], "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in manifest_lines))
        write_atomic(quality_path, json.dumps(quality, ensure_ascii=False, indent=1))

    mode = "APPLIED" if args.apply else "DRY RUN (nothing written; add --apply to merge)"
    print(f"{mode}: merge {len(merged)} books, already merged {len(unchanged)}, skipped {len(skipped)}")
    print(f"pages: kept {stats['pages_kept']} (low trust {stats['pages_low']}), emptied {stats['pages_dropped']}")
    reasons = {}
    for why in skipped.values():
        key = why.split(" (")[0]
        reasons[key] = reasons.get(key, 0) + 1
    for why, n in sorted(reasons.items(), key=lambda kv: -kv[1]):
        print(f"  skipped: {n} x {why}")
    if args.apply and merged:
        print("next: python scripts\\library\\build_library_fts.py ; python scripts\\library\\build_library_vectors.py")
    return {"merged": merged, "unchanged": unchanged, "skipped": skipped, "stats": stats}


if __name__ == "__main__":
    main()


# → OCR로 읽은 스캔본 책을 품질 기준(신뢰도 85 이상 사용, 60~85 낮은 신뢰 표시, 60 미만은 빈 쪽)으로 걸러서 도서 라이브러리에 합치겠다.
#   기본은 미리보기이고 --apply일 때만 쓰며, 옛 텍스트와 manifest는 먼저 백업하고 임시 파일에 쓴 뒤 교체한다.
#   이미 합친 책은 건드리지 않아서 여러 번 실행해도 결과가 같다.
