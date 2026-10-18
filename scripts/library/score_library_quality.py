# ===== Text quality score per book (finds bad OCR layers to re-OCR later) =====
# Usage (PowerShell, C:\dev\jarvis, venv active):
#   python scripts\library\score_library_quality.py
# Writes C:\jarvis_data\library\quality.csv (stays on this PC) and prints only a summary.
# The score is a warning signal, not a verdict: books full of real Hanja (e.g. Korean medicine) can score low.

import csv
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import library_common as lc


# ===== Character classes =====
HANGUL = re.compile(r"[가-힣]")
LATIN = re.compile(r"[A-Za-z]")
HANJA = re.compile(r"[\u4e00-\u9fff]")
# Symbols that rarely appear in real text but often in broken OCR
ODD = re.compile(r"[^\s0-9A-Za-z가-힣\u4e00-\u9fff.,;:!?()\[\]{}'\"%/\-+=<>~·…–—°%]")


def score_text(text, sample_chars=200_000):
    # Score a sample from the middle of the book (front matter and indexes are unusual)
    if len(text) > sample_chars:
        mid = len(text) // 2
        text = text[mid - sample_chars // 2: mid + sample_chars // 2]
    n = max(len(text), 1)
    letters = len(HANGUL.findall(text)) + len(LATIN.findall(text)) + len(HANJA.findall(text))
    hanja_ratio = len(HANJA.findall(text)) / max(letters, 1)
    odd_ratio = len(ODD.findall(text)) / n
    # Korean prose has a space every ~3-6 characters; long runs without spaces = lost spacing
    korean_runs = re.findall(r"[가-힣]{12,}", text)
    nospace_ratio = sum(len(r) for r in korean_runs) / max(len(HANGUL.findall(text)), 1)
    # 100 = clean; subtract penalties (weights are rough and documented, not tuned)
    score = 100 - 150 * odd_ratio - 60 * hanja_ratio - 40 * nospace_ratio
    return round(max(score, 0), 1), round(hanja_ratio, 3), round(odd_ratio, 3), round(nospace_ratio, 3)


paths = lc.get_paths()
out_path = paths["library_dir"] / "quality.csv"
rows = []
for m in lc.iter_manifest(paths["manifest"]):
    if m["status"] != "ok":
        rows.append([m["id"], m["folder"], m["title"], m["status"], "", "", "", ""])
        continue
    text = lc.read_text(paths["text_dir"] / f"{m['id']}.txt")
    rows.append([m["id"], m["folder"], m["title"], "ok", *score_text(text)])

with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f)
    w.writerow(["id", "folder", "title", "status", "score", "hanja_ratio", "odd_ratio", "nospace_ratio"])
    w.writerows(rows)

# ===== Summary only (no book content printed) =====
scored = [r for r in rows if r[3] == "ok"]
bands = {"good (>=80)": 0, "check (50-79)": 0, "bad (<50)": 0}
for r in scored:
    s = r[4]
    bands["good (>=80)" if s >= 80 else "check (50-79)" if s >= 50 else "bad (<50)"] += 1
print(f"scored {len(scored)} text books, {len(rows) - len(scored)} scanned/failed")
for k, v in bands.items():
    print(f"  {k}: {v}")
print(f"details -> {out_path}")
