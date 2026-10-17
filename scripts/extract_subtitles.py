# ===== Extract .srt subtitles -> subtitles.csv (spoken text only) =====


import csv
import re
from pathlib import Path


DATA = Path(r"C:\instagram_export\data")
OUT = Path(r"C:\instagram_export\output\subtitles.csv")

# Lines like "00:00:01,000 --> 00:00:03,500"
TIME_LINE = re.compile(r"^\d{2}:\d{2}:\d{2}[,.]\d{3}\s*-->")


# srt_to_text(): keep only the spoken lines, drop numbers and time codes
def srt_to_text(raw):
    lines = []
    for line in raw.splitlines():
        line = line.strip()
        # Skip empty lines, pure numbers (cue index), and time code lines
        if not line or line.isdigit() or TIME_LINE.match(line):
            continue
        # Skip the same line repeated back to back (common in auto captions)
        if lines and lines[-1] == line:
            continue
        lines.append(line)
    return " ".join(lines)


rows = []
# rglob("*.srt"): find every .srt file in all sub-folders
for path in DATA.rglob("*.srt"):
    # utf-8-sig handles files that start with a BOM marker; errors="replace" avoids crashes
    raw = path.read_text(encoding="utf-8-sig", errors="replace")
    text = srt_to_text(raw)
    rows.append({
        # Path relative to data/, e.g. media/posts/202501/xxx.srt (links back to the video)
        "srt_path": path.relative_to(DATA).as_posix(),
        "chars": len(text),
        "text": text,
    })

with open(OUT, "w", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["srt_path", "chars", "text"])
    writer.writeheader()
    writer.writerows(rows)

non_empty = [r for r in rows if r["chars"] > 0]
total_chars = sum(r["chars"] for r in rows)
print(f"Subtitle files: {len(rows)}  (non-empty: {len(non_empty)})")
print(f"Total characters: {total_chars:,}  Avg per file: {total_chars // max(len(non_empty), 1):,}")
print(f"Written: {OUT}")


# → data 폴더 안의 .srt 자막 6,477개를 전부 찾아서 번호 줄과 시간 줄을 지우고
#   말한 내용만 이어 붙여 CSV로 저장하겠다. 각 자막이 어느 영상 것인지 경로도 같이 남긴다.
#   마지막에 파일 수, 빈 자막을 뺀 수, 전체 글자 수만 요약한다.