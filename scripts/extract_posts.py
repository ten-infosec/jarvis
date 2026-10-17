# ===== Extract my posts (old format posts_1..4.json) -> my_posts.csv =====


import json
import csv
from pathlib import Path
from datetime import datetime, timezone, timedelta
from collections import Counter


# ===== Paths =====
SRC_DIR = Path(r"C:\instagram_export\data\your_instagram_activity\media")
OUT_DIR = Path(r"C:\instagram_export\output")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT = OUT_DIR / "my_posts.csv"
KST = timezone(timedelta(hours=9))


# ===== Helpers =====
def fix(text):
    if not text:
        return ""
    try:
        return text.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text


# to_kst(): seconds since 1970 -> "YYYY-MM-DD HH:MM" in Korean time
def to_kst(ts):
    return datetime.fromtimestamp(ts, KST).strftime("%Y-%m-%d %H:%M") if ts else ""


# parse_post(): one post may hold several photos/videos (a carousel)
def parse_post(post):
    media = post.get("media", [])
    first = media[0] if media else {}
    # Carousel posts keep caption/time on the post itself; single posts keep them on the media
    caption = fix(post.get("title")) or fix(first.get("title"))
    ts = post.get("creation_timestamp") or first.get("creation_timestamp")
    return {
        "posted_at": to_kst(ts),
        "caption": caption,
        "media_count": len(media),
        # Relative path inside the export, e.g. media/posts/202501/xxx.jpg (used later for OCR)
        "first_media_uri": first.get("uri", ""),
        "source_app": (first.get("cross_post_source") or {}).get("source_app", ""),
    }


# ===== Main =====
rows = []
# sorted(): posts_1, posts_2, posts_3, posts_4 in order
for path in sorted(SRC_DIR.glob("posts_*.json")):
    with open(path, encoding="utf-8") as f:
        rows.extend(parse_post(p) for p in json.load(f))

with open(OUT, "w", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

# Summary: counts only
with_caption = sum(1 for r in rows if r["caption"])
dates = sorted(r["posted_at"] for r in rows if r["posted_at"])
per_year = Counter(r["posted_at"][:4] for r in rows if r["posted_at"])
print(f"My posts: {len(rows)}  (with caption: {with_caption})")
print(f"Date range: {dates[0]} ~ {dates[-1]}")
print("Per year:", dict(sorted(per_year.items())))
print(f"Written: {OUT}")


# → posts_1~4.json의 게시물 3만 개를 하나씩 풀어서 올린 날짜·캡션·사진 개수·첫 사진 경로를
#   한 줄씩 뽑아 CSV로 저장하겠다. 위도·경도 같은 위치 정보는 넣지 않는다.
#   마지막에 개수, 기간, 연도별 게시물 수만 요약해 보여 준다.