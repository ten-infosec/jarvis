# ===== Extract saved_posts.json -> saved_posts.csv =====


# json: read the export file
import json
# csv: write a spreadsheet-friendly table
import csv
# Path: build file paths safely on Windows
from pathlib import Path
# datetime/timezone/timedelta: turn timestamps into Korean time
from datetime import datetime, timezone, timedelta


# ===== Paths =====
# Input: the saved posts file we already extracted
SRC = Path(r"C:\instagram_export\data\your_instagram_activity\saved\saved_posts.json")
# Output folder: next to the data, never inside the code repo
OUT_DIR = Path(r"C:\instagram_export\output")
# exist_ok=True: no error if the folder already exists
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT = OUT_DIR / "saved_posts.csv"

# KST = UTC + 9 hours
KST = timezone(timedelta(hours=9))


# ===== Helpers =====
# fix(): undo Instagram's Latin-1 / UTF-8 mix-up so Korean reads correctly
def fix(text):
    if not text:
        return ""
    try:
        return text.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text


# rows_of(): nested "dict" can be a list or a single dict -> always a list
def rows_of(block):
    return block if isinstance(block, list) else [block]


# parse_post(): turn one saved post into one flat row (a Python dict)
def parse_post(post):
    row = {"saved_at": "", "url": "", "owner_username": "", "owner_name": "",
           "hashtags": "", "caption": ""}
    # timestamp is seconds since 1970 (UTC) -> readable Korean date-time
    ts = post.get("timestamp")
    if ts:
        row["saved_at"] = datetime.fromtimestamp(ts, KST).strftime("%Y-%m-%d %H:%M")

    captions = []
    tags = []
    for lv in post.get("label_values", []):
        label = fix(lv.get("label"))
        title = fix(lv.get("title"))
        # Simple label/value pairs
        if label == "URL" and not row["url"]:
            row["url"] = lv.get("href") or fix(lv.get("value"))
        elif label == "캡션":
            text = fix(lv.get("value"))
            # The same caption appears twice -> keep only one copy
            if text and text not in captions:
                captions.append(text)
        # Nested groups: hashtags and owner
        elif title == "해시태그":
            for r in rows_of(lv.get("dict", [])):
                for sub in r.get("dict", []) if isinstance(r, dict) else []:
                    if fix(sub.get("label")) == "이름" and sub.get("value"):
                        tags.append(fix(sub["value"]))
        elif title == "소유자":
            for r in rows_of(lv.get("dict", [])):
                for sub in r.get("dict", []) if isinstance(r, dict) else []:
                    sub_label = fix(sub.get("label"))
                    if sub_label == "사용자 이름":
                        row["owner_username"] = fix(sub.get("value"))
                    elif sub_label == "이름":
                        row["owner_name"] = fix(sub.get("value"))

    row["caption"] = "\n".join(captions)
    row["hashtags"] = ", ".join(tags)
    return row


# ===== Main =====
with open(SRC, encoding="utf-8") as f:
    data = json.load(f)

rows = [parse_post(p) for p in data]

# utf-8-sig: adds a marker so Excel opens Korean text correctly
with open(OUT, "w", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

# Summary: counts only, no content
with_caption = sum(1 for r in rows if r["caption"])
with_owner = sum(1 for r in rows if r["owner_username"])
dates = sorted(r["saved_at"] for r in rows if r["saved_at"])
print(f"Saved posts: {len(rows)}")
print(f"With caption: {with_caption} / With owner: {with_owner}")
print(f"Date range: {dates[0]} ~ {dates[-1]}")
print(f"Written: {OUT}")


# → 저장 게시물 1,197개를 하나씩 풀어서 날짜·링크·작성자·해시태그·캡션을 한 줄씩 뽑고,
#   깨진 한글을 복원하고 중복 캡션을 하나로 줄여서 엑셀용 CSV로 저장하겠다.
#   마지막에는 내용이 아니라 개수와 기간만 요약해서 보여 준다.