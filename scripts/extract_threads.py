# ===== Extract threads_and_replies.json -> threads.csv =====


import json
import csv
from pathlib import Path
from datetime import datetime, timezone, timedelta


SRC = Path(r"C:\instagram_export\data\your_instagram_activity\threads\threads_and_replies.json")
OUT = Path(r"C:\instagram_export\output\threads.csv")
KST = timezone(timedelta(hours=9))


def fix(text):
    if not text:
        return ""
    try:
        return text.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text


with open(SRC, encoding="utf-8") as f:
    # The list sits under the key "text_post_app_text_posts"
    posts = json.load(f)["text_post_app_text_posts"]

rows = []
for p in posts:
    media = p.get("media", [])
    first = media[0] if media else {}
    ts = p.get("creation_timestamp") or first.get("creation_timestamp")
    app = first.get("text_app_post") or {}
    rows.append({
        "posted_at": datetime.fromtimestamp(ts, KST).strftime("%Y-%m-%d %H:%M") if ts else "",
        "is_reply": app.get("is_reply", False),
        "text": fix(p.get("title")) or fix(first.get("title")),
    })

with open(OUT, "w", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["posted_at", "is_reply", "text"])
    writer.writeheader()
    writer.writerows(rows)

with_text = sum(1 for r in rows if r["text"])
replies = sum(1 for r in rows if r["is_reply"])
dates = sorted(r["posted_at"] for r in rows if r["posted_at"])
print(f"Threads: {len(rows)}  (with text: {with_text}, replies: {replies})")
print(f"Date range: {dates[0]} ~ {dates[-1]}")
print(f"Written: {OUT}")


# → 스레드 글 3,504개마다 쓴 날짜·답글 여부·본문을 뽑아 CSV로 저장하고,
#   글이 있는 개수, 답글 개수, 기간만 요약해서 보여 주겠다.