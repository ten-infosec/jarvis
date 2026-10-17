# ===== Extract saved_collections.json -> collections.csv (collection name -> post URL) =====


import json
import csv
from pathlib import Path
from collections import Counter


SRC = Path(r"C:\instagram_export\data\your_instagram_activity\saved\saved_collections.json")
SAVED_CSV = Path(r"C:\instagram_export\output\saved_posts.csv")
OUT = Path(r"C:\instagram_export\output\collections.csv")


def fix(text):
    if not text:
        return ""
    try:
        return text.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text


with open(SRC, encoding="utf-8") as f:
    data = json.load(f)

rows = []
for col in data:
    name = ""
    for lv in col.get("label_values", []):
        # Collection name lives in a plain label/value pair
        if fix(lv.get("label")) == "이름":
            name = fix(lv.get("value"))
        # Posts in the collection live in the "미디어" group
        elif fix(lv.get("title")) == "미디어":
            items = lv["dict"] if isinstance(lv["dict"], list) else [lv["dict"]]
            for item in items:
                subs = item.get("dict", []) if isinstance(item, dict) else []
                for sub in subs:
                    if fix(sub.get("label")) == "URL":
                        url = sub.get("href") or fix(sub.get("value"))
                        if url:
                            rows.append({"collection": name, "url": url})

with open(OUT, "w", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["collection", "url"])
    writer.writeheader()
    writer.writerows(rows)

# How many collection links also appear in saved_posts.csv?
with open(SAVED_CSV, encoding="utf-8-sig") as f:
    saved_urls = {r["url"] for r in csv.DictReader(f)}
matched = sum(1 for r in rows if r["url"] in saved_urls)

print(f"Collections: {len(data)}  Links: {len(rows)}  Matched to saved posts: {matched}")
print("--- biggest collections ---")
for name, n in Counter(r["collection"] for r in rows).most_common(15):
    print(f"{n:5d}  {name}")
print(f"Written: {OUT}")


# → 컬렉션 60개마다 이름과 그 안에 든 게시물 링크를 뽑아 "컬렉션 - URL" 표로 저장하겠다.
#   그리고 그 링크가 saved_posts.csv에 몇 개나 있는지 맞춰 보고, 큰 컬렉션 15개의 이름과 개수를 보여 준다.