# ===== Build corpus.jsonl: one unified document list from all CSVs =====


import csv
import json
from pathlib import Path
from collections import Counter, defaultdict


OUT_DIR = Path(r"C:\instagram_export\output")
OUT = OUT_DIR / "corpus.jsonl"


# read_csv(): load one CSV made by our extract scripts as a list of dicts
def read_csv(name):
    with open(OUT_DIR / name, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


# ===== Collections: url -> [collection names] =====
# defaultdict(list): a dict that creates an empty list for a new key automatically
collections_by_url = defaultdict(list)
for r in read_csv("collections.csv"):
    collections_by_url[r["url"]].append(r["collection"])

docs = []

# ===== Saved posts =====
for i, r in enumerate(read_csv("saved_posts.csv")):
    if not r["caption"].strip():
        continue
    parts = [r["caption"]]
    if r["hashtags"]:
        parts.append(f"해시태그: {r['hashtags']}")
    docs.append({
        "id": f"saved-{i:05d}", "source": "saved", "date": r["saved_at"],
        "text": "\n".join(parts),
        "meta": {"url": r["url"], "owner": r["owner_username"],
                 "collections": collections_by_url.get(r["url"], [])},
    })

# ===== My posts =====
for i, r in enumerate(read_csv("my_posts.csv")):
    if not r["caption"].strip():
        continue
    docs.append({
        "id": f"post-{i:05d}", "source": "my_post", "date": r["posted_at"],
        "text": r["caption"], "meta": {"media": r["first_media_uri"]},
    })

# ===== Threads =====
for i, r in enumerate(read_csv("threads.csv")):
    if not r["text"].strip():
        continue
    docs.append({
        "id": f"thread-{i:05d}", "source": "thread", "date": r["posted_at"],
        "text": r["text"], "meta": {"is_reply": r["is_reply"] == "True"},
    })

# ===== Subtitles =====
for i, r in enumerate(read_csv("subtitles.csv")):
    if not r["text"].strip():
        continue
    docs.append({
        "id": f"sub-{i:05d}", "source": "subtitle", "date": "",
        "text": r["text"], "meta": {"srt": r["srt_path"]},
    })

# ===== Remove exact duplicates (same source + same text) =====
seen = set()
unique = []
dupes = Counter()
for d in docs:
    key = (d["source"], d["text"].strip())
    if key in seen:
        dupes[d["source"]] += 1
        continue
    seen.add(key)
    unique.append(d)

# ensure_ascii=False: keep Korean as Korean (not \uXXXX codes)
with open(OUT, "w", encoding="utf-8") as f:
    for d in unique:
        f.write(json.dumps(d, ensure_ascii=False) + "\n")

# ===== Summary =====
count = Counter(d["source"] for d in unique)
chars = Counter()
for d in unique:
    chars[d["source"]] += len(d["text"])
print(f"{'source':10} {'docs':>8} {'dupes removed':>14} {'chars':>12}")
for s in ["saved", "my_post", "thread", "subtitle"]:
    print(f"{s:10} {count[s]:8,} {dupes[s]:14,} {chars[s]:12,}")
print(f"{'TOTAL':10} {len(unique):8,} {sum(dupes.values()):14,} {sum(chars.values()):12,}")
print(f"Written: {OUT}")


# → CSV 4개(+컬렉션)를 읽어서 출처가 달라도 id·source·date·text·meta 모양으로 통일하고,
#   같은 출처 안에서 글이 완전히 같은 문서는 하나만 남긴 뒤 corpus.jsonl로 저장하겠다.
#   마지막에 출처별 문서 수, 지운 중복 수, 글자 수를 표로 보여 준다.