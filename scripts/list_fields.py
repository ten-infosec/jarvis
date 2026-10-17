# ===== List label / title names in an Instagram export file (names only) =====
# Usage: python scripts/list_fields.py <path-to-json>


import sys
import json
from collections import Counter


# fix(): undo Instagram's Latin-1 / UTF-8 mix-up
def fix(text):
    if not text:
        return text
    try:
        return text.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text


path = sys.argv[1]
with open(path, encoding="utf-8") as f:
    data = json.load(f)

# Some files wrap the list inside a dict -> take the first list we find
if isinstance(data, dict):
    data = next(v for v in data.values() if isinstance(v, list))

# top: label/title names directly under label_values
# inner: label names one level deeper (inside "dict" groups)
top = Counter()
inner = Counter()
for item in data:
    for lv in item.get("label_values", []):
        if "dict" in lv:
            title = fix(lv.get("title")) or "(no title)"
            top[f"[group] {title}"] += 1
            rows = lv["dict"] if isinstance(lv["dict"], list) else [lv["dict"]]
            for r in rows:
                subs = r.get("dict", []) if isinstance(r, dict) else []
                for sub in subs:
                    inner[f"{title} > {fix(sub.get('label'))}"] += 1
        else:
            top[fix(lv.get("label")) or "(no label)"] += 1

print(f"Items: {len(data)}")
print("--- top-level labels ---")
for name, n in top.most_common():
    print(f"{n:7d}  {name}")
print("--- inside groups ---")
for name, n in inner.most_common(40):
    print(f"{n:7d}  {name}")


# → 파일 안의 항목마다 이름표(label)와 묶음 이름(title), 묶음 속 이름표를 세어
#   어떤 필드가 몇 번 나오는지 많은 순서로 보여 주겠다. 값(내용)은 출력하지 않는다.