# ===== Peek inside the "dict" entries: titles and inner labels only =====


import sys
import json


# fix(): undo Instagram's Latin-1 / UTF-8 mix-up
def fix(text):
    if not text:
        return text
    try:
        return text.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text


path = sys.argv[1]
index = int(sys.argv[2]) if len(sys.argv) > 2 else 0

with open(path, encoding="utf-8") as f:
    data = json.load(f)

post = data[index]
for i, lv in enumerate(post.get("label_values", [])):
    # Only look at entries that hold a nested "dict"
    if "dict" not in lv:
        continue
    # title = the group name of this nested block
    print(f"[{i}] title={fix(lv.get('title'))!r}  dict_type={type(lv['dict']).__name__}")
    inner = lv["dict"]
    # The nested block can be a list of small dicts or a single dict
    rows = inner if isinstance(inner, list) else [inner]
    for row in rows:
        # A row may itself hold another "dict" one level deeper
        if isinstance(row, dict) and "dict" in row:
            for sub in row["dict"]:
                print(f"      label={fix(sub.get('label'))!r:16} keys={sorted(sub.keys())}  value_len={len(fix(sub.get('value')) or '')}")
        elif isinstance(row, dict):
            print(f"    label={fix(row.get('label'))!r:16} keys={sorted(row.keys())}  value_len={len(fix(row.get('value')) or '')}")


# → 저장 게시물 하나에서 "dict"를 가진 항목만 골라, 묶음 이름(title)과
#   그 안의 이름표(label), 키 이름, 값의 글자 수만 보여 주겠다. 실제 값은 출력하지 않는다.