# ===== Count label names in saved_posts.json (names only, no values) =====


# sys: read the file path from the command line
import sys
# json: read the JSON file
import json
# Counter: a dict that counts how many times each item appears
from collections import Counter


# sys.argv[1] = the JSON path you pass after the script name
path = sys.argv[1]
# Open as UTF-8 and load the whole list of saved posts
with open(path, encoding="utf-8") as f:
    data = json.load(f)

# counts: label name -> how many posts have it
counts = Counter()
# Go through every saved post
for item in data:
    # .get("label_values", []): use an empty list if the key is missing
    for lv in item.get("label_values", []):
        # Count only the label NAME, never the value
        counts[lv.get("label", "(no label)")] += 1

# Print total posts, then each label with its count (most common first)
print(f"Posts: {len(data)}")
for label, n in counts.most_common():
    print(f"{n:6d}  {label}")


# → 저장 게시물 1,197개를 하나씩 돌면서, label_values 안의 "이름표"만 세어
#   어떤 이름표가 몇 번 나오는지 많은 순서로 보여 주겠다. 값(내용)은 출력하지 않는다.