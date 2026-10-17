# ===== Peek at one saved post: label, value length, link domain only =====


# sys: read command-line arguments
import sys
# json: read the JSON file
import json
# urlparse: split a URL into parts so we can show only the domain
from urllib.parse import urlparse


# fix(): undo Instagram's Latin-1 / UTF-8 mix-up so Korean shows correctly
def fix(text):
    # Empty or missing text: return as is
    if not text:
        return text
    try:
        # Turn the broken characters back into raw bytes, then read them as UTF-8
        return text.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        # Text that was never broken: keep it unchanged
        return text


# sys.argv[1] = JSON path, sys.argv[2] = which post to look at (default 0 = first)
path = sys.argv[1]
index = int(sys.argv[2]) if len(sys.argv) > 2 else 0

# Load all saved posts
with open(path, encoding="utf-8") as f:
    data = json.load(f)

# Pick one post by its position in the list
post = data[index]
print(f"Post #{index}  timestamp={post.get('timestamp')}")
# enumerate(): gives (position, item) pairs: 0, 1, 2 ...
for i, lv in enumerate(post.get("label_values", [])):
    label = fix(lv.get("label")) or "(no label)"
    value = fix(lv.get("value")) or ""
    href = lv.get("href") or ""
    # netloc = the domain part of a URL, e.g. www.instagram.com
    domain = urlparse(href).netloc if href else ""
    # Show the label, how long the value is, and the link's domain only
    print(f"  [{i}] label={label!r:12}  value_len={len(value):5d}  href_domain={domain}")


# → 저장 게시물 하나를 골라서, 8개 항목마다 이름표(한글 복원), 값의 글자 수,
#   링크의 도메인만 보여 주겠다. 캡션 내용이나 계정 이름은 출력하지 않는다.