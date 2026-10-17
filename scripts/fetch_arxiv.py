# ===== Fetch medical-AI paper metadata (title + abstract) from the arXiv API =====
# Usage: python scripts/fetch_arxiv.py --max 300
# Output: data/arxiv/papers.jsonl (same format as corpus.jsonl, so the same RAG pipeline can use it)


import argparse
import json
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import requests


# ===== Settings =====
API_URL = "https://export.arxiv.org/api/query"
# Project root = parent of the scripts folder, so the script works from any current directory
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "arxiv" / "papers.jsonl"
# XML namespaces used by arXiv's Atom feed (needed to find tags like <entry>, <title>)
NS = {"a": "http://www.w3.org/2005/Atom"}
# Imaging / vision / ML / NLP papers whose abstract mentions a medical keyword
QUERY = ("(cat:eess.IV OR cat:cs.CV OR cat:cs.LG OR cat:cs.CL) AND "
         "(abs:medical OR abs:clinical OR abs:radiology OR abs:healthcare)")
# Papers per request, and the wait between requests that arXiv asks for
PAGE = 100
SLEEP_SEC = 3

parser = argparse.ArgumentParser()
# --max: how many search results to go through in total
parser.add_argument("--max", type=int, default=300)
args = parser.parse_args()


def clean(text):
    # Collapse line breaks and repeated spaces into single spaces
    return " ".join((text or "").split())


# ===== Resume: skip papers already saved =====
OUT.parent.mkdir(parents=True, exist_ok=True)
seen = set()
if OUT.exists():
    with open(OUT, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                seen.add(json.loads(line)["id"])
print(f"Already saved: {len(seen)}")

# ===== Fetch page by page =====
added = 0
with open(OUT, "a", encoding="utf-8") as f:
    for start in range(0, args.max, PAGE):
        r = requests.get(
            API_URL,
            params={
                "search_query": QUERY,
                "start": start,
                "max_results": PAGE,
                # Newest papers first
                "sortBy": "submittedDate",
                "sortOrder": "descending",
            },
            # Identify the client politely (good manners for public APIs)
            headers={"User-Agent": "jarvis-study/0.1 (personal learning project)"},
            timeout=60,
        )
        r.raise_for_status()
        root = ET.fromstring(r.content)
        entries = root.findall("a:entry", NS)
        if not entries:
            print("No more results.")
            break

        for e in entries:
            # "http://arxiv.org/abs/2410.12345v2" -> "2410.12345" (drop the version suffix)
            raw_id = e.findtext("a:id", default="", namespaces=NS).rsplit("/abs/", 1)[-1]
            arxiv_id = re.sub(r"v\d+$", "", raw_id)
            doc_id = f"arxiv:{arxiv_id}"
            if doc_id in seen:
                continue

            title = clean(e.findtext("a:title", default="", namespaces=NS))
            abstract = clean(e.findtext("a:summary", default="", namespaces=NS))
            published = e.findtext("a:published", default="", namespaces=NS)[:10]
            authors = [clean(a.findtext("a:name", default="", namespaces=NS))
                       for a in e.findall("a:author", NS)]
            categories = [c.get("term") for c in e.findall("a:category", NS)]

            # Same shape as corpus.jsonl: id / source / date / text / meta
            doc = {
                "id": doc_id,
                "source": "arxiv",
                "date": published,
                "text": f"{title}\n\n{abstract}",
                "meta": {
                    "url": f"https://arxiv.org/abs/{arxiv_id}",
                    "title": title,
                    "authors": authors[:10],
                    "categories": categories,
                },
            }
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
            seen.add(doc_id)
            added += 1

        print(f"  start={start:4d}  got {len(entries):3d}  new so far {added}")
        # Respect arXiv's request interval
        time.sleep(SLEEP_SEC)

print(f"Done. New papers: {added}  File total: {len(seen)}  -> {OUT}")


# → arXiv API에 "의료 키워드가 초록에 들어간 영상·비전·머신러닝·자연어 논문"을 최신순으로 100개씩 요청하고,
#   요청 사이에 3초씩 쉬면서 제목, 초록, 저자, 분야, 링크를 corpus.jsonl과 같은 형식으로 저장하겠다.
#   이미 저장된 논문은 건너뛰기 때문에 여러 번 실행해도 중복되지 않는다.