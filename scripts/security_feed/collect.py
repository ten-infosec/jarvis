"""Collect public AI-security news (arXiv + RSS/Atom) into a news.json for morning_paper.py.

Usage:  python scripts/security_feed/collect.py [--out news.json] [--sources sources.json]
Output: [{"title", "summary", "url", "section", "source", "date"}]
Only public data is fetched. Fetched text is treated as data: it is never executed or followed.
"""
import argparse
import json
import re
import sys
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
SEEN = HERE.parent.parent / "data" / "security" / "seen.json"  # data/ is gitignored
UA = "jarvis-security-feed/0.1 (personal research)"
TIMEOUT = 20
MAX_BYTES = 2_000_000  # cap response size so one bad feed cannot eat memory
SUMMARY_LEN = 300
ARXIV_API = "https://export.arxiv.org/api/query"
OSV_API = "https://api.osv.dev/v1/query"
NS ={"a": "http://www.w3.org/2005/Atom"}


# ===== Fetch =====
def allowed_hosts(cfg):
    urls = [s["url"] for s in cfg.get("rss", [])] + [ARXIV_API, OSV_API]
    return {urllib.parse.urlparse(u).hostname for u in urls}


def fetch(url, hosts, body=None):
    parsed = urllib.parse.urlparse(url)
    # allowlist: only hosts named in sources.json (or the arXiv/OSV APIs) are requested
    if parsed.scheme != "https" or parsed.hostname not in hosts:
        raise ValueError(f"host not allowed: {url}")
    headers = {"User-Agent": UA}
    if body is not None:
        headers["Content-Type"] = "application/json"
        body = json.dumps(body).encode()
    req = urllib.request.Request(url, data=body, headers=headers)
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        return r.read(MAX_BYTES)


# ===== Parse =====
def clean(text):
    text = re.sub(r"<[^>]+>", " ", text or "")  # strip tags; rendering escapes the rest
    return " ".join(text.split())


def safe_url(u):
    # only http(s) links survive (same idea as morning_paper.safe_url)
    return u if urllib.parse.urlparse(u or "").scheme in ("http", "https") else ""


def parse_arxiv(raw, section, limit):
    items = []
    for e in ET.fromstring(raw).findall("a:entry", NS)[:limit]:
        text = lambda tag: (e.findtext(f"a:{tag}", default="", namespaces=NS))
        summary = clean(text("summary"))
        items.append({
            "title": clean(text("title")), "summary": summary[:SUMMARY_LEN],
            "url": safe_url(text("id")), "section": section,
            "source": "arXiv", "date": text("published")[:10],
        })
    return items


def parse_feed(raw, src, limit):
    """Handles RSS 2.0 (<item>) and Atom (<entry>); keeps entries matching src keywords."""
    root = ET.fromstring(raw)
    kws = [k.lower() for k in src.get("keywords", [])]
    items = []
    for e in list(root.iter("item")) + list(root.iter("{%s}entry" % NS["a"])):
        title = clean(e.findtext("title") or e.findtext("a:title", namespaces=NS))
        body = clean(e.findtext("description") or e.findtext("a:summary", namespaces=NS)
                     or e.findtext("a:content", namespaces=NS))
        link = e.findtext("link")
        if not link:
            l = e.find("a:link", NS)
            link = l.get("href") if l is not None else ""
        date = e.findtext("pubDate") or e.findtext("a:updated", namespaces=NS) or ""
        hay = f"{title} {body}".lower()
        if kws and not any(k in hay for k in kws):
            continue
        items.append({
            "title": title, "summary": body[:SUMMARY_LEN], "url": safe_url(link),
            "section": src["section"], "source": src["name"], "date": date.strip(),
        })
        if len(items) >= limit:
            break
    return items


def parse_osv(raw, pkg, limit):
    """OSV returns every known advisory for a package; keep the most recently modified ones.
    Only metadata is recorded: the package is never installed or executed."""
    vulns = json.loads(raw).get("vulns", [])
    vulns.sort(key=lambda v: v.get("modified", ""), reverse=True)
    items = []
    for v in vulns[:limit]:
        vid = v.get("id", "")
        kind = "악성 패키지" if vid.startswith("MAL-") else "취약점"
        text = clean(v.get("summary") or v.get("details") or "")
        items.append({
            "title": f"[{pkg['ecosystem']}] {pkg['name']} {kind}: {vid}",
            "summary": text[:SUMMARY_LEN],
            "url": safe_url("https://osv.dev/vulnerability/" + urllib.parse.quote(vid)),
            "section": pkg["section"], "source": "OSV", "date": v.get("modified", "")[:10],
        })
    return items


# ===== Rank =====
def parse_date(s):
    """RSS dates are RFC 822, Atom/OSV/arXiv are ISO 8601; returns an aware datetime or None."""
    try:
        d = parsedate_to_datetime(s)
    except (TypeError, ValueError):
        try:
            d = datetime.fromisoformat((s or "").replace("Z", "+00:00"))
        except ValueError:
            return None
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


def score(item, prio, now):
    hay = f"{item['title']} {item['summary']}".lower()
    s = 3 * min(sum(k.lower() in hay for k in prio), 3)  # capped so keyword stuffing cannot dominate
    d = parse_date(item["date"])
    if d:
        age = (now - d).days
        s += 3 if age <= 7 else 2 if age <= 30 else 1 if age <= 90 else 0
    if item["source"] == "OSV":
        s += 4  # packages are on the watch list on purpose; old advisories must not be capped out
    if "MAL-" in item["title"]:
        s += 5  # confirmed malicious-package reports always float up
    return s


def rank(items, cfg, now):
    seen_urls, out = set(), []
    for i in items:  # dedupe by URL (same advisory returned for several packages)
        if i["url"] not in seen_urls:
            seen_urls.add(i["url"])
            i["score"] = score(i, cfg.get("priority_keywords", []), now)
            out.append(i)
    out.sort(key=lambda i: (i["section"], -i["score"]))
    cap, kept, count = cfg.get("max_per_section", 10), [], {}
    for i in out:
        count[i["section"]] = count.get(i["section"], 0) + 1
        if count[i["section"]] <= cap:
            kept.append(i)
    return kept


def load_seen(path):
    try:
        return set(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError):
        return set()


# ===== Main =====
def collect(cfg):
    hosts, limit, items, errors = allowed_hosts(cfg), cfg.get("max_items_per_source", 8), [], []
    for s in cfg.get("arxiv", []):
        q = urllib.parse.urlencode({"search_query": s["query"], "max_results": limit,
                                    "sortBy": "submittedDate", "sortOrder": "descending"})
        try:
            items += parse_arxiv(fetch(f"{ARXIV_API}?{q}", hosts), s["section"], limit)
        except Exception as ex:  # one failing source must not stop the others
            errors.append(("arXiv", repr(ex)))
    for p in cfg.get("osv_packages", []):
        try:
            body = {"package": {"name": p["name"], "ecosystem": p["ecosystem"]}}
            items += parse_osv(fetch(OSV_API, hosts, body), p, limit)
        except Exception as ex:
            errors.append((f"OSV {p['name']}", repr(ex)))
    for s in cfg.get("rss", []):
        try:
            items += parse_feed(fetch(s["url"], hosts), s, limit)
        except Exception as ex:
            errors.append((s["name"], repr(ex)))
    return [i for i in items if i["title"] and i["url"]], errors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", default=str(HERE / "sources.json"))
    ap.add_argument("--out", default=str(HERE / "news.json"))
    ap.add_argument("--seen", default=str(SEEN))
    ap.add_argument("--all", action="store_true", help="ignore and do not update the seen list")
    ap.add_argument("--merge", action="store_true", help="keep items already in --out (same-day reruns accumulate)")
    a = ap.parse_args()
    cfg = json.loads(Path(a.sources).read_text(encoding="utf-8"))
    items, errors = collect(cfg)
    items = rank(items, cfg, datetime.now(timezone.utc))
    seen_path = Path(a.seen)
    if not a.all:
        seen = load_seen(seen_path)
        items = [i for i in items if i["url"] not in seen]
        seen_path.parent.mkdir(parents=True, exist_ok=True)
        seen_path.write_text(json.dumps(sorted(seen | {i["url"] for i in items})), encoding="utf-8")
    if a.merge and Path(a.out).exists():
        try:
            old = json.loads(Path(a.out).read_text(encoding="utf-8"))
        except ValueError:
            old = []
        items = rank(items + old, cfg, datetime.now(timezone.utc))  # new first; rank dedupes by URL
    Path(a.out).write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{len(items)} items -> {a.out}  ({datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC)")
    by = {}
    for i in items:
        by[i["source"]] = by.get(i["source"], 0) + 1
    for k, v in by.items():
        print(f"  ok   {k}: {v}")
    for name, err in errors:
        print(f"  FAIL {name}: {err}", file=sys.stderr)


if __name__ == "__main__":
    main()
