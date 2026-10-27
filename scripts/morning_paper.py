"""Build the daily "morning knowledge newspaper" (static HTML) from public arXiv papers + web news.

Usage:  python scripts/morning_paper.py [--date YYYY-MM-DD] [--news news.json] [--out DIR]
news.json: [{"title": "...", "summary": "...", "url": "https://..."}]  (written by the search step)
Private data is never read: only data/arxiv/papers.jsonl (public).
"""
import argparse
import html
import json
from datetime import date
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent.parent
PAPERS = ROOT / "data" / "arxiv" / "papers.jsonl"
DEFAULT_OUT = ROOT.parent / "jarvis_drafts" / "morning_paper"
N_PAPERS = 5


# ===== Data =====
def pick_papers(day, n=N_PAPERS):
    """Newest papers on or before `day`, so each edition is reproducible."""
    rows = []
    with PAPERS.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("date", "") <= day:
                rows.append(r)
    rows.sort(key=lambda r: (r["date"], r["id"]), reverse=True)
    return rows[:n]


def safe_url(u):
    # only http(s) links are rendered as hrefs (XSS guard, same idea as app/static)
    return u if urlparse(u).scheme in ("http", "https") else ""


# ===== Render =====
CSS = """body{font-family:Georgia,'Malgun Gothic',serif;max-width:860px;margin:2rem auto;padding:0 1rem;color:#1a1a1a;background:#faf8f3}
h1{text-align:center;font-size:2.4rem;border-bottom:4px double #1a1a1a;padding-bottom:.4rem;margin-bottom:.2rem}
.date{text-align:center;color:#555;margin-bottom:1.5rem}h2{border-bottom:1px solid #1a1a1a;margin-top:2rem}
article{margin:1rem 0}article h3{margin:0 0 .2rem}.meta{color:#666;font-size:.85rem}a{color:#0b4f8a}
nav{text-align:center;margin:.5rem 0}.note{background:#f0ece0;padding:.6rem;font-size:.9rem}"""


def esc(s):
    return html.escape(s, quote=True)


def render(day, papers, news, prev_day, next_day, title="아침지식 신문"):
    nav = " · ".join(
        [f'<a href="{d}.html">{lbl}</a>' for d, lbl in ((prev_day, "← 어제"), (next_day, "내일 →")) if d]
        + ['<a href="index.html">전체 목록</a>']
    )
    parts = [f"<!doctype html><html lang='ko'><meta charset='utf-8'><title>{esc(title)} {day}</title>",
             f"<style>{CSS}</style><h1>{esc(title)}</h1><div class='date'>{day}</div><nav>{nav}</nav>"]
    if papers:
        parts.append("<h2>오늘의 논문 (Jarvis · arXiv 의료 AI)</h2>")
    for p in papers:
        ptitle, _, abstract = p["text"].partition("\n\n")
        abstract = " ".join(abstract.split())
        short = abstract[:320] + ("…" if len(abstract) > 320 else "")
        aid = p["id"].split(":", 1)[-1]
        parts.append(f"<article><h3>{esc(ptitle)}</h3><div class='meta'>{esc(p['date'])} · "
                     f"<a href='https://arxiv.org/abs/{esc(aid)}'>arXiv:{esc(aid)}</a></div><p>{esc(short)}</p></article>")
    if news:
        # group by optional "section" (e.g. 양자, 의료 AI); items without one go under "웹 뉴스"
        sections = {}
        for n in news:
            sections.setdefault(n.get("section") or "웹 뉴스", []).append(n)
        for name, items in sections.items():
            parts.append(f"<h2>{esc(name)}</h2>")
            for n in items:
                link = safe_url(n.get("url", ""))
                t = esc(n["title"])
                t = f"<a href='{esc(link)}'>{t}</a>" if link else t
                # optional "source" / "date" (written by security_feed/collect.py)
                meta = " · ".join(esc(str(n[k])[:16] if k == "date" else str(n[k])) for k in ("source", "date") if n.get(k))
                meta = f"<div class='meta'>{meta}</div>" if meta else ""
                parts.append(f"<article><h3>{t}</h3>{meta}<p>{esc(n.get('summary', ''))}</p></article>")
    else:
        parts.append("<h2>웹 뉴스</h2>")
        parts.append("<p class='note'>오늘은 신뢰할 만한 뉴스를 찾지 못했습니다.</p>")
    parts.append("<p class='note'>요약은 원문 발췌이며 LLM 요약이 아닙니다. 웹 뉴스는 자동 수집 결과라 원문 확인이 필요합니다.</p></html>")
    return "\n".join(parts)


def rebuild_index(out, title="아침지식 신문"):
    days = sorted((p.stem for p in out.glob("20??-??-??.html")), reverse=True)
    items = "".join(f"<li><a href='{d}.html'>{d}</a></li>" for d in days)
    (out / "index.html").write_text(
        f"<!doctype html><html lang='ko'><meta charset='utf-8'><title>{esc(title)} 목록</title>"
        f"<style>{CSS}</style><h1>{esc(title)}</h1><ul>{items}</ul></html>", encoding="utf-8")
    return days


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=date.today().isoformat())
    ap.add_argument("--news")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--title", default="아침지식 신문")
    ap.add_argument("--no-papers", action="store_true", help="news only (e.g. the security edition)")
    # preset, so callers (PowerShell 5.1) need not pass non-ASCII arguments on the command line
    ap.add_argument("--edition", choices=["security"], help="security = --title 'AI 보안 신문' --no-papers")
    a = ap.parse_args()
    if a.edition == "security":
        a.title, a.no_papers = "AI 보안 신문", True
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    news = json.loads(Path(a.news).read_text(encoding="utf-8")) if a.news else []
    days = sorted(p.stem for p in out.glob("20??-??-??.html") if p.stem != a.date)
    prev_day = max((d for d in days if d < a.date), default=None)
    next_day = min((d for d in days if d > a.date), default=None)
    page = out / f"{a.date}.html"
    papers = [] if a.no_papers else pick_papers(a.date)
    page.write_text(render(a.date, papers, news, prev_day, next_day, a.title), encoding="utf-8")
    rebuild_index(out, a.title)
    print(page)


if __name__ == "__main__":
    main()
