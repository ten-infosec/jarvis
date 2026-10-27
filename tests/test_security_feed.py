"""Tests for scripts/security_feed/collect.py and the news-only mode of morning_paper.py.
Fake responses only: no network, no keys.
"""
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


collect = _load("sec_collect", "scripts/security_feed/collect.py")
paper = _load("sec_paper", "scripts/morning_paper.py")

NOW = datetime(2026, 10, 25, tzinfo=timezone.utc)

RSS = b"""<?xml version="1.0"?><rss><channel>
<item><title>LiteLLM key leak</title><description>&lt;b&gt;bad&lt;/b&gt; token theft</description>
<link>https://example.com/a</link><pubDate>Fri, 23 Oct 2026 10:00:00 GMT</pubDate></item>
<item><title>Cooking tips</title><description>pasta</description><link>https://example.com/b</link></item>
<item><title>Agent exploit</title><description>x</description><link>javascript:alert(1)</link></item>
</channel></rss>"""

ATOM = b"""<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom">
<entry><title>Prompt injection defense</title><summary>agent security</summary>
<id>https://arxiv.org/abs/1</id><published>2026-10-22T00:00:00Z</published></entry></feed>"""


# ===== Parsing =====
def test_parse_arxiv():
    items = collect.parse_arxiv(ATOM, "방어", 8)
    assert items[0]["title"] == "Prompt injection defense"
    assert items[0]["url"] == "https://arxiv.org/abs/1" and items[0]["date"] == "2026-10-22"


def test_parse_feed_keyword_filter_and_html_strip():
    src = {"name": "T", "section": "S", "keywords": ["litellm", "agent"]}
    items = collect.parse_feed(RSS, src, 8)
    assert [i["title"] for i in items] == ["LiteLLM key leak", "Agent exploit"]
    assert "<b>" not in items[0]["summary"]
    assert items[1]["url"] == ""  # javascript: link is dropped


def test_parse_osv_marks_malicious_and_sorts():
    raw = json.dumps({"vulns": [
        {"id": "GHSA-old", "summary": "old", "modified": "2025-01-01T00:00:00Z"},
        {"id": "MAL-2026-1", "summary": "Malicious code", "modified": "2026-10-21T00:00:00Z"},
    ]})
    pkg = {"ecosystem": "PyPI", "name": "litellm", "section": "S"}
    items = collect.parse_osv(raw, pkg, 1)
    assert len(items) == 1 and "악성 패키지" in items[0]["title"]
    assert items[0]["url"] == "https://osv.dev/vulnerability/MAL-2026-1"


# ===== Safety =====
def test_fetch_rejects_unlisted_host_and_http():
    hosts = {"example.com"}
    with pytest.raises(ValueError):
        collect.fetch("https://evil.test/x", hosts)
    with pytest.raises(ValueError):
        collect.fetch("http://example.com/x", hosts)


def test_collect_drops_items_without_url(monkeypatch):
    monkeypatch.setattr(collect, "fetch", lambda *a, **k: RSS)
    cfg = {"rss": [{"name": "T", "section": "S", "url": "https://example.com/f", "keywords": ["agent"]}]}
    items, errors = collect.collect(cfg)
    assert items == [] and errors == []


def test_one_failing_source_does_not_stop_others(monkeypatch):
    def fake(url, hosts, body=None):
        if "bad" in url:
            raise OSError("boom")
        return RSS
    monkeypatch.setattr(collect, "fetch", fake)
    cfg = {"rss": [{"name": "Bad", "section": "S", "url": "https://bad.example/f"},
                   {"name": "Good", "section": "S", "url": "https://good.example/f", "keywords": ["litellm"]}]}
    items, errors = collect.collect(cfg)
    assert len(items) == 1 and errors[0][0] == "Bad"


# ===== Ranking =====
def _item(title, url, source="X", date="2026-10-24"):
    return {"title": title, "summary": "", "url": url, "section": "S", "source": source, "date": date}


def test_rank_dedupes_and_orders():
    cfg = {"priority_keywords": ["litellm"], "max_per_section": 10}
    items = [_item("plain", "u1"), _item("litellm leak", "u2"), _item("litellm leak", "u2"),
             _item("[PyPI] x MAL-1", "u3", source="OSV")]
    out = collect.rank(items, cfg, NOW)
    assert [i["url"] for i in out] == ["u3", "u2", "u1"]


def test_rank_caps_per_section():
    out = collect.rank([_item(f"t{n}", f"u{n}") for n in range(5)], {"max_per_section": 2}, NOW)
    assert len(out) == 2


def test_parse_date_formats():
    assert collect.parse_date("Fri, 23 Oct 2026 10:00:00 GMT").day == 23
    assert collect.parse_date("2026-10-22T00:00:00Z").day == 22
    assert collect.parse_date("") is None


# ===== Seen list =====
def test_second_run_yields_no_new_items(monkeypatch, tmp_path):
    items = [_item("litellm", "https://e.com/1")]
    monkeypatch.setattr(collect, "collect", lambda cfg: (list(items), []))
    src = tmp_path / "s.json"
    src.write_text("{}", encoding="utf-8")
    seen, out = tmp_path / "seen.json", tmp_path / "n.json"
    argv = ["collect.py", "--sources", str(src), "--out", str(out), "--seen", str(seen)]
    monkeypatch.setattr(sys, "argv", argv)
    collect.main()
    assert len(json.loads(out.read_text(encoding="utf-8"))) == 1
    collect.main()
    assert json.loads(out.read_text(encoding="utf-8")) == []


def test_merge_keeps_earlier_items_of_the_same_day(monkeypatch, tmp_path):
    src = tmp_path / "s.json"
    src.write_text("{}", encoding="utf-8")
    seen, out = tmp_path / "seen.json", tmp_path / "n.json"
    monkeypatch.setattr(sys, "argv", ["collect.py", "--sources", str(src), "--out", str(out),
                                      "--seen", str(seen), "--merge"])
    monkeypatch.setattr(collect, "collect", lambda cfg: ([_item("first", "https://e.com/1")], []))
    collect.main()
    monkeypatch.setattr(collect, "collect", lambda cfg: ([_item("second", "https://e.com/2")], []))
    collect.main()
    urls = {i["url"] for i in json.loads(out.read_text(encoding="utf-8"))}
    assert urls == {"https://e.com/1", "https://e.com/2"}


# ===== Dependency check =====
check_deps = _load("sec_check_deps", "scripts/security_feed/check_deps.py")


def test_read_pins_skips_comments_and_reports_unpinned(tmp_path):
    f = tmp_path / "r.txt"
    f.write_text("# c\nfoo==1.0\n\nbar>=2\n-r other.txt\nBaz_x==3.1 ; python_version>'3'\n", encoding="utf-8")
    pins, loose = check_deps.read_pins(f)
    assert pins == [("foo", "1.0"), ("Baz_x", "3.1")] and loose == ["bar>=2"]


def test_query_maps_results_in_order():
    pins = [("a", "1"), ("b", "2")]
    fake = lambda url, hosts, body=None: json.dumps(
        {"results": [{}, {"vulns": [{"id": "MAL-1"}, {"id": "GHSA-x"}]}]}).encode()
    assert check_deps.query(pins, fetch=fake) == {("a", "1"): [], ("b", "2"): ["MAL-1", "GHSA-x"]}


def test_check_deps_exit_codes(monkeypatch, tmp_path, capsys):
    f = tmp_path / "r.txt"
    f.write_text("a==1\n", encoding="utf-8")
    monkeypatch.setattr(check_deps, "query", lambda pins: {("a", "1"): []})
    assert check_deps.main([str(f)]) == 0
    monkeypatch.setattr(check_deps, "query", lambda pins: {("a", "1"): ["MAL-9"]})
    assert check_deps.main([str(f)]) == 1 and "MALICIOUS" in capsys.readouterr().out

    def boom(pins):
        raise OSError("down")
    monkeypatch.setattr(check_deps, "query", boom)
    assert check_deps.main([str(f)]) == 2  # a failed lookup is never reported as "clean"


# ===== Newspaper rendering =====
def test_news_only_edition_escapes_hostile_input():
    news = [{"title": "<script>x</script>", "summary": "<img src=x onerror=1>", "url": "javascript:alert(1)",
             "section": "S", "source": "<b>e</b>", "date": "2026-10-25"}]
    html = paper.render("2026-10-25", [], news, None, None, title="AI 보안 신문")
    assert "<script>" not in html and "<img" not in html and "href='javascript" not in html
    assert "AI 보안 신문" in html and "오늘의 논문" not in html
