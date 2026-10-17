"""Tests for app/main.py (rate limit, XFF handling, validation, error hiding, docs, logging).

Runs without API keys or network: app.rag_cloud / app.rag are replaced by fake modules.
Run from the repo root:  python -m pytest tests -v
"""
import importlib
import json
import sys
import types
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

HIT = {"n": 1, "doc_id": "d1", "source": "arxiv", "date": "2026-01-01",
       "url": "https://arxiv.org/abs/1", "via": "bm25", "text": "t"}


# ===== Fake engines =====
def make_engine(fail=False):
    class FakeEngine:
        def __init__(self):
            self.docs = {"d1": "t"}

        def retrieve(self, question, source, mode, top_n):
            if fail:
                raise RuntimeError("secret internal detail")
            return "q", [dict(HIT)]

        def generate(self, question, hits):
            return "answer"
    return FakeEngine


def load_main(monkeypatch, tmp_path, mode="cloud", fail=False):
    # Fresh import of app.main with the requested mode and a fake engine module
    monkeypatch.setenv("JARVIS_DATA_DIR", str(tmp_path))
    if mode == "cloud":
        monkeypatch.setenv("JARVIS_MODE", "cloud")
        fake = types.ModuleType("app.rag_cloud")
        fake.CloudRAG = make_engine(fail)
        monkeypatch.setitem(sys.modules, "app.rag_cloud", fake)
    else:
        monkeypatch.delenv("JARVIS_MODE", raising=False)
        fake = types.ModuleType("app.rag")
        fake.RAG = make_engine(fail)
        monkeypatch.setitem(sys.modules, "app.rag", fake)
    sys.modules.pop("app.main", None)
    main = importlib.import_module("app.main")
    main._recent.clear()
    main._daily.update(date=None, count=0)
    return main


@pytest.fixture
def cloud(monkeypatch, tmp_path):
    main = load_main(monkeypatch, tmp_path, "cloud")
    return main, TestClient(main.app)


def ask(client, xff=None, **body):
    payload = {"question": "hello"}
    payload.update(body)
    headers = {} if xff is None else {"X-Forwarded-For": xff}
    return client.post("/ask", json=payload, headers=headers)


def read_log(main):
    return [json.loads(l) for l in main.LOG_FILE.read_text(encoding="utf-8").splitlines()]


# ===== 1. Per-IP rate limit =====
def test_rate_limit_same_ip(cloud):
    main, c = cloud
    codes = [ask(c, "9.9.9.9").status_code for _ in range(6)]
    assert codes == [200] * 5 + [429]


# ===== 2. Spoofed leftmost entries, same rightmost =====
def test_spoofing_leftmost_ignored(cloud):
    main, c = cloud
    codes = [ask(c, f"1.1.1.{n}, 9.9.9.9").status_code for n in range(6)]
    assert codes == [200] * 5 + [429]
    assert list(main._recent.keys()) == ["9.9.9.9"]


# ===== 3. Different rightmost IPs are separate =====
def test_different_ips_separate(cloud):
    main, c = cloud
    for _ in range(5):
        assert ask(c, "2.2.2.2").status_code == 200
    assert ask(c, "2.2.2.2").status_code == 429
    assert ask(c, "3.3.3.3").status_code == 200
    assert ask(c, "1.1.1.1, 3.3.3.3").status_code == 200  # same client as above


# ===== 4. Daily cap =====
def test_daily_cap(cloud):
    main, c = cloud
    main.DAILY_CAP = 3
    codes = [ask(c, f"5.5.5.{i}").status_code for i in range(6)]
    assert codes == [200, 200, 200, 429, 429, 429]
    assert len(main._recent) == 3
    assert "5.5.5.3" not in main._recent
    assert main._daily["count"] == 3
    rejected = [r for r in read_log(main) if r["status"] == "rejected"]
    assert len(rejected) == 3 and "Daily limit" in rejected[0]["error"]


def test_per_ip_reject_does_not_grow_deque(cloud):
    main, c = cloud
    for _ in range(10):
        ask(c, "9.9.9.9")
    assert len(main._recent["9.9.9.9"]) == 5
    assert main._daily["count"] == 5


# ===== 5. Memory bound =====
def test_max_tracked_ips(cloud):
    main, c = cloud
    main.MAX_TRACKED_IPS = 5
    main.DAILY_CAP = 1000
    for i in range(20):
        assert ask(c, f"10.0.0.{i}").status_code == 200
        assert len(main._recent) <= 5
    # the newest IPs are the ones kept
    assert list(main._recent.keys()) == [f"10.0.0.{i}" for i in range(15, 20)]


# ===== 6. source validation =====
@pytest.mark.parametrize("source,expected", [
    ("saved", 200), (None, 200), ("x" * 1000, 422), ("", 422), ("my_post", 200), ("bogus", 422),
], ids=["saved", "null", "x1000", "empty", "my_post", "bogus"])
def test_source_validation(cloud, source, expected):
    main, c = cloud
    r = ask(c, "4.4.4.4", source=source)
    assert r.status_code == expected


def test_422_does_not_consume_rate_limit(cloud):
    main, c = cloud
    for _ in range(10):
        assert ask(c, "4.4.4.4", source="bad").status_code == 422
    assert main._daily["count"] == 0 and len(main._recent) == 0


# ===== 7. Cloud 500 hides details, log keeps them =====
def test_cloud_error_hidden(monkeypatch, tmp_path, capsys):
    main = load_main(monkeypatch, tmp_path, "cloud", fail=True)
    c = TestClient(main.app)
    r = ask(c, "9.9.9.9")
    assert r.status_code == 500
    assert "secret internal detail" not in r.text
    assert r.json()["detail"] == "Internal error. Please try again later."
    log_text = (tmp_path / "logs" / "requests.jsonl").read_text(encoding="utf-8")
    assert "secret internal detail" in log_text
    out = capsys.readouterr().out
    assert '"severity": "ERROR"' in out and "secret internal detail" in out


def test_local_error_shows_details(monkeypatch, tmp_path):
    main = load_main(monkeypatch, tmp_path, "local", fail=True)
    r = ask(TestClient(main.app))
    assert r.status_code == 500
    assert "RuntimeError: secret internal detail" in r.json()["detail"]


# ===== 8. /docs visibility =====
def test_docs_hidden_in_cloud(cloud):
    main, c = cloud
    for path in ("/docs", "/openapi.json", "/redoc"):
        assert c.get(path).status_code == 404


def test_docs_available_in_local(monkeypatch, tmp_path):
    main = load_main(monkeypatch, tmp_path, "local")
    c = TestClient(main.app)
    assert c.get("/docs").status_code == 200
    assert c.get("/openapi.json").status_code == 200
    # local mode: no rate limit
    assert all(ask(c).status_code == 200 for _ in range(8))


# ===== 9. client_ip edge cases =====
class FakeReq:
    def __init__(self, xff=None, host="7.7.7.7"):
        self.headers = {} if xff is None else {"x-forwarded-for": xff}
        self.client = types.SimpleNamespace(host=host) if host else None


def test_client_ip_no_header(cloud):
    main, _ = cloud
    assert main.client_ip(FakeReq()) == "7.7.7.7"
    assert main.client_ip(FakeReq(host=None)) == "unknown"


def test_client_ip_empty_header(cloud):
    main, _ = cloud
    assert main.client_ip(FakeReq("")) == "7.7.7.7"


def test_client_ip_blank_entries(cloud):
    main, _ = cloud
    assert main.client_ip(FakeReq("a, ,b")) == "b"
    assert main.client_ip(FakeReq("a,,b, ")) == "b"


def test_client_ip_hops_larger_than_entries(cloud):
    main, _ = cloud
    main.TRUSTED_PROXY_HOPS = 5
    assert main.client_ip(FakeReq("a, b")) == "a"
    main.TRUSTED_PROXY_HOPS = 1
    assert main.client_ip(FakeReq("a, b, c")) == "b"


def test_client_ip_only_commas(cloud):
    # Header with no real entries (e.g. "," or " , ") must not crash
    main, _ = cloud
    assert main.client_ip(FakeReq(" , ")) == "7.7.7.7"


def test_only_commas_over_http(cloud):
    main, c = cloud
    r = ask(c, ",")
    assert r.status_code == 200


def test_empty_header_over_http(cloud):
    main, c = cloud
    assert ask(c, "").status_code == 200


# ===== 10. Log content =====
def test_log_has_hops_no_ip(cloud, capsys):
    main, c = cloud
    ask(c, "1.2.3.4, 5.6.7.8, 9.10.11.12")
    rec = read_log(main)[-1]
    assert rec["xff_hops"] == 3 and rec["status"] == "ok"
    raw = main.LOG_FILE.read_text(encoding="utf-8")
    for ip in ("1.2.3.4", "5.6.7.8", "9.10.11.12", "testclient"):
        assert ip not in raw
    out = capsys.readouterr().out
    assert "9.10.11.12" not in out and '"xff_hops": 3' in out
