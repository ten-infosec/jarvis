"""Tests for POST /ask/stream (NDJSON streaming) and the engines' generate_stream().

Same approach as tests/test_main.py: no API keys, no network, engines are fake modules.
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
GENERIC = "Internal error. Please try again later."


# ===== Fake engines =====
def make_engine(kind="stream", fail_retrieve=False):
    # kind: "stream"   -> has generate_stream, yields 3 pieces (one empty, which must be skipped)
    #       "nostream" -> only generate() (fallback path)
    #       "fail_mid" -> generate_stream yields one piece, then raises
    class FakeEngine:
        calls = []

        def __init__(self):
            self.docs = {"d1": "t"}

        def retrieve(self, question, source, mode, top_n):
            if fail_retrieve:
                raise RuntimeError("secret retrieve detail")
            return "q", [dict(HIT)]

        def generate(self, question, hits):
            FakeEngine.calls.append("generate")
            return "answer"

    if kind == "stream":
        def generate_stream(self, question, hits):
            FakeEngine.calls.append("generate_stream")
            yield "Hel"
            yield ""
            yield "lo **[1]**"
        FakeEngine.generate_stream = generate_stream
    elif kind == "fail_mid":
        def generate_stream(self, question, hits):
            yield "partial "
            raise RuntimeError("secret internal detail")
        FakeEngine.generate_stream = generate_stream
    return FakeEngine


def load_main(monkeypatch, tmp_path, mode="cloud", kind="stream", fail_retrieve=False):
    # Fresh import of app.main with the requested mode and a fake engine module
    monkeypatch.setenv("JARVIS_DATA_DIR", str(tmp_path))
    engine = make_engine(kind, fail_retrieve)
    if mode == "cloud":
        monkeypatch.setenv("JARVIS_MODE", "cloud")
        fake = types.ModuleType("app.rag_cloud")
        fake.CloudRAG = engine
        monkeypatch.setitem(sys.modules, "app.rag_cloud", fake)
    else:
        monkeypatch.delenv("JARVIS_MODE", raising=False)
        fake = types.ModuleType("app.rag")
        fake.RAG = engine
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


def post(client, path="/ask/stream", xff=None, **body):
    payload = {"question": "hello"}
    payload.update(body)
    headers = {} if xff is None else {"X-Forwarded-For": xff}
    return client.post(path, json=payload, headers=headers)


def events(resp):
    return [json.loads(l) for l in resp.text.splitlines() if l.strip()]


def read_log(main):
    if not main.LOG_FILE.exists():
        return []
    return [json.loads(l) for l in main.LOG_FILE.read_text(encoding="utf-8").splitlines()]


# ===== 1. Event order and content =====
def test_event_order(cloud):
    main, c = cloud
    r = post(c, xff="9.9.9.9")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/x-ndjson")
    evs = events(r)
    types_ = [e["type"] for e in evs]
    assert types_[0] == "sources" and types_[-1] == "done"
    assert set(types_[1:-1]) == {"chunk"}
    # empty pieces are not sent
    assert [e["text"] for e in evs if e["type"] == "chunk"] == ["Hel", "lo **[1]**"]
    assert evs[-1]["answer_chars"] == len("Hello **[1]**")
    assert set(evs[-1]["timing_sec"]) == {"retrieve", "generate"}


def test_sources_same_shape_as_ask(cloud):
    main, c = cloud
    plain = post(c, "/ask", xff="1.1.1.1").json()
    first = events(post(c, xff="2.2.2.2"))[0]
    assert first["sources"] == plain["sources"]
    assert first["search_query"] == plain["search_query"]
    assert first["question"] == plain["question"]
    # full text never leaves the server, only the preview
    assert "text" not in first["sources"][0]


def test_one_log_record_ok(cloud, capsys):
    main, c = cloud
    post(c, xff="1.2.3.4, 5.6.7.8")
    recs = read_log(main)
    assert len(recs) == 1
    rec = recs[0]
    assert rec["status"] == "ok" and rec["stream"] is True
    assert rec["answer_chars"] == len("Hello **[1]**")
    assert rec["xff_hops"] == 2 and rec["doc_ids"] == ["d1"]
    raw = main.LOG_FILE.read_text(encoding="utf-8")
    assert "1.2.3.4" not in raw and "5.6.7.8" not in raw
    assert '"severity": "INFO"' in capsys.readouterr().out


def test_retrieve_only_no_chunks(cloud):
    main, c = cloud
    main.rag.calls.clear()
    evs = events(post(c, xff="9.9.9.9", retrieve_only=True))
    assert [e["type"] for e in evs] == ["sources", "done"]
    assert main.rag.calls == []
    assert read_log(main)[-1]["answer_chars"] == 0


# ===== 2. Rate limit applies (counted once per request) =====
def test_rate_limit_stream(cloud):
    main, c = cloud
    codes = [post(c, xff="9.9.9.9").status_code for _ in range(6)]
    assert codes == [200] * 5 + [429]
    assert main._daily["count"] == 5
    assert len(main._recent["9.9.9.9"]) == 5
    r = post(c, xff="9.9.9.9")
    assert "Too many requests" in r.json()["detail"]
    recs = read_log(main)
    assert [x["status"] for x in recs] == ["ok"] * 5 + ["rejected"] * 2


def test_rate_limit_shared_with_ask(cloud):
    main, c = cloud
    for i in range(5):
        path = "/ask" if i % 2 else "/ask/stream"
        assert post(c, path, xff="8.8.8.8").status_code == 200
    assert post(c, "/ask/stream", xff="8.8.8.8").status_code == 429
    assert post(c, "/ask", xff="8.8.8.8").status_code == 429


def test_daily_cap_stream(cloud):
    main, c = cloud
    main.DAILY_CAP = 2
    codes = [post(c, xff=f"5.5.5.{i}").status_code for i in range(4)]
    assert codes == [200, 200, 429, 429]
    assert "Daily limit" in post(c, xff="5.5.5.9").json()["detail"]


def test_local_mode_no_limit(monkeypatch, tmp_path):
    main = load_main(monkeypatch, tmp_path, "local")
    c = TestClient(main.app)
    assert all(post(c).status_code == 200 for _ in range(8))


# ===== 3. Validation (same AskRequest -> 422, no limit used, nothing logged) =====
@pytest.mark.parametrize("body", [
    {"question": "x"}, {"question": "x" * 501}, {"source": "bogus"}, {"source": ""},
    {"top_n": 0}, {"top_n": 11}, {"mode": "magic"},
], ids=["short", "long", "bogus-source", "empty-source", "top0", "top11", "mode"])
def test_validation_422(cloud, body):
    main, c = cloud
    r = post(c, xff="4.4.4.4", **body)
    assert r.status_code == 422
    assert main._daily["count"] == 0 and len(main._recent) == 0
    assert read_log(main) == []


# ===== 4. Errors: details hidden in cloud, kept in log =====
def test_mid_stream_error_hidden_in_cloud(monkeypatch, tmp_path, capsys):
    main = load_main(monkeypatch, tmp_path, "cloud", kind="fail_mid")
    r = post(TestClient(main.app), xff="9.9.9.9")
    # headers were already sent -> 200, and the error is the last event
    assert r.status_code == 200
    evs = events(r)
    assert [e["type"] for e in evs] == ["sources", "chunk", "error"]
    assert evs[-1]["detail"] == GENERIC
    assert "secret internal detail" not in r.text
    recs = read_log(main)
    assert len(recs) == 1 and recs[0]["status"] == "error"
    assert "secret internal detail" in recs[0]["error"]
    assert recs[0]["answer_chars"] == len("partial ")
    out = capsys.readouterr().out
    assert '"severity": "ERROR"' in out and "secret internal detail" in out


def test_mid_stream_error_shows_details_locally(monkeypatch, tmp_path):
    main = load_main(monkeypatch, tmp_path, "local", kind="fail_mid")
    evs = events(post(TestClient(main.app)))
    assert evs[-1] == {"type": "error", "detail": "RuntimeError: secret internal detail"}


def test_retrieve_error_is_plain_500_in_cloud(monkeypatch, tmp_path):
    main = load_main(monkeypatch, tmp_path, "cloud", fail_retrieve=True)
    r = post(TestClient(main.app), xff="9.9.9.9")
    assert r.status_code == 500
    assert r.json()["detail"] == GENERIC
    assert "secret" not in r.text
    recs = read_log(main)
    assert len(recs) == 1 and recs[0]["status"] == "error"
    assert "secret retrieve detail" in recs[0]["error"]


# ===== 5. Fallback: engine without generate_stream =====
def test_fallback_without_generate_stream(monkeypatch, tmp_path):
    main = load_main(monkeypatch, tmp_path, "cloud", kind="nostream")
    assert not hasattr(main.rag, "generate_stream")
    evs = events(post(TestClient(main.app), xff="9.9.9.9"))
    assert [e["type"] for e in evs] == ["sources", "chunk", "done"]
    assert evs[1]["text"] == "answer"
    assert read_log(main)[-1]["answer_chars"] == len("answer")


def test_ask_unchanged(cloud):
    # /ask keeps its old JSON shape (and still uses generate(), not the stream)
    main, c = cloud
    main.rag.calls.clear()
    body = post(c, "/ask", xff="9.9.9.9").json()
    assert set(body) == {"question", "search_query", "answer", "sources", "timing_sec"}
    assert body["answer"] == "answer"
    assert main.rag.calls == ["generate"]
    assert "stream" not in read_log(main)[-1]


# ===== 6. Engines' generate_stream (heavy libraries replaced by stubs) =====
@pytest.fixture
def stub_heavy_libs(monkeypatch):
    # chromadb / rank_bm25 / sentence_transformers / google-genai are not test dependencies:
    # only the names the engine modules import at the top are needed
    for name in ("chromadb", "rank_bm25", "sentence_transformers", "dotenv"):
        mod = types.ModuleType(name)
        mod.PersistentClient = mod.BM25Okapi = mod.SentenceTransformer = object
        mod.load_dotenv = lambda *a, **k: None
        monkeypatch.setitem(sys.modules, name, mod)

    class Config:
        def __init__(self, **kw):
            self.kw = kw

    gtypes = types.ModuleType("google.genai.types")
    gtypes.GenerateContentConfig = Config
    gtypes.EmbedContentConfig = Config
    genai = types.ModuleType("google.genai")
    genai.types = gtypes
    genai.Client = object
    google = types.ModuleType("google")
    google.genai = genai
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.genai", genai)
    monkeypatch.setitem(sys.modules, "google.genai.types", gtypes)


def fresh_import(monkeypatch, name):
    # Drop any cached / faked copy (monkeypatch puts the original back after the test)
    monkeypatch.delitem(sys.modules, name, raising=False)
    return importlib.import_module(name)


def test_cloud_engine_generate_stream(monkeypatch, stub_heavy_libs):
    rag_cloud = fresh_import(monkeypatch, "app.rag_cloud")
    seen = {}

    class Models:
        def generate_content_stream(self, **kw):
            seen.update(kw)
            return iter([types.SimpleNamespace(text="A"), types.SimpleNamespace(text=None),
                         types.SimpleNamespace(text="B")])

    engine = object.__new__(rag_cloud.CloudRAG)
    engine.client = types.SimpleNamespace(models=Models())
    assert list(engine.generate_stream("q?", [dict(HIT)])) == ["A", "B"]
    # same model / system prompt / temperature as generate()
    assert seen["model"] == rag_cloud.LLM_MODEL
    assert seen["config"].kw == {"system_instruction": rag_cloud.SYSTEM_PROMPT, "temperature": 0.2}
    assert "[1] (2026-01-01) t" in seen["contents"] and "q?" in seen["contents"]
    # no hits -> fixed message, no API call
    seen.clear()
    assert list(engine.generate_stream("q?", [])) == ["관련 논문을 찾지 못했어요. (No matching papers found.)"]
    assert seen == {}


class FakeOllamaResponse:
    def __init__(self, lines):
        self.lines = lines

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def raise_for_status(self):
        pass

    def iter_lines(self):
        return iter(self.lines)


def test_local_engine_generate_stream(monkeypatch, stub_heavy_libs):
    rag = fresh_import(monkeypatch, "app.rag")
    seen = {}
    lines = [
        json.dumps({"message": {"content": "안녕"}, "done": False}).encode(),
        b"",
        json.dumps({"message": {"content": "하세요"}, "done": False}).encode(),
        json.dumps({"message": {"content": ""}, "done": True}).encode(),
        json.dumps({"message": {"content": "after done"}}).encode(),
    ]

    def fake_post(url, json=None, timeout=None, stream=False):
        seen.update(url=url, body=json, stream=stream)
        return FakeOllamaResponse(lines)

    monkeypatch.setattr(rag.requests, "post", fake_post)
    engine = object.__new__(rag.RAG)
    assert list(engine.generate_stream("질문", [dict(HIT)])) == ["안녕", "하세요"]
    assert seen["stream"] is True and seen["body"]["stream"] is True
    assert seen["body"]["messages"][0]["content"] == rag.SYSTEM_PROMPT


def test_local_engine_stream_error_line(monkeypatch, stub_heavy_libs):
    rag = fresh_import(monkeypatch, "app.rag")
    lines = [json.dumps({"message": {"content": "a"}}).encode(), json.dumps({"error": "model crashed"}).encode()]
    monkeypatch.setattr(rag.requests, "post", lambda *a, **k: FakeOllamaResponse(lines))
    gen = object.__new__(rag.RAG).generate_stream("q", [dict(HIT)])
    assert next(gen) == "a"
    with pytest.raises(RuntimeError, match="model crashed"):
        next(gen)
