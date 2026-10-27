# ===== Tests for app/query_rewrite.py and its use inside CloudRAG.retrieve: no network, no keys =====
import importlib
import json
import sys
import types
from collections import defaultdict

import pytest

from app.query_rewrite import (DEFAULT_GLOSSARY, RARE_DF, W_BM25_DEFAULT, W_BM25_HANGUL_MATCHED,
                               W_BM25_HANGUL_NO_MATCH, Glossary, bm25_weight, hangul_ratio, has_rare_latin_term,
                               latin_doc_freq, rewrite_query)


# Small in-memory glossary so most tests do not depend on the full JSON contents
@pytest.fixture
def g():
    return Glossary({"폐결핵": ["pulmonary tuberculosis", "tuberculosis"], "결핵": ["tuberculosis"],
                     "당뇨망막병증": ["diabetic retinopathy"], "당뇨": ["diabetes"], "망막": ["retina"],
                     "간": ["liver"], "뇌": ["brain"], "요추": ["lumbar spine"]})


class FakeTranslator:
    def __init__(self, text="lumbar spine MRI AI"):
        self.text, self.calls = text, 0

    def translate(self, text):
        self.calls += 1
        return self.text


class BrokenTranslator:
    def translate(self, text):
        raise RuntimeError("429 quota")


# ===== 1. Glossary =====
def test_full_glossary_file_loads_and_is_large():
    full = Glossary.load()
    # 300+ general terms, every value is a non-empty English list
    assert len(full.terms) >= 300
    assert all(v and all(isinstance(x, str) and x for x in v) for v in full.terms.values())
    assert "tuberculosis" in full.expand("폐결핵 검진")


def test_longest_match_wins_and_blocks_inner_keys(g):
    # 당뇨망막병증 must not also add "diabetes" / "retina"
    assert g.expand("당뇨망막병증 등급 분류") == ["diabetic retinopathy"]
    assert [k for k, _ in g.match("폐결핵을 찾아줘")] == ["폐결핵"]


def test_spaced_compound_is_joined(g):
    assert g.expand("당뇨 망막병증") == ["diabetic retinopathy"]


def test_single_syllable_only_as_whole_word(g):
    # 시간 contains 간 but must NOT become "liver"; 간을 / 뇌 should match
    assert g.expand("시간이 오래 걸리는 분석") == []
    assert g.expand("간을 분할하는 모델") == ["liver"]
    assert g.expand("뇌 MRI") == ["brain"]


def test_glossary_json_has_no_eval_answer_ids():
    # The glossary must not contain arXiv ids (that would mean it was built from the evaluation answers)
    raw = DEFAULT_GLOSSARY.read_text(encoding="utf-8")
    assert "arxiv:" not in raw and "26" not in json.loads(raw)["_meta"].get("source_ids", "")


# ===== 2. Rewrite and weight rule =====
def test_hangul_ratio():
    assert hangul_ratio("요추 MRI") == pytest.approx(2 / 5)
    assert hangul_ratio("lumbar MRI") == 0.0
    assert hangul_ratio("1234 !!") == 0.0


def test_english_query_unchanged(g):
    r = rewrite_query("papers about AI for tuberculosis screening", g)
    assert r.bm25_query == "AI tuberculosis screening"
    assert r.expansions == [] and r.w_bm25 == W_BM25_DEFAULT


def test_korean_query_expanded_and_weight_lowered(g):
    r = rewrite_query("폐결핵 검진을 인공지능으로 하는 연구 있어?", g)
    assert "tuberculosis" in r.bm25_query
    # The vector query keeps the original words
    assert "폐결핵" in r.vector_query and "tuberculosis" not in r.vector_query
    assert r.w_bm25 == W_BM25_HANGUL_MATCHED


def test_weight_rule_values():
    assert bm25_weight("모기가 옮기는 병", n_matched=0) == W_BM25_HANGUL_NO_MATCH
    assert bm25_weight("모기가 옮기는 병", n_matched=2) == W_BM25_HANGUL_MATCHED
    assert bm25_weight("mosquito disease", n_matched=0) == W_BM25_DEFAULT
    assert bm25_weight("모기가 옮기는 병", n_matched=0, translated=True) == W_BM25_DEFAULT
    assert W_BM25_HANGUL_NO_MATCH < W_BM25_HANGUL_MATCHED < W_BM25_DEFAULT


def test_weight_rule_can_be_disabled(g):
    assert rewrite_query("요추 분석", g, use_weight_rule=False).w_bm25 == W_BM25_DEFAULT


def test_translator_hook_used_for_korean_only(g):
    t = FakeTranslator()
    r = rewrite_query("요추 MRI 분석하는 AI 논문 있어?", g, translator=t)
    assert r.translation == "lumbar spine MRI AI"
    assert "lumbar" in r.bm25_query and r.w_bm25 == W_BM25_DEFAULT
    rewrite_query("lumbar spine MRI", g, translator=t)
    # An English query never calls the translator
    assert t.calls == 1


def test_translator_failure_falls_back_to_glossary(g, capsys):
    r = rewrite_query("요추 MRI", g, translator=BrokenTranslator())
    assert r.translation is None and "lumbar spine" in r.bm25_query
    assert "translation failed" in capsys.readouterr().out


# ===== 3. Rare English term guard =====
def freq():
    # 100 papers: "mri" is common, "chexpert" appears in 5, "xyzterm" in none
    return latin_doc_freq(["mri scan chexpert"] * 5 + ["mri scan lumbar"] * 95)


def test_latin_doc_freq_counts_documents_not_occurrences():
    f = latin_doc_freq(["mri mri mri", "mri", "icd-10 code"])
    assert f["mri"] == 2 and f["icd-10"] == 1


def test_has_rare_latin_term():
    f = freq()
    assert RARE_DF == 40
    assert has_rare_latin_term("CheXpert 흉부 엑스레이", f)          # df 5 <= 40
    assert has_rare_latin_term("XYZterm 모델", f)                    # not in the corpus (df 0)
    assert not has_rare_latin_term("요추 MRI 분석", f)               # df 100
    assert not has_rare_latin_term("AI 진단", f)                     # 2-letter tokens are skipped
    assert not has_rare_latin_term("폐결핵 검진", f)                 # no Latin word at all


def test_rare_term_skips_the_rewrite(g):
    f = freq()
    r = rewrite_query("CheXpert 폐결핵 논문", g, doc_freq=f)
    # Nothing is added and the BM25 weight stays at the default: the user is searching by name
    assert r.expansions == [] and r.w_bm25 == W_BM25_DEFAULT
    assert r.bm25_query == r.vector_query == "CheXpert 폐결핵"
    # Same question without a rare term is rewritten as usual
    r2 = rewrite_query("MRI 폐결핵 논문", g, doc_freq=f)
    assert "tuberculosis" in r2.bm25_query and r2.w_bm25 == W_BM25_HANGUL_MATCHED


def test_doc_freq_is_optional(g):
    # Without document frequencies the guard is off (keeps the module usable on its own)
    r = rewrite_query("CheXpert 폐결핵 논문", g)
    assert "tuberculosis" in r.bm25_query


# ===== 4. Inside CloudRAG.retrieve (heavy libraries stubbed, fake search backends) =====
@pytest.fixture
def cloud(monkeypatch):
    for name in ("chromadb", "rank_bm25", "dotenv"):
        mod = types.ModuleType(name)
        mod.PersistentClient = mod.BM25Okapi = object
        mod.load_dotenv = lambda *a, **k: None
        monkeypatch.setitem(sys.modules, name, mod)

    class Config:
        def __init__(self, **kw):
            self.kw = kw

    gtypes = types.ModuleType("google.genai.types")
    gtypes.GenerateContentConfig = gtypes.EmbedContentConfig = Config
    genai = types.ModuleType("google.genai")
    genai.types, genai.Client = gtypes, object
    google = types.ModuleType("google")
    google.genai = genai
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.genai", genai)
    monkeypatch.setitem(sys.modules, "google.genai.types", gtypes)
    monkeypatch.delitem(sys.modules, "app.rag_cloud", raising=False)
    return importlib.import_module("app.rag_cloud")


def make_engine(mod, vec_ids):
    ids = ["d1", "d2", "d3"]
    docs = {i: {"id": i, "date": "2026-01-01", "text": f"text of {i}", "meta": {"url": f"u/{i}"}} for i in ids}
    engine = object.__new__(mod.CloudRAG)
    engine.docs, engine.ids = docs, ids
    engine.glossary = Glossary({"결핵": ["tuberculosis"]})
    engine.doc_freq = defaultdict(int, {"chexpert": 3})
    seen = {"embedded": [], "bm25": []}

    class Bm25:
        # Scores the document whose id appears in the query tokens highest
        def get_scores(self, tokens):
            seen["bm25"].append(list(tokens))
            return [1.0 if ("tuberculosis" in tokens and i == "d3") else 0.0 for i in ids]

    class Col:
        def query(self, query_embeddings, n_results):
            return {"ids": [vec_ids]}

    engine.bm25, engine.col = Bm25(), Col()
    engine._embed_query = lambda q: seen["embedded"].append(q) or [0.0]
    return engine, seen


def test_retrieve_rewrites_only_the_bm25_side(cloud, monkeypatch):
    monkeypatch.setattr(cloud, "USE_REWRITE", True)
    engine, seen = make_engine(cloud, ["d1", "d2"])
    query, hits = engine.retrieve("결핵 논문", top_n=3)
    # Vector side: cleaned original question. BM25 side: glossary term added
    assert seen["embedded"] == ["결핵"]
    assert "tuberculosis" in seen["bm25"][0]
    assert query == "결핵 tuberculosis"
    assert "d3" in [h["doc_id"] for h in hits]


def test_retrieve_with_rewrite_switched_off(cloud, monkeypatch):
    monkeypatch.setattr(cloud, "USE_REWRITE", False)
    engine, seen = make_engine(cloud, ["d1", "d2"])
    query, hits = engine.retrieve("결핵 논문", top_n=3)
    assert query == "결핵" and "tuberculosis" not in seen["bm25"][0]
    assert [h["doc_id"] for h in hits] == ["d1", "d2"]      # BM25 found nothing, vectors only


def test_retrieve_english_question_unchanged(cloud, monkeypatch):
    monkeypatch.setattr(cloud, "USE_REWRITE", True)
    engine, seen = make_engine(cloud, ["d1"])
    query, _ = engine.retrieve("papers about tuberculosis", top_n=3)
    assert query == "tuberculosis" and seen["embedded"] == ["tuberculosis"]


def test_retrieve_rare_term_question_unchanged(cloud, monkeypatch):
    monkeypatch.setattr(cloud, "USE_REWRITE", True)
    engine, seen = make_engine(cloud, ["d1"])
    query, _ = engine.retrieve("CheXpert 결핵 논문", top_n=3)
    assert query == "CheXpert 결핵" and "tuberculosis" not in seen["bm25"][0]
