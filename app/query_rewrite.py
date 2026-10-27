# ===== Korean query rewrite for the English arXiv corpus =====
# (a) deterministic Korean -> English glossary expansion (medical_glossary_ko_en.json)
# (b) optional LLM translation hook (Translator protocol; GeminiTranslator for later, fakes in tests)
# (c) weight rule: a mostly-Hangul query gets a lower BM25 weight in the RRF fusion
# No network call happens unless a real translator object is passed in.


import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Protocol


# ===== Settings =====
HERE = Path(__file__).resolve().parent
DEFAULT_GLOSSARY = HERE / "medical_glossary_ko_en.json"

# Same request-word list as app/rag_cloud.py (copied so this module runs without chromadb / gemini)
STOPWORDS = {
    "관련", "관련된", "논문", "최신", "최근", "알려줘", "찾아줘", "보여줘", "정리해줘",
    "뭐", "뭐야", "어떤", "있어", "있는", "좀", "대한", "관한",
    "what", "which", "paper", "papers", "about", "on", "the", "a", "an", "of", "for",
    "in", "is", "are", "show", "me", "find", "tell", "related", "recent", "latest",
}

# Korean particles allowed after a 1-syllable key (뇌를, 폐의, 간에서 ...)
PARTICLES = ("", "은", "는", "이", "가", "을", "를", "의", "에", "에서", "도", "와", "과", "로", "으로", "만")

# Hangul share at or above this = "mostly Hangul" query
HANGUL_THRESHOLD = 0.5
# Default fusion weights (same as rag_cloud.py)
W_VEC_DEFAULT = 1.5
W_BM25_DEFAULT = 1.0
# BM25 weight for a mostly-Hangul query with NO glossary hit: BM25 only sees noise tokens like "AI", "MRI"
W_BM25_HANGUL_NO_MATCH = 0.3
# BM25 weight for a mostly-Hangul query WITH glossary hits: English terms help, but coverage is partial
W_BM25_HANGUL_MATCHED = 0.7


# A Latin token that appears in at most this many papers counts as a rare (name-like) term.
# Fixed in advance at 2% of the 2,000-paper corpus; not tuned on the evaluation questions
RARE_DF = 40


# ===== Document frequency of Latin tokens (used by the rare-term guard) =====
def latin_tokens(text):
    # Same token shape in the corpus and in the question; hyphens stay inside (ICD-10, ChestX-ray14 -> chestx-ray14)
    return re.findall(r"[a-z0-9][a-z0-9-]*", text.lower())


def latin_doc_freq(texts):
    # {token: number of documents containing it}
    freq = {}
    for t in texts:
        for tok in set(latin_tokens(t)):
            freq[tok] = freq.get(tok, 0) + 1
    return freq


def has_rare_latin_term(question, doc_freq, rare_df=RARE_DF):
    # Words of 3+ letters starting with a letter (skips "AI"-style 2-letter tokens and bare numbers)
    words = re.findall(r"[A-Za-z][A-Za-z0-9-]{2,}", question)
    return any(doc_freq.get(w.lower(), 0) <= rare_df for w in words)


# ===== Language helpers =====
def hangul_ratio(text):
    # Count only letters: Hangul syllables vs Latin letters (digits, spaces, punctuation ignored)
    n_ko = len(re.findall(r"[가-힣]", text))
    n_en = len(re.findall(r"[A-Za-z]", text))
    # No letters at all -> treat as not Hangul
    return n_ko / (n_ko + n_en) if (n_ko + n_en) else 0.0


def is_mostly_hangul(text, threshold=HANGUL_THRESHOLD):
    return hangul_ratio(text) >= threshold


def clean_query(question):
    # Same behavior as rag_cloud.clean_query: drop request words, keep content words
    words = re.findall(r"[가-힣]+|[a-zA-Z0-9]+", question)
    kept = [w for w in words if w.lower() not in STOPWORDS]
    return " ".join(kept) if kept else question


# ===== (a) Glossary =====
class Glossary:
    def __init__(self, terms):
        # terms: {"폐결핵": ["pulmonary tuberculosis", "tuberculosis"], ...}
        self.terms = {k: list(v) for k, v in terms.items() if k}
        # Multi-syllable keys are matched as substrings, longest first (폐결핵 before 결핵)
        self.multi = sorted([k for k in self.terms if len(k) > 1], key=len, reverse=True)
        # 1-syllable keys (뇌, 폐, 간, 암 ...) are matched only as whole words (+ particle) to avoid 시간 -> liver
        self.single = {k for k in self.terms if len(k) == 1}

    @classmethod
    def load(cls, path=DEFAULT_GLOSSARY):
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        # File format: {"_meta": {...}, "terms": {key: {"en": [...], "cat": "..."}}}
        terms = raw.get("terms", raw)
        return cls({k: (v["en"] if isinstance(v, dict) else v) for k, v in terms.items()})

    def match(self, text):
        # Returns [(korean_key, [english, ...]), ...] in order of appearance
        found = []
        # Join Hangul words split by spaces so "당뇨 망막병증" also hits "당뇨망막병증"
        joined = re.sub(r"(?<=[가-힣])\s+(?=[가-힣])", "", text)
        # Mutable copy: matched spans are blanked so shorter keys inside them are not matched again
        buf = list(joined)
        for key in self.multi:
            start = 0
            while True:
                i = "".join(buf).find(key, start)
                if i < 0:
                    break
                found.append((i, key))
                # Blank the span with a non-Hangul char (keeps positions stable)
                for j in range(i, i + len(key)):
                    buf[j] = "\0"
                start = i + len(key)
        # 1-syllable keys: check each ORIGINAL space-separated Hangul word = key + optional particle
        # (the joined string glues words together, so "간을 분할" would never look like a word there)
        for m in re.finditer(r"[가-힣]+", text):
            word = m.group()
            for key in self.single:
                if word.startswith(key) and word[len(key):] in PARTICLES:
                    # Placed after all multi-syllable hits so positions from the two strings never mix
                    found.append((len(joined) + m.start(), key))
        found.sort()
        return [(k, self.terms[k]) for _, k in found]

    def expand(self, text):
        # English terms for every matched key, de-duplicated, in order
        out = []
        for _, en_list in self.match(text):
            for en in en_list:
                if en not in out:
                    out.append(en)
        return out


# ===== (b) LLM translation hook =====
class Translator(Protocol):
    # Any object with translate(text) -> English text can be plugged in
    def translate(self, text: str) -> str: ...


class GeminiTranslator:
    # Real translator for later (NOT used in tests / overnight runs). Needs GEMINI_API_KEY.
    PROMPT = (
        "Translate this medical literature search query into concise English search keywords. "
        "Use standard medical terminology (e.g. 폐결핵 -> pulmonary tuberculosis). "
        "Output only the English query, no explanation.\n\nQuery: {q}"
    )

    def __init__(self, client=None, model="gemini-flash-latest"):
        # Lazy import: importing this module never requires google-genai
        if client is None:
            import os
            from google import genai
            client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        self.client = client
        self.model = model
        # In-memory cache: the same question never costs a second API call
        self.cache = {}

    def translate(self, text):
        if text in self.cache:
            return self.cache[text]
        from google.genai import types
        out = self.client.models.generate_content(
            model=self.model,
            contents=self.PROMPT.format(q=text),
            # temperature 0: same input -> same translation (stable retrieval)
            config=types.GenerateContentConfig(temperature=0.0),
        )
        result = (out.text or "").strip()
        self.cache[text] = result
        return result


# ===== (c) Weight rule =====
def bm25_weight(question, n_matched, translated=False, base=W_BM25_DEFAULT):
    # English (or mixed) query, or a successful LLM translation -> BM25 keeps full weight
    if translated or not is_mostly_hangul(question):
        return base
    # Mostly Hangul: lower BM25 weight; lowest when the glossary found nothing
    return W_BM25_HANGUL_MATCHED if n_matched else W_BM25_HANGUL_NO_MATCH


# ===== Rewrite entry point =====
@dataclass
class Rewrite:
    original: str
    # Query for BM25 (original content words + English expansions)
    bm25_query: str
    # Query for the embedding model (multilingual -> original cleaned question by default)
    vector_query: str
    hangul_ratio: float
    matched: list = field(default_factory=list)
    expansions: list = field(default_factory=list)
    translation: Optional[str] = None
    w_bm25: float = W_BM25_DEFAULT
    w_vec: float = W_VEC_DEFAULT


def rewrite_query(question, glossary=None, translator=None, use_glossary=True, use_weight_rule=True,
                  doc_freq=None, rare_df=RARE_DF):
    base = clean_query(question)
    ratio = hangul_ratio(question)
    matched, expansions, translation = [], [], None

    # Only rewrite queries that contain Hangul; pure English queries pass through unchanged
    has_hangul = bool(re.search(r"[가-힣]", question))
    # A rare English term (CheXpert, TotalSegmentator ...) means the user is searching by name:
    # glossary words would dilute that keyword and the lowered BM25 weight would hide it -> leave the query alone
    if has_hangul and doc_freq is not None and has_rare_latin_term(question, doc_freq, rare_df):
        return Rewrite(original=question, bm25_query=base, vector_query=base, hangul_ratio=round(ratio, 3))
    if has_hangul and use_glossary and glossary is not None:
        pairs = glossary.match(question)
        matched = [k for k, _ in pairs]
        expansions = glossary.expand(question)

    # Optional LLM translation; any failure (quota, network) falls back to glossary-only
    if has_hangul and translator is not None:
        try:
            translation = (translator.translate(question) or "").strip() or None
        except Exception as e:
            print(f"[warn] translation failed -> glossary only ({type(e).__name__}: {str(e)[:200]})")
            translation = None

    # BM25 sees: original content words + translation + glossary terms (stopwords removed again)
    parts = [base] + ([translation] if translation else []) + expansions
    bm25_query = clean_query(" ".join(parts))

    w = bm25_weight(question, len(matched), translated=bool(translation)) if use_weight_rule else W_BM25_DEFAULT
    return Rewrite(original=question, bm25_query=bm25_query, vector_query=base, hangul_ratio=round(ratio, 3),
                   matched=matched, expansions=expansions, translation=translation, w_bm25=w)


# → 한국어 질문을 영어 논문 코퍼스에서 찾을 수 있게 다시 쓰겠다.
#   (a) 용어집(JSON)에서 한국어 의학·AI 용어를 가장 긴 것부터 찾아 영어 용어를 BM25 질의에 덧붙이고,
#   (b) 나중에 Gemini 번역을 꽂을 수 있게 translate(text) 인터페이스만 열어 두며(실패하면 용어집만 사용),
#   (c) 한글 위주 질문은 RRF에서 BM25 가중치를 낮춘다(용어집 일치 0개면 0.3, 있으면 0.7).
#   벡터 검색용 질의는 원문을 그대로 쓴다(Gemini 임베딩은 다국어라서).
