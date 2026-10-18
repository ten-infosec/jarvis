"""Browser-side checks for the book cards in app/static/index.html, run with Node (skipped when Node is missing).

The page script is executed in a sandbox with a fake DOM; no browser, network or private data is involved.
"""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

INDEX = Path(__file__).resolve().parents[1] / "app" / "static" / "index.html"

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")

JS = r"""
const fs = require("fs"), vm = require("vm");
const html = fs.readFileSync(process.argv[1], "utf8");
const script = html.split("<script>")[1].split("</script>")[0];
const el = () => ({ style: {}, value: "", checked: false, disabled: false, textContent: "", innerHTML: "",
  addEventListener() {}, appendChild() {}, focus() {}, requestSubmit() {} });
const sandbox = { document: { getElementById: el, createElement: el, body: { scrollHeight: 0 } },
  window: { scrollTo() {} }, fetch: () => new Promise(() => {}), console };
vm.createContext(sandbox);
vm.runInContext(script, sandbox);
const api = vm.runInContext("({libraryLink, sourcesHtml, renderAnswer, sourceUrl})", sandbox);

const ID = "0123456789abcdef";
const out = {};
out.linkOk = api.libraryLink({ book_id: ID, page: 5 });
out.linkBad = ["0123456789ABCDEF", "0123456789abcde", "../etc/passwd", "javascript:alert(1)", "", null, ID + "/x"]
  .map(id => api.libraryLink({ book_id: id, page: 5 }));
out.pageBad = [0, -1, 1.5, "abc", 100001, null, undefined].map(p => api.libraryLink({ book_id: ID, page: p }));

const book = { n: 1, source: "library", date: "", url: "javascript:alert(1)", via: "bm25", preview: "p <b>x</b>",
  title: "책 <img src=x onerror=alert(1)>", book_id: ID, page: 5, pages: "5-6", ocr: 88.5, ocr_low: true };
out.cardHtml = api.sourcesHtml([book]);
out.textBook = api.sourcesHtml([{ ...book, ocr: undefined, ocr_low: undefined, title: "책A" }]);
out.badIdCard = api.sourcesHtml([{ ...book, book_id: "../x" }]);
out.paper = api.sourcesHtml([{ n: 2, source: "arxiv", date: "2026-01-01", url: "https://arxiv.org/abs/1", via: "vec", preview: "t" }]);
out.paperHttp = api.sourcesHtml([{ n: 2, source: "arxiv", date: "2026-01-01", url: "http://evil.example", via: "vec", preview: "t" }]);

const div = { innerHTML: "" };
api.renderAnswer(div, { answer: "근거 [1] 와 [2]", search_query: "q", timing_sec: { retrieve: 0.1, generate: 0.2 },
  sources: [{ ...book, url: "" }, { n: 2, source: "saved", date: "d", url: "", via: "vec", preview: "t" }] });
out.answerHtml = div.innerHTML;
console.log(JSON.stringify(out));
"""


@pytest.fixture(scope="module")
def result():
    run = subprocess.run(["node", "-e", JS, str(INDEX)], capture_output=True, text=True, encoding="utf-8", timeout=60)
    assert run.returncode == 0, run.stderr
    return json.loads(run.stdout)


def test_page_script_still_loads(result):
    assert result["linkOk"]


def test_book_link_is_built_only_from_valid_parts(result):
    assert result["linkOk"] == "/library/pdf/0123456789abcdef#page=5"
    assert result["linkBad"] == [""] * 7                  # bad ids never become links
    assert result["pageBad"] == [""] * 7                  # bad page numbers never become links


def test_book_card_shows_fields_and_escapes_html(result):
    card = result["cardHtml"]
    assert 'href="/library/pdf/0123456789abcdef#page=5"' in card and "PDF 5쪽 열기" in card
    assert "p.5-6" in card and "OCR 89" in card and "낮은 신뢰" in card
    assert "<img" not in card and "&lt;img" in card         # the title cannot inject HTML
    assert "javascript:" not in card                        # the server's url is ignored for books
    assert "<b>x</b>" not in card and "&lt;b&gt;x&lt;/b&gt;" in card


def test_text_book_has_no_ocr_badge_and_bad_id_has_no_link(result):
    assert "OCR" not in result["textBook"] and "낮은 신뢰" not in result["textBook"]
    assert "href" not in result["badIdCard"]


def test_non_book_sources_keep_the_https_only_rule(result):
    assert 'href="https://arxiv.org/abs/1"' in result["paper"]
    assert "href" not in result["paperHttp"]


def test_citations_link_to_the_pdf_page_for_books_only(result):
    html = result["answerHtml"]
    assert '<a class="cite" href="/library/pdf/0123456789abcdef#page=5"' in html
    assert "[2]" in html and html.count('class="cite"') == 1   # the post without a link stays plain text
