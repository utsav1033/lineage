"""Fetch arXiv papers as plain text with citation markers kept inline, plus their bibliography."""
import copy
import hashlib
import io
import json
import os
import re
import threading
import time
import warnings
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from urllib.parse import quote, urljoin

import requests
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

CACHE = Path(os.environ.get("LINEAGE_CACHE_DIR", Path(__file__).parent / ".cache"))
CACHE.mkdir(parents=True, exist_ok=True)
UA = {"User-Agent": "PaperLineage/0.1 (research demo for the Beam challenge)"}
ARXIV_ID = re.compile(r"(\d{4}\.\d{4,5}|[a-z\-]+(?:\.[A-Z]{2})?/\d{7})(v\d+)?", re.I)
_last_arxiv_api = 0.0
_arxiv_lock = threading.Lock()


@dataclass
class Paper:
    arxiv_id: str
    title: str
    text: str  # body text; citations appear as [ref:N]
    refs: dict = field(default_factory=dict)  # N -> raw bibliography entry text
    math: dict = field(default_factory=dict)  # LaTeX source -> MathML, for display in the reader
    figures: dict = field(default_factory=dict)  # N -> {kind, imgs, table}; text marks them as [[fig:N]]

    def to_json(self):
        return json.dumps(self.__dict__)


def _cached(key: str, fn):
    path = CACHE / (hashlib.sha1(key.encode()).hexdigest() + ".json")
    if path.exists():
        return json.loads(path.read_text())
    val = fn()
    if val is not None:
        path.write_text(json.dumps(val))
    return val


def parse_arxiv_id(s: str) -> str | None:
    m = ARXIV_ID.search(s)
    return m.group(1) if m else None


def _clean_table(table) -> str:
    """A table's HTML with only layout attributes kept (LaTeXML classes carry alignment and rules)."""
    t = copy.copy(table)
    for el in [t, *t.find_all(True)]:
        el.attrs = {k: v for k, v in el.attrs.items() if k in ("class", "colspan", "rowspan", "display")}
    return str(t)


def _html_to_paper(arxiv_id: str, html: str, base_url: str = "") -> Paper:
    soup = BeautifulSoup(html, "html.parser")
    for junk in soup.select("math annotation, math annotation-xml, .ltx_page_footer, .ltx_authors, nav, header, footer, script, style"):
        junk.decompose()

    refs = {}
    for item in soup.select("li.ltx_bibitem"):
        m = re.search(r"bib(\d+)$", item.get("id", ""))
        if m:
            refs[m.group(1)] = " ".join(item.get_text(" ").split())
        item.decompose()

    # Replace each citation link with a stable [ref:N] marker the model can point at.
    for a in soup.select("cite a.ltx_ref"):
        m = re.search(r"bib\.bib(\d+)", a.get("href", ""))
        if m:
            a.replace_with(f"[ref:{m.group(1)}]")
    # Figures and tables: keep images and table HTML (with MathML) for the reader, captions for the model.
    figures = {}
    tops = [f for f in soup.select("figure.ltx_figure, figure.ltx_table") if not f.find_parent("figure")]
    for n, fig in enumerate(tops):
        kind = "table" if "ltx_table" in fig.get("class", []) else "figure"
        # Figures come as <img> or, for vector graphics, <object data="….svg">. Figures sometimes lay out
        # sub-figures with a table, so only real tables keep their table HTML.
        imgs = [urljoin(base_url, im["src"]) for im in fig.select("img[src]")]
        imgs += [urljoin(base_url, ob["data"]) for ob in fig.select("object[data]")]
        table = fig.select_one("table.ltx_tabular") if kind == "table" or not imgs else None
        figures[str(n)] = {"kind": kind, "imgs": imgs, "table": _clean_table(table) if table else ""}
        fig["data-fig"] = str(n)

    # The model reads math as LaTeX; the reader shows arXiv's own MathML, since LaTeXML's TeX often uses
    # macros (\\mathchoice, \\raisebox...) that browser TeX renderers can't handle.
    mathml = {}
    for math in soup.select("math"):
        tex = math.get("alttext", "")
        for el in [math, *math.find_all(True)]:
            for attr in [a for a in el.attrs if a.lower().startswith("on") or a.lower() in ("href", "style")]:
                del el[attr]
        mathml[tex] = str(math)
        math.replace_with(f"\\[{tex}\\]" if math.get("display") == "block" else f"\\({tex}\\)")

    for fig in soup.select("figure[data-fig]"):
        cap = fig.find("figcaption")
        text = " ".join(cap.get_text(" ").split()) if cap else ""
        rows = [" | ".join(" ".join(c.get_text(" ").split()) for c in tr.find_all(["td", "th"]))
                for tr in fig.select("table.ltx_tabular tr")]
        if rows:
            text += " {{rows: " + " ; ".join(rows) + "}}"
        marker = soup.new_tag("p")
        marker.string = f"[[fig:{fig['data-fig']}]] {text}"
        fig.replace_with(marker)

    title_el = soup.select_one("h1.ltx_title")
    title = " ".join(title_el.get_text(" ").split()) if title_el else arxiv_id
    body = soup.select_one("article") or soup.body
    blocks = []
    for el in body.select("h2, h3, h4, h6.ltx_title_abstract, p, figcaption, table.ltx_equation"):
        t = " ".join(el.get_text(" ").split())
        if t:
            blocks.append(("\n## " + t) if el.name.startswith("h") else t)
    text = re.sub(r"\[\s*((?:\[ref:\d+\][\s,;]*)+)\]", r"\1", "\n".join(dict.fromkeys(blocks)))
    return Paper(arxiv_id, title, text, refs, mathml, figures)


def arxiv_title(arxiv_id: str) -> str | None:
    """The paper's title from arXiv's metadata API (the PDF's first line is often a header or affiliation)."""
    def load():
        _arxiv_wait()
        r = requests.get(f"https://export.arxiv.org/api/query?id_list={arxiv_id}", headers=UA, timeout=30)
        entry = BeautifulSoup(r.text, "html.parser").find("entry")
        t = entry and entry.find("title")
        return {"title": " ".join(t.get_text().split())} if t else {}
    return (_cached(f"meta:{arxiv_id}", load) or {}).get("title")


def _pdf_paragraphs(text: str) -> list[str]:
    """pypdf gives one line per visual line; rebuild paragraphs so the reader shows prose, not a ragged column."""
    lines = [l.strip() for l in text.splitlines()]
    lengths = sorted(len(l) for l in lines if len(l) > 20)
    full = lengths[int(len(lengths) * 0.6)] if lengths else 80  # a typical full-width line
    paras, cur = [], ""
    for line in lines:
        if not line:
            if cur: paras.append(cur); cur = ""
            continue
        cur = cur[:-1] + line if cur.endswith("-") and line[:1].islower() else (cur + " " + line).strip()
        # A short line that ends a sentence usually ends a paragraph.
        if len(line) < 0.7 * full and line[-1:] in ".:?!":
            paras.append(cur); cur = ""
    if cur:
        paras.append(cur)
    return [p for p in paras if len(p) > 1]


def _pdf_to_paper(arxiv_id: str) -> Paper | None:
    from pypdf import PdfReader

    r = requests.get(f"https://arxiv.org/pdf/{arxiv_id}", headers=UA, timeout=60)
    if r.status_code != 200:
        return None
    reader = PdfReader(io.BytesIO(r.content))
    paras = _pdf_paragraphs("\n".join(p.extract_text() or "" for p in reader.pages))
    title = arxiv_title(arxiv_id) or next((p for p in paras if len(p) > 10), arxiv_id)[:200]
    return Paper(arxiv_id, title, "\n".join(paras), {})


def fetch_paper(arxiv_id: str) -> Paper | None:
    def load():
        r = requests.get(f"https://arxiv.org/html/{arxiv_id}", headers=UA, timeout=60)
        p = _html_to_paper(arxiv_id, r.text, r.url) if r.status_code == 200 and "ltx_document" in r.text else None
        # Some "HTML" versions only embed the PDF ("See pages 1-last of …"): fall back to the PDF's text.
        if not p or len(p.text) < 3000:
            p = _pdf_to_paper(arxiv_id) or p
        return p.__dict__ if p else None

    d = _cached(f"paper:{arxiv_id}", load)
    return Paper(**d) if d else None


def _arxiv_wait():
    """arXiv asks API clients for at most one request every 3 seconds, across all threads."""
    global _last_arxiv_api
    with _arxiv_lock:
        wait = 3.1 - (time.time() - _last_arxiv_api)
        if wait > 0:
            time.sleep(wait)
        _last_arxiv_api = time.time()


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


def search_arxiv_by_title(title: str) -> dict | None:
    """Return {'arxiv_id', 'title'} for the best title match, or None if nothing is close enough."""

    def load():
        _arxiv_wait()
        q = quote(f'ti:"{re.sub(r"[^A-Za-z0-9 ]", " ", title)}"')
        r = requests.get(f"https://export.arxiv.org/api/query?search_query={q}&max_results=3", headers=UA, timeout=30)
        soup = BeautifulSoup(r.text, "html.parser")
        best = None
        for entry in soup.find_all("entry"):
            t = " ".join(entry.find("title").get_text().split())
            score = SequenceMatcher(None, _norm(t), _norm(title)).ratio()
            if score > 0.85 and (not best or score > best["score"]):
                best = {"arxiv_id": parse_arxiv_id(entry.find("id").get_text()), "title": t, "score": score}
        return best or {}

    return _cached(f"title:{_norm(title)}", load) or None


def resolve_ref(ref_text: str, title_hint: str | None = None) -> dict | None:
    """Map a bibliography entry to an arXiv paper, via an explicit arXiv id or a title search."""
    if m := re.search(r"arXiv[:\s]*(?:preprint\s*)?(?:arXiv:)?\s*(\d{4}\.\d{4,5})", ref_text, re.I):
        return {"arxiv_id": m.group(1), "title": title_hint or ref_text[:120]}
    if title_hint:
        return search_arxiv_by_title(title_hint)
    return None
