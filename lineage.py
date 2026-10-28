"""Trace where an idea in a paper came from, hop by hop through its citations, using Beam."""
import json
import os
import re
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from huggingface_hub import get_token
from openai import OpenAI

from papers import Paper, _cached, fetch_paper, resolve_ref

MODEL = "reflection/Beam-1.0-501B-A23B:baseten"
PRICE_IN, PRICE_OUT = 0.40 / 1e6, 1.20 / 1e6  # Baseten, $ per token
MAX_PAPER_CHARS = 400_000

TOKEN = get_token()  # on a Space: the HF_TOKEN secret
client = OpenAI(base_url="https://router.huggingface.co/v1", api_key=TOKEN or "missing")
spent = {"usd": 0.0, "calls": 0}

# A public deployment bills every visitor's request to the owner's token, so spending is capped per day.
DAILY_BUDGET = float(os.environ.get("LINEAGE_DAILY_BUDGET_USD", "3"))
_today = {"day": time.strftime("%Y-%m-%d"), "usd": 0.0}
_budget_lock = threading.Lock()


class BudgetError(RuntimeError):
    pass


def _complete(**kw):
    """Every Beam call goes through here: refuse when unconfigured or over today's budget, and record the cost."""
    if not TOKEN:
        raise BudgetError("This deployment has no HF_TOKEN secret set, so it can't reach Beam yet.")
    with _budget_lock:
        day = time.strftime("%Y-%m-%d")
        if _today["day"] != day:
            _today.update(day=day, usd=0.0)
        if _today["usd"] >= DAILY_BUDGET:
            raise BudgetError("Today's Beam budget for this demo is used up. Please come back tomorrow!")
    r = client.chat.completions.create(model=MODEL, **kw)
    usd = r.usage.prompt_tokens * PRICE_IN + r.usage.completion_tokens * PRICE_OUT
    with _budget_lock:
        spent["calls"] += 1
        spent["usd"] += usd
        _today["usd"] += usd
    return r

NODE_PROMPT = """You are tracing the intellectual origin of an idea through the scientific literature.

The idea being traced: {concept}
{parent_context}
Below is the full text of the paper "{title}". Citations appear inline as [ref:N]; the bibliography follows.

Return ONLY a JSON object with these keys:
- "found": true if this paper discusses the idea at all.
- "quote": the single most relevant passage, copied VERBATIM from the paper text (1-3 sentences, keep [ref:N] markers). Empty string if not found.
- "role": one of "introduces" (this paper presents the idea as its own new contribution), "uses" (applies it, attributing it to others), "related" (discusses something close but not the same idea), "absent".
- "supports_parent": {support_question}
- "explanation": 2-3 plain-language sentences on what this paper contributes to the idea.
- "sources": up to {max_sources} references this paper credits for THIS idea (not other ideas), as a list of {{"ref": N, "title": "<title of the cited work, from the bibliography>", "why": "<what this paper credits it for>"}}. Empty if the paper introduces the idea without crediting anyone.

=== PAPER TEXT ===
{text}

=== BIBLIOGRAPHY ===
{bib}
"""

SUPPORT_Q_ROOT = 'null (this is the starting paper).'
SUPPORT_Q_CHILD = ('one of "yes" (this paper genuinely contains what the citing paper credits it for), '
                   '"partial" (related but the citing paper overstates or reframes it), "no" (it does not contain it). '
                   'Judge only the specific credit, not how the citing paper later combines or applies it.')


@dataclass
class Node:
    arxiv_id: str
    title: str
    year: str = ""
    found: bool = False
    quote: str = ""
    quote_verified: bool = False
    role: str = ""
    supports_parent: str | None = None
    credited_for: str = ""  # what the parent paper credited this paper for
    explanation: str = ""
    error: str = ""
    children: list = field(default_factory=list)
    unresolved: list = field(default_factory=list)  # cited sources we couldn't fetch (books, non-arXiv)

    def to_dict(self):
        d = dict(self.__dict__)
        d["children"] = [c.to_dict() for c in self.children]
        return d


def _ask_json(prompt: str, attempts: int = 2, max_tokens: int = 4000) -> dict:
    for attempt in range(attempts):
        r = _complete(messages=[{"role": "user", "content": prompt}],
                      response_format={"type": "json_object"}, temperature=0.3, max_tokens=max_tokens)
        out = r.choices[0].message.content or ""
        m = re.search(r"\{.*\}", out, re.S)
        try:
            return json.loads(m.group(0) if m else out)
        except json.JSONDecodeError:
            if attempt == attempts - 1:
                raise


EXPLAIN_PROMPT = """You are a quiet reading companion for a research paper.

The reader highlighted: "{selection}"
In this passage: "{passage}"

Explain what the highlighted part means in THIS paper, in 2-4 plain sentences. Define jargon briefly, say why it
matters here, and mention what it builds on if the paper says so. No preamble, no headings, no bullet points.
Write math as \\( ... \\).

=== PAPER: {title} ===
{text}
"""


def explain(paper: Paper, selection: str, passage: str) -> str:
    r = _complete(temperature=0.3, max_tokens=2000,
                  messages=[{"role": "user", "content": EXPLAIN_PROMPT.format(
                      selection=selection, passage=passage, title=paper.title, text=paper.text[:MAX_PAPER_CHARS])}])
    return (r.choices[0].message.content or "").strip()


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"\[ref:\d+\]", "", s)).lower().strip(" .")


def verify_quote(quote: str, text: str) -> bool:
    """Check the model's 'verbatim' quote really is in the paper, so we never show a made-up citation."""
    q, t = _norm(quote), _norm(text)
    if not q:
        return False
    if q in t:
        return True
    # Tolerate small extraction differences: require most 8-word shingles to appear.
    words = q.split()
    shingles = [" ".join(words[i:i + 8]) for i in range(0, max(1, len(words) - 7), 4)]
    return sum(s in t for s in shingles) / len(shingles) >= 0.7


def _year(ref_text: str) -> str:
    years = re.findall(r"\b(19[5-9]\d|20[0-3]\d)\b", ref_text)
    return years[-1] if years else ""


def analyze(paper: Paper, concept: str, parent: Node | None, credited_for: str, max_sources: int) -> dict:
    parent_context = ""
    if parent:
        parent_context = (f'\nThe paper "{parent.title}" cites this paper for: "{credited_for}". '
                          "Check whether that credit is accurate.\n")
    bib = "\n".join(f"[ref:{k}] {v}" for k, v in paper.refs.items()) or "(bibliography not available)"
    return _ask_json(NODE_PROMPT.format(
        concept=concept, parent_context=parent_context, title=paper.title,
        support_question=SUPPORT_Q_CHILD if parent else SUPPORT_Q_ROOT,
        max_sources=max_sources, text=paper.text[:MAX_PAPER_CHARS], bib=bib,
    ))


def trace(arxiv_id: str, concept: str, depth: int = 2, max_sources: int = 3,
          parent: Node | None = None, credited_for: str = "", year: str = "", seen: set | None = None,
          on_progress=None) -> Node:
    seen = seen if seen is not None else set()
    seen.add(arxiv_id)
    paper = fetch_paper(arxiv_id)
    node = Node(arxiv_id, paper.title if paper else arxiv_id, year=year, credited_for=credited_for)
    if not paper:
        node.error = "Could not fetch paper text"
        return node
    if on_progress:
        on_progress(f"Reading {paper.title}")

    try:
        res = analyze(paper, concept, parent, credited_for, max_sources)
    except Exception as e:  # noqa: BLE001 - surface any model/parse failure on the node itself
        node.error = f"Analysis failed: {e}"
        return node

    node.found = bool(res.get("found"))
    node.quote = res.get("quote") or ""
    node.quote_verified = verify_quote(node.quote, paper.text)
    node.role = res.get("role") or ""
    node.supports_parent = res.get("supports_parent")
    node.explanation = res.get("explanation") or ""

    if depth <= 0 or node.role == "introduces":
        return node

    jobs = []
    for src in (res.get("sources") or [])[:max_sources]:
        ref_text = paper.refs.get(str(src.get("ref")), "")
        hit = resolve_ref(ref_text, src.get("title"))
        label = {"title": src.get("title") or ref_text[:120], "why": src.get("why", ""), "year": _year(ref_text)}
        if not hit or not hit.get("arxiv_id"):
            node.unresolved.append(label)
        elif hit["arxiv_id"] not in seen:
            seen.add(hit["arxiv_id"])
            jobs.append((hit["arxiv_id"], label))

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(trace, aid, concept, depth - 1, max_sources, node, lab["why"], lab["year"], seen,
                               on_progress) for aid, lab in jobs]
        node.children = [f.result() for f in futures]
    return node


TITLES_PROMPT = """For each bibliography entry below, extract the title of the cited work.
Return ONLY a JSON object mapping the entry number to its title, like {{"1": "Layer normalization"}}.

{bib}
"""

SYNTH_PROMPT = """A reader of the paper "{title}" highlighted: "{selection}"
It appears in this passage: "{passage}"

Below are the full texts of works that paper cites for this idea, labelled [S1], [S2], ...
Using ONLY these sources, write a short answer explaining what the highlighted idea is and where it comes from:
what each source contributed, in historical order where possible. 2-3 short paragraphs, under 200 words.

Return ONLY a JSON object:
{{"paragraphs": [[{{"text": "<a sentence or clause of the answer>",
                    "cite": [{{"s": <source number>, "quote": "<1-2 sentences copied VERBATIM from that source>"}}]}}]],
  "gap": "<one sentence on what these sources do NOT explain about the idea, or empty string>"}}
"paragraphs" is a list of paragraphs; each paragraph is a list of segments that read naturally when joined.
Every segment that states a fact must cite at least one source with a verbatim supporting quote.
Cite only [S#] sources, never the highlighted paper itself. In the text, name works the way a paper would
("Bahdanau et al.", "the GNMT system"), never by their S-labels; the labels belong only in "cite".

{sources}
"""
MAX_SOURCE_CHARS = 150_000  # per source: whole papers; ~10 of them stay well inside Beam's 1M-token window


def ref_titles(paper: Paper) -> dict:
    bib = "\n".join(f"[{k}] {v}" for k, v in paper.refs.items())
    return _cached(f"titles:{paper.arxiv_id}", lambda: _ask_json(TITLES_PROMPT.format(bib=bib))) if bib else {}


def cited_near(paper: Paper, selection: str, block: int, limit: int = 10) -> list[str]:
    """References cited in the highlighted passage first, then wherever else the phrase appears."""
    blocks = paper.text.split("\n")
    counts = Counter()
    for i in range(max(0, block - 1), min(len(blocks), block + 2)):
        counts.update({n: 3 if i == block else 1 for n in re.findall(r"\[ref:(\d+)\]", blocks[i])})
    key = selection.lower().strip()[:60]
    for b in blocks:
        if key and key in b.lower():
            counts.update(re.findall(r"\[ref:(\d+)\]", b))
    return [n for n, _ in counts.most_common(limit)]


def synthesize(paper: Paper, selection: str, block: int, on_progress=None) -> dict:
    progress = on_progress or (lambda m: None)
    blocks = paper.text.split("\n")
    passage = blocks[block] if 0 <= block < len(blocks) else ""
    nums = cited_near(paper, selection, block)
    if not nums:
        return {"error": "This passage doesn't cite anything to answer from."}

    progress("Reading the bibliography")
    titles = ref_titles(paper)
    progress(f"Fetching {len(nums)} cited works")

    def load(n):
        ref = paper.refs.get(n, "")
        hit = resolve_ref(ref, titles.get(n))
        src = fetch_paper(hit["arxiv_id"]) if hit and hit.get("arxiv_id") else None
        return n, ref, src

    with ThreadPoolExecutor(max_workers=4) as pool:
        loaded = list(pool.map(load, nums))
    readable = lambda src: src and len(src.text) > 2000  # PDF extraction can come back (nearly) empty
    sources = [(n, ref, src) for n, ref, src in loaded if readable(src)]
    unresolved = [{"ref": n, "title": titles.get(n) or ref[:120], "year": _year(ref)}
                  for n, ref, src in loaded if not readable(src)]
    if not sources:
        return {"error": "None of the works cited here have full text on arXiv.", "unresolved": unresolved}

    progress(f"Reading {len(sources)} papers at once")
    body = "\n\n".join(f"=== [S{i}] {ref} ===\n{src.text[:MAX_SOURCE_CHARS]}"
                        for i, (n, ref, src) in enumerate(sources, 1))
    res = _ask_json(SYNTH_PROMPT.format(title=paper.title, selection=selection, passage=passage, sources=body),
                    max_tokens=24_000)  # reading many papers at once needs room to reason before answering

    paragraphs = []
    for para in res.get("paragraphs") or []:
        segs = []
        for seg in para if isinstance(para, list) else []:
            cites = []
            for c in seg.get("cite") or []:
                i = int(c.get("s", 0)) if str(c.get("s", "")).isdigit() else 0
                if 1 <= i <= len(sources):
                    quote = c.get("quote") or ""
                    cites.append({"s": i, "quote": quote, "verified": verify_quote(quote, sources[i - 1][2].text)})
            segs.append({"text": seg.get("text") or "", "cite": cites})
        paragraphs.append(segs)
    return {
        "paragraphs": paragraphs,
        "gap": res.get("gap") or "",
        "sources": [{"s": i, "ref": n, "arxiv_id": src.arxiv_id, "title": titles.get(n) or src.title, "year": _year(ref)}
                    for i, (n, ref, src) in enumerate(sources, 1)],
        "unresolved": unresolved,
    }


def _clip(s, n: int) -> str:
    s = " ".join(str(s or "").split())
    return s if len(s) <= n else s[: n - 1] + "…"


_TEX_UNICODE = {r"\\top": "ᵀ", r"\\sum": "Σ", r"\\sqrt": "√", r"\\infty": "∞", r"\\approx": "≈", r"\\neq": "≠",
                r"\\leq": "≤", r"\\geq": "≥", r"\\cdot": "·", r"\\times": "×", r"\\to": "→", r"\\alpha": "α",
                r"\\beta": "β", r"\\gamma": "γ", r"\\lambda": "λ", r"\\sigma": "σ", r"\\theta": "θ", r"\\epsilon": "ε"}


_SUP = dict(zip("0123456789+-=()abcdefghijklmnoprstuvwxyzT", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ᵃᵇᶜᵈᵉᶠᵍʰⁱʲᵏˡᵐⁿᵒᵖʳˢᵗᵘᵛʷˣʸᶻᵀ"))
_SUB = dict(zip("0123456789+-=()aehijklmnoprstuvx", "₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎ₐₑₕᵢⱼₖₗₘₙₒₚᵣₛₜᵤᵥₓ"))


def _scripts(s: str) -> str:
    """x^{2}, e^-m, z_j → x², e⁻ᵐ, zⱼ when every character has a Unicode super/subscript form."""
    def conv(table):
        def f(m):
            body = m.group(1) or m.group(2)
            return "".join(table[c] for c in body) if all(c in table for c in body) else m.group(0)
        return f
    s = re.sub(r"\^(?:\{([^}]*)\}|(-?[A-Za-z0-9]+))", conv(_SUP), s)
    return re.sub(r"_(?:\{([^}]*)\}|([A-Za-z0-9])(?![A-Za-z]))", conv(_SUB), s)  # z_max stays: "ax" has no forms


def _plain(s, n: int) -> str:
    """SVG text can't typeset: turn stray LaTeX into readable Unicode-ish text."""
    s = re.sub(r"\\[()\[\]]", "", str(s or ""))
    for tex, uni in _TEX_UNICODE.items():
        s = re.sub(tex + r"(?![a-zA-Z])", uni, s)
    s = re.sub(r"\\tilde\s*\{?(\w)\}?", "\\1\u0303", s)
    s = _scripts(s)
    s = re.sub(r"\\(?:mathrm|mathbf|text|operatorname|mathcal|mathit)\s*\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\[a-zA-Z]+", "", s)
    # Drop grouping braces, but keep ^{…}/_{…} groups so the page can raise or lower them.
    s = re.sub(r"([\^_])\{([^{}]*)\}", "\\1\x01\\2\x02", s).replace("{", "").replace("}", "")
    s = s.replace("\x01", "{").replace("\x02", "}").replace("^ᵀ", "ᵀ")
    return _clip(s, n)


def _num(v):
    try:
        f = float(v)
        return f if f == f and abs(f) != float("inf") else None
    except (TypeError, ValueError):
        return None


SKILLS_DIR = Path(__file__).parent / "skills"
CANVAS_API = (SKILLS_DIR / "canvas-api.md").read_text()
SKILLS = {kind: (SKILLS_DIR / f"{name}.md").read_text() + "\n\n" + CANVAS_API
          for kind, name in (("play", "toy"), ("sketch", "sketch"))}

PLAY_PROMPT = """{skill}

---

The reader of "{title}" highlighted: "{selection}"
In this passage: "{passage}"

{task}
Return ONLY the JSON object.

=== PAPER ===
{text}
"""

FIX_PROMPT = """{skill}

---

The reader of "{title}" highlighted: "{selection}"
In this passage: "{passage}"

You wrote this, but its automatic test failed:

FAILURE: {error}

What you returned:
{spec}

Fix the cause (not the symptom) and return the complete corrected JSON object, in the same format. Keep what was
working. Return ONLY the JSON object.
"""

TOY_TYPES = {"slider", "select", "toggle", "time"}
IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,23}$")


def _clean_toy(d: dict) -> dict:
    """Validate a toy spec's shape. Whether the code actually works is checked by running it in the browser."""
    code = str(d.get("code") or "")
    if "function draw" not in code:
        raise ValueError("The toy has no draw() function.")
    if len(code) > 12_000:
        raise ValueError("The toy's code is too long.")
    controls, seen = [], set()
    for c in (d.get("controls") or [])[:3]:
        cid, kind = str(c.get("id") or ""), c.get("type")
        if not IDENT.match(cid) or cid in seen or kind not in TOY_TYPES:
            continue
        seen.add(cid)
        out = {"id": cid, "type": kind, "label": _plain(c.get("label") or cid, 28)}
        if kind in ("slider", "time"):
            lo, hi, step = _num(c.get("min")), _num(c.get("max")), _num(c.get("step"))
            if lo is None or hi is None or hi <= lo:
                continue
            step = step if step and 0 < step <= hi - lo else (hi - lo) / 100
            val = _num(c.get("value"))
            out.update(min=lo, max=hi, step=step, value=min(hi, max(lo, val if val is not None else lo)))
            if kind == "time":
                out["seconds"] = min(12.0, max(1.0, _num(c.get("seconds")) or 4.0))
        elif kind == "select":
            opts = [_clip(o, 14) for o in (c.get("options") or [])[:5] if str(o).strip()]
            if len(opts) < 2:
                continue
            out.update(options=opts, value=c.get("value") if c.get("value") in opts else opts[0])
        else:
            out["value"] = bool(c.get("value"))
        controls.append(out)
    height = _num(d.get("height")) or 190
    return {"controls": controls, "height": int(min(280, max(120, height))), "code": code,
            "hint": _plain(d.get("hint"), 170), "caption": _plain(d.get("caption"), 200)}


TASKS = {
    "play": "Build the toy for the highlighted idea. Use the paper below for its exact formulas, constants and settings.",
    "sketch": "Sketch what the highlighted phrase is. The paper below is context for what it means here; explain it "
              "with everything you know.",
}


def _clean(kind: str, d: dict) -> dict:
    spec = _clean_toy(d)
    if kind == "sketch":
        spec.update(controls=[], hint="")  # a still picture
    return spec


def make(kind: str, paper: Paper, selection: str, passage: str) -> dict:
    """Beam writes the code for a toy ("play") or a still sketch ("sketch"); the browser tests it and asks for fixes."""
    spec = _ask_json(PLAY_PROMPT.format(skill=SKILLS[kind], title=paper.title, selection=selection, passage=passage,
                                        task=TASKS[kind], text=paper.text[:MAX_PAPER_CHARS]), max_tokens=24_000)
    return _clean(kind, spec)


def fix(kind: str, paper: Paper, selection: str, passage: str, spec: dict, error: str) -> dict:
    """One repair round: Beam gets its failing code and the exact test failure, and returns a corrected version."""
    fixed = _ask_json(FIX_PROMPT.format(skill=SKILLS[kind], title=paper.title, selection=selection, passage=passage,
                                        error=_clip(error, 600), spec=json.dumps(spec, ensure_ascii=False)),
                      max_tokens=24_000)
    return _clean(kind, fixed)


if __name__ == "__main__":
    import sys

    aid, concept = sys.argv[1], sys.argv[2]
    tree = trace(aid, concept, depth=int(sys.argv[3]) if len(sys.argv) > 3 else 2, on_progress=print)
    print(json.dumps(tree.to_dict(), indent=2))
    print(f"\n{spent['calls']} Beam calls, ${spent['usd']:.4f}")
