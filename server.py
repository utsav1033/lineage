"""PaperLineage web app: a paper reader with Beam-powered explanations and idea lineage in the margin."""
import json
import os
import queue
import threading
import time
from collections import defaultdict, deque
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

import lineage
from papers import fetch_paper, parse_arxiv_id

app = FastAPI()
STATIC = Path(__file__).parent / "static"

# Per-visitor limit on Beam-backed requests, so one person can't spend the whole daily budget.
RATE_LIMIT = int(os.environ.get("LINEAGE_REQUESTS_PER_HOUR", "40"))
_hits: dict[str, deque] = defaultdict(deque)
_hits_lock = threading.Lock()


@app.middleware("http")
async def rate_limit(request: Request, call_next):
    if request.method == "POST" and request.url.path.startswith("/api/"):
        ip = (request.headers.get("x-forwarded-for") or (request.client.host if request.client else "?")).split(",")[0].strip()
        now = time.time()
        with _hits_lock:
            q = _hits[ip]
            while q and now - q[0] > 3600:
                q.popleft()
            if len(q) >= RATE_LIMIT:
                return JSONResponse({"error": "You've hit this demo's hourly limit. Try again in a bit!"}, status_code=429)
            q.append(now)
    return await call_next(request)


class Ask(BaseModel):
    arxiv_id: str
    selection: str
    block: int  # index of the paragraph the selection came from


def _load(arxiv_id: str):
    aid = parse_arxiv_id(arxiv_id)
    paper = fetch_paper(aid) if aid else None
    if not paper:
        raise HTTPException(404, "Couldn't find that paper on arXiv.")
    return paper


def _passage(paper, block: int) -> str:
    blocks = paper.text.split("\n")
    return blocks[block] if 0 <= block < len(blocks) else ""


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/toy")
def toy_runtime():
    """The sandbox page toys run in. The reader embeds it as <iframe sandbox="allow-scripts">."""
    return FileResponse(STATIC / "toy.html", headers={"Cache-Control": "no-cache"})


@app.get("/api/paper/{arxiv_id:path}")
def get_paper(arxiv_id: str):
    p = _load(arxiv_id)
    return {"arxiv_id": p.arxiv_id, "title": p.title, "blocks": p.text.split("\n"), "refs": p.refs, "math": p.math, "figures": p.figures}


@app.post("/api/explain")
def post_explain(q: Ask):
    p = _load(q.arxiv_id)
    before = lineage.spent["usd"]
    try:
        text = lineage.explain(p, q.selection[:1000], _passage(p, q.block))
    except Exception as e:  # noqa: BLE001 - budget, missing token, provider errors: show them in the note
        return {"error": str(e)}
    return {"text": text, "usd": lineage.spent["usd"] - before}


class Fix(Ask):
    spec: dict
    error: str


def _make(kind: str, q: Ask, spec: dict | None = None, error: str = ""):
    p = _load(q.arxiv_id)
    before = lineage.spent["usd"]
    try:
        args = (kind, p, q.selection[:1000], _passage(p, q.block))
        toy = lineage.fix(*args, spec, error) if spec is not None else lineage.make(*args)
    except Exception as e:  # noqa: BLE001 - show the reason in the note instead of a 500
        return {"error": str(e)}
    return {"toy": toy, "usd": lineage.spent["usd"] - before}


@app.post("/api/visualize")
def post_visualize(q: Ask):
    return _make("sketch", q)


@app.post("/api/play")
def post_play(q: Ask):
    return _make("play", q)


@app.post("/api/{kind}/fix")
def post_fix(kind: str, q: Fix):
    if kind not in lineage.SKILLS:
        raise HTTPException(404, "Unknown kind.")
    return _make(kind, q, q.spec, q.error)


@app.post("/api/trace")
def post_trace(q: Ask):
    """Stream newline-delimited JSON: progress messages, then the finished tree."""
    p = _load(q.arxiv_id)
    concept = (f'"{q.selection[:1000]}" (the phrase the reader highlighted; trace only this). '
               f'For context, it appears in this passage of "{p.title}": "{_passage(p, q.block)}"')
    events: queue.Queue = queue.Queue()

    def run():
        before = lineage.spent["usd"]
        try:
            tree = lineage.trace(p.arxiv_id, concept, depth=2,
                                 on_progress=lambda m: events.put({"progress": m}))
            events.put({"tree": tree.to_dict(), "usd": lineage.spent["usd"] - before})
        except Exception as e:  # noqa: BLE001 - report any failure to the client
            events.put({"error": str(e)})
        events.put(None)

    threading.Thread(target=run, daemon=True).start()

    def stream():
        while (ev := events.get()) is not None:
            yield json.dumps(ev) + "\n"

    return StreamingResponse(stream(), media_type="application/x-ndjson")


@app.post("/api/synthesize")
def post_synthesize(q: Ask):
    """Answer from the cited papers themselves; streams progress, then the footnoted answer."""
    p = _load(q.arxiv_id)
    events: queue.Queue = queue.Queue()

    def run():
        before = lineage.spent["usd"]
        try:
            res = lineage.synthesize(p, q.selection[:1000], q.block, on_progress=lambda m: events.put({"progress": m}))
            events.put({"answer": res, "usd": lineage.spent["usd"] - before})
        except Exception as e:  # noqa: BLE001 - report any failure to the client
            events.put({"error": str(e)})
        events.put(None)

    threading.Thread(target=run, daemon=True).start()

    def stream():
        while (ev := events.get()) is not None:
            yield json.dumps(ev) + "\n"

    return StreamingResponse(stream(), media_type="application/x-ndjson")
