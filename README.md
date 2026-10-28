# lineage.

**An annotated reader for arXiv, powered by [Reflection's Beam](https://huggingface.co/reflection/Beam-1.0-501B-A23B).**

Paste any arXiv link. The paper opens as a clean, centered page (with its math, figures and tables). Highlight any
phrase and pick one of four actions. Answers appear as notes in the margin, each underlined in its own color.

- **What is this?** A plain-language explanation of the phrase, grounded in the paper.
- **Sketch.** A hand-drawn whiteboard diagram of what the idea *is*. Beam writes the drawing as code; it's checked
  for overlapping labels and wasted space, and anything that fails goes back to Beam to fix.
- **Play.** A tiny interactive simulation of the idea — drag a slider and watch the paper's claim happen (precision
  cliffs, attention saturating, optimizer bias…). Beam writes the toy; a sandbox runs every control at its extremes
  (crashes, endless loops, NaNs, dead controls); failures go back to Beam with the exact error, up to twice.
- **Where's it from?** Beam fetches the papers cited around the phrase and reads them all in one pass (Beam has a
  1M-token context), then writes a short answer where every claim is footnoted with a quote that is verified
  against the cited paper's text.

## How it uses Beam

| Feature | What Beam does |
|---|---|
| What is this? | Reads the whole paper and explains the phrase in context |
| Sketch / Play | Writes JavaScript against a small drawing + numerics API (incl. exact fp16/bf16/fp32 rounding), then repairs it from test failures — an agentic write → test → fix loop |
| Where's it from? | Reads up to ~10 full cited papers at once and cites them with verbatim quotes |

The design rules Beam follows live in `skills/` (`toy.md`, `sketch.md`, `canvas-api.md`).

## Safety

Beam's code runs in a Web Worker inside a sandboxed iframe (opaque origin, no network, killed if it runs too long).
This public demo has a daily spending cap and a per-visitor hourly limit.

## Run it yourself

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/utsav1033/lineage)

Or locally:

```bash
pip install -r requirements.txt
HF_TOKEN=hf_... uvicorn server:app --port 7860
```

Optional settings: `LINEAGE_DAILY_BUDGET_USD` (default 3), `LINEAGE_REQUESTS_PER_HOUR` (default 40).

Built for the Beam challenge.
