# Skill: playable toys

A reader highlighted a phrase in a research paper. You build a **tiny interactive toy** that sits in the paper's
margin and lets them *play with the idea*: drag a slider, flip a switch, press play, and watch the phenomenon the
phrase describes actually happen. Think Bret Victor's explorable explanations, made for one sentence.

It is drawn in a hand-sketched Excalidraw style on a small canvas (260 units wide), with 1–3 controls underneath
and a few live numbers ("readouts").

## What makes a great toy

1. **One interaction reveals the insight.** Pick the single knob that makes the idea click — the variable the
   phrase is about. The reader should get it within five seconds of dragging.
2. **Start in the "before" state, and say what to try.** Defaults show the ordinary, well-behaved case; the
   `hint` tells the reader exactly what to do to see the interesting thing ("drag m past 16 in fp32", "press ▶ and
   watch the loss"). The hint must match the defaults: if it says "turn scaling off", scaling starts **on**; if it
   says "drag dₖ up", dₖ starts low.
3. **Real, not decorative.** Compute the actual math: the paper's formulas, constants, sizes and number formats.
   If the phrase is about precision, use `api.round` / `api.f16` / `api.bf16` / `api.f32` — JS numbers are fp64,
   so without them nothing interesting will happen. If you must simplify (a toy 2-D version of a big model), keep
   the mechanism exact and say so in the caption.
4. **Show the consequence, not just the input.** Pair the knob with what it changes: a curve that bends, bars
   that collapse onto one spike, a gradient that hits zero, a loss that diverges. Readouts state the punchline as
   numbers ("gradient 0", "max weight 0.99").
5. **Show the whole structure, then let the control explore it.** Prefer the view where the pattern is visible
   at a glance — all dimensions as a heatmap with the slider highlighting one row, a full curve with a moving
   point — over a view that only shows one slice at a time.
6. **Contrast when it helps.** A faint "reference" curve or a toggle between the broken and fixed version
   (standard vs. stable, with vs. without scaling) makes the effect undeniable.
7. **Show the trend, not the noise.** When the toy uses random data, average over enough draws (e.g. 32–128
   trials, computed in `setup`) so curves show the expected behaviour, and reuse the same random vectors across
   settings so moving a slider changes only what the slider controls.
8. **Clean, smooth, never aliased.** Sample curves densely (about 200–400 points across the visible range, and
   at least ~8 points per period of the fastest oscillation shown). Pick axis ranges that frame the interesting
   part, with round tick values.
9. **Minimal ink, but nothing ambiguous.** One accent color for the thing that matters; everything else ink or
   muted. When there are two or more curves, **label each one** (`ax.line(xs, ys, {label: "standard"})`, placed with
   `labelAt` where the curves are apart) and give readouts the same colors as their curves. Labels are short, in
   Unicode math (`QKᵀ`, `√dₖ`, `e⁻ᵐ`, `σ²`, `gₜ`). No titles — the note already has one.

Good fits: precision/overflow/underflow, scaling laws and asymptotics, softmax/temperature/attention patterns,
optimization dynamics (use a `time` control), sampling/noise, normalization, initialization variance through depth,
regularization strength, schedules, sparsity, thresholds and phase changes, small algorithms run step by step.

## Output — return ONLY one JSON object

```json
{
  "controls": [ ... 1 to 3 controls ... ],
  "height": 190,
  "code": "function setup(api) { ... }\nfunction draw(p, api, state) { ... }",
  "hint": "Short instruction: what to drag and what to watch for.",
  "caption": "One sentence: what the toy demonstrates."
}
```

`height` is the canvas height in units (120–280; 160–200 is typical). Width is always 260.

### Controls
```json
{"id": "m",    "type": "slider", "label": "margin m",  "min": 0, "max": 40, "step": 0.1, "value": 5}
{"id": "prec", "type": "select", "label": "precision", "options": ["fp16", "bf16", "fp32"], "value": "fp32"}
{"id": "fix",  "type": "toggle", "label": "stable version", "value": false}
{"id": "t",    "type": "time",   "label": "step", "min": 0, "max": 300, "step": 1, "value": 0, "seconds": 4}
```
- `id` is a JS identifier; read it in `draw` as `p.m`, `p.prec`, `p.fix`, `p.t`.
- `select` has 2–5 short options. `time` is a slider with a ▶ button that sweeps from min to max over `seconds`.
- **Every control must visibly change the canvas or a readout at its extremes** — this is checked automatically.
- Choose ranges that include the interesting regime, with the default outside it.

### Code
Plain modern JavaScript (no imports, no DOM, no `window`, `document`, `fetch`, timers, `Date` or `Math.random`).
It runs sandboxed in a worker, gets re-run on every control change, and is checked at every control's extremes.

- `function setup(api)` — optional, runs once; return any precomputed state (arrays, a seeded RNG, weights).
- `function draw(p, api, state)` — required; reads controls from `p` and draws with `api`. Must be pure: same `p`
  → same picture. For `time` toys, recompute the trajectory up to `p.t` (cache the whole run in `setup` if cheap).
- Keep each `draw` under ~20 ms: ≤ 400 points per curve, ≤ ~50k arithmetic steps. Precompute in `setup`.
- Keep the code short and readable: ≤ 70 lines.

## The `api`

See the API reference appended below.

## Example 1 — "temperature controls how peaked sampling is"

```json
{"controls": [{"id": "T", "type": "slider", "label": "temperature T", "min": 0.1, "max": 5, "step": 0.05, "value": 1}],
 "height": 180,
 "code": "function draw(p, api) {\n  const logits = [2.0, 1.0, 0.5, 0.2, -0.5];\n  const probs = api.softmax(logits.map(z => z / p.T));\n  api.bars({ values: probs, labels: [\"a\", \"b\", \"c\", \"d\", \"e\"], max: 1, highlight: 0, at: [20, 10, 220, 140] });\n  const H = -probs.reduce((s, q) => s + (q > 0 ? q * Math.log(q) : 0), 0);\n  api.text(240, 24, p.T < 0.4 ? \"≈ greedy\" : p.T > 3 ? \"≈ uniform\" : \"\", { align: \"end\", color: \"muted\" });\n  api.readout(\"top probability\", probs[0]);\n  api.readout(\"entropy\", H);\n}",
 "hint": "Drag T toward 0, then toward 5.",
 "caption": "Dividing logits by T sharpens the distribution as T → 0 and flattens it as T grows."}
```

## Example 2 — "(1 + x) − 1 loses x in low precision"

```json
{"controls": [{"id": "prec", "type": "select", "label": "precision", "options": ["fp16", "bf16", "fp32", "fp64"], "value": "fp64"},
              {"id": "logx", "type": "slider", "label": "log₁₀ x", "min": -12, "max": 0, "step": 0.1, "value": -4}],
 "height": 200,
 "code": "function setup(api) { return { xs: api.logspace(-12, 0, 160) }; }\nfunction draw(p, api, s) {\n  const r = v => api.round(v, p.prec);\n  const err = s.xs.map(x => Math.abs(r(r(1 + x) - 1) - x) / x);\n  const ax = api.axes({ x: [1e-12, 1], y: [1e-8, 10], xlog: true, ylog: true, xlabel: \"x\", ylabel: \"rel. error\",\n    xticks: [1e-12, 1e-6, 1], xtickLabels: [\"10⁻¹²\", \"10⁻⁶\", \"1\"], yticks: [1e-8, 1e-4, 1], ytickLabels: [\"10⁻⁸\", \"10⁻⁴\", \"1\"] });\n  ax.line(s.xs, err, { color: \"accent\" });\n  const x = 10 ** p.logx, e = Math.abs(r(r(1 + x) - 1) - x) / x;\n  ax.vline(x);\n  ax.point(x, Math.max(e, 1e-8), e >= 0.99 ? \"lost\" : \"\");\n  api.readout(\"(1 + x) − 1\", r(r(1 + x) - 1));\n  api.readout(\"relative error\", e);\n}",
 "hint": "Switch to fp16 and slide x down: the error jumps to 100%.",
 "caption": "Adding a tiny x to 1 rounds it away; the smaller the format's mantissa, the sooner x vanishes."}
```

## Before you answer, check

1. Does dragging the main control make the phrase's claim visibly happen? Is the default the calm "before"
   state, and does the hint's instruction start from exactly those defaults?
2. Is the math the paper's (formulas, constants, formats), with rounding applied wherever precision matters?
3. Is every curve smooth (dense sampling) and every curve/series labelled? Does every control change something at its min and max? Are all values finite for every setting
   (guard divisions, logs and empty arrays)?
4. Is the code ≤ 70 lines, deterministic, with no DOM/network/timers/Math.random?
5. Is the `hint` a concrete action, and the `caption` one sentence about what it demonstrates?
