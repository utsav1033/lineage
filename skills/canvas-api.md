# The drawing and numerics API (shared by sketches and toys)

Canvas units: x ∈ [0, 260] left→right, y ∈ [0, height] top→bottom. `api.W` = 260, `api.H` = height.
Colors (`color` option): `"ink"` (default), `"muted"`, `"accent"`, `"good"`, `"bad"`, `"faint"`.
Text sizes (`size`): `"xs"` (ticks), `"s"` (default), `"m"`, `"l"`.

**Charts — use these for anything quantitative**
- `const ax = api.axes({x: [x0, x1], y: [y0, y1], xlog, ylog, xlabel, ylabel, xticks, yticks, xtickLabels, ytickLabels, at})`
  draws L-shaped axes and returns a mapper. Default frame `at = [40, 10, W − 50, H − 44]` leaves room for labels.
  Ticks default to the two ends; give `xticks`/`yticks` arrays (and optional label arrays) for more.
  On a log y-axis, values ≤ 0 are drawn on the floor — so "dropped to exactly zero" stays visible.
  - `ax.line(xs, ys, {color, width, dashed, label, labelAt})` — a data curve (non-finite points break the line;
    values above the frame are clamped to its top). `label` writes text near point `labelAt` (default: last).
  - `ax.point(x, y, label, {color, size})` — a highlighted dot (accent by default).
  - `ax.dots(xs, ys, {color, size})`, `ax.vline(x, label, opts)`, `ax.hline(y, label, opts)` (dashed guides),
    `ax.band(x0, x1, {color, alpha, label})` (shaded regime), `ax.text(x, y, str, opts)` (data coordinates).
  - `ax.X(v)`, `ax.Y(v)` map data → canvas units; `ax.frame` is `[x, y, w, h]`.
- `api.bars({values, labels, max, highlight, colors, showValues, fillAll, at})` — a bar chart; `highlight` is an
  index or array of indices drawn in accent with hatching. Default frame `[20, 10, W − 40, H − 40]`.
- `api.heatmap({matrix, max, min, color, rowLabels, colLabels, at})` — e.g. attention weights.

**Diagram pieces**
- `api.box(x, y, w, h, label, {color, fill, size, textColor, dashed})` — a box with a centred label that wraps to
  fit inside it. Size boxes from their text: `w = api.textWidth(label, "s") + 20`, `h ≈ 30` per line.
- `api.arrow(points, {color, dashed, smooth, width})` — head at the last point; with 3+ points and
  `smooth: true` it curves (use for skip connections and loops).
- `api.text(x, y, str, {size, align: "start"|"middle"|"end", color, rotate, halo, maxWidth, valign: "middle"})` —
  `y` is the baseline (or the vertical centre with `valign: "middle"`). With `maxWidth` the text wraps.
- `api.textWidth(str, size)` — approximate width in canvas units, for laying things out.
- `api.line(x1, y1, x2, y2, o)`, `api.path(points, {smooth, ...o})`, `api.rect(x, y, w, h, {fill, alpha, ...o})`,
  `api.circle(cx, cy, diameter, {fill, ...o})`.
Options shared by strokes: `color`, `width`, `dashed`, `crisp` (no hand-drawn wobble).

**Layout is checked automatically**: overlapping labels, or a drawing that fills under ~30% of the canvas, fail
the test and come back to you to fix. Compute positions from a few variables (column x's, row y's) rather than
scattering literals, leave ~8 units between labels, and size `height` to the content.

**Readouts** — `api.readout(label, value, {color, digits})`: a live number row under the canvas. 1–3 of them,
stating the punchline. Numbers are formatted automatically (tiny/huge values in scientific notation).

**Numerics**
- `api.round(x, fmt)` with fmt `"fp16" | "bf16" | "fp32" | "fp64"`; also `api.f16(x)`, `api.bf16(x)`, `api.f32(x)`.
  Exact IEEE round-to-nearest-even, including overflow to ∞ and subnormals. To emulate a computation in a
  format, round **after every operation**: `const r = v => api.round(v, p.prec); r(r(a * b) + c)`.
- `api.softmax(xs, fmt?)` — max-shifted softmax, optionally computed entirely in `fmt`.
- `api.linspace(a, b, n)`, `api.logspace(a, b, n)` (10^a … 10^b), `api.range(n)`, `api.sum`, `api.mean`,
  `api.clamp(v, lo, hi)`, `api.fmt(v)`.
- `api.rng(seed)` → `u()` uniform in [0, 1), `u.normal()` standard normal. Deterministic: call it in `setup`
  or at the top of `draw` with a fixed seed.

