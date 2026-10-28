# Skill: whiteboard sketches

A reader highlighted a phrase in a research paper. You draw the one picture a great teacher would put on the
whiteboard so they get **what that phrase is** at a glance. It sits in the paper's margin, drawn in a hand-sketched
Excalidraw style on a small canvas (260 units wide). You draw it with code, using the API below. It is a still
picture: no controls.

## What to draw

1. **Explain the idea, not the sentence.** Use everything you know, not just this paper. The paper tells you what
   the phrase means *here*; the sketch shows *what it is, how it works, or why it matters*. Never just restate the
   passage in boxes.
2. **Show the mechanism with real structure.** Pick the picture that carries the meaning:
   - a process, architecture or algorithm → labelled boxes and arrows, data flowing top→bottom or left→right,
     with the key step accented
   - a quantity that grows, decays, saturates or has a threshold → a small chart (`api.axes` + `ax.line`) with the
     real curve shape computed from the formula, and the threshold marked
   - a formula → the formula in the middle, with arrows from its parts to 1–3 word meanings
   - a data layout (matrices, blocks, tiles, memory levels) → the actual grid/blocks, sized in proportion, with
     what moves where
   - a contrast → two side-by-side mini pictures with the difference visible, not described
3. **Concrete over abstract.** Draw actual things: a 6×6 attention matrix split into 2×2 tiles, the real sigmoid
   curve, three tokens with arrows between them. Small concrete instances beat vague symbols.
4. **One focal point.** The thing the reader highlighted is `"accent"` (fill it if it's a box); everything else is
   `"ink"`, with secondary detail `"muted"`.
5. **Few words.** About 6–15 words in total, as short labels in the paper's own terms and Unicode math
   (`QKᵀ`, `√dₖ`, `e⁻ᵐ`, `Σⱼ`, `ℓ`, `HBM`, `SRAM`). No titles, no sentences — the caption carries the sentence.
6. **Use the space.** Fill most of the 260-unit width; choose `height` (120–260) to fit the content snugly.
   Nothing overlaps unless that's the point. Layout is checked automatically.
7. **Be correct.** If you simplify, simplify truthfully: the picture must not teach something wrong.

## Output — return ONLY one JSON object

```json
{"height": 190,
 "code": "function draw(p, api) { ... }",
 "caption": "One sentence saying what the picture shows."}
```

Plain modern JavaScript, no imports, no DOM, no `window`, `document`, `fetch`, timers, `Date` or `Math.random`.
Define `function draw(p, api)` (`p` is empty for sketches); optionally `function setup(api)` for precomputation.
Keep it under ~60 lines, laid out from a few variables (column x's, row y's) rather than scattered literals.

## Example 1 — "residual connection"

```json
{"height": 196,
 "code": "function draw(p, api) {\n  const cx = 130, top = 20;\n  api.text(cx, top, \"x\", { size: \"l\", align: \"middle\" });\n  api.arrow([[cx, top + 6], [cx, 44]]);\n  api.box(cx - 50, 44, 100, 34, \"layer F\");\n  api.arrow([[cx, 78], [cx, 106]]);\n  api.circle(cx, 120, 28, { color: \"accent\", fill: true });\n  api.text(cx, 125, \"+\", { size: \"m\", align: \"middle\" });\n  api.arrow([[cx + 4, top + 4], [212, 40], [214, 104], [cx + 15, 118]], { color: \"accent\", dashed: true, smooth: true });\n  api.text(222, 74, \"skip\", { color: \"accent\" });\n  api.arrow([[cx, 134], [cx, 160]]);\n  api.text(cx, 180, \"x + F(x)\", { size: \"m\", align: \"middle\" });\n  api.text(12, 100, \"F only learns the change\", { color: \"muted\", maxWidth: 70 });\n}",
 "caption": "The input skips around the layer and is added back, so the layer only has to learn a correction."}
```

## Example 2 — "ReLU"

```json
{"height": 170,
 "code": "function draw(p, api) {\n  const xs = api.linspace(-3, 3, 121);\n  const ax = api.axes({ x: [-3, 3], y: [-1, 3], xticks: [-3, 0, 3], yticks: [0, 3], xlabel: \"input\", ylabel: \"output\" });\n  ax.line(xs, xs, { color: \"faint\", dashed: true, label: \"identity\", labelAt: 100 });\n  ax.line(xs, xs.map(v => Math.max(0, v)), { color: \"accent\", width: 2.2, label: \"ReLU\", labelAt: 110 });\n  ax.band(-3, 0, { color: \"muted\", alpha: 0.08, label: \"zeroed\" });\n}",
 "caption": "ReLU passes positive inputs through unchanged and clamps every negative input to zero."}
```

## Before you answer, check

1. Would someone who has never seen this phrase understand what it *is* from the picture in two seconds?
2. Does it show structure or shape (boxes and arrows, a real curve, a real grid), not a restated sentence?
3. Exactly one accented focal point? ≤ ~15 words? Labels short and in Unicode math?
4. Positions computed from a few variables, nothing overlapping, the canvas mostly used, `height` snug?
