# CONVENTIONS.md — Equations Rendered

> Sections 1–2 below are the real, verbatim standing reference (confirmed
> 2026-08-07). Sections 3–7 are new additions established over the course of
> the visual-upgrade and streaming-architecture work in this same conversation.
> One direct conflict was found and resolved — see the note in 3.3.

---

## 1. Reporting Rules

- NEVER reference images via file:// paths or local paths in a report. Always attach
  actual image files directly in the message.
- NEVER describe or narrate what an image "shows" unless that image is actually attached
  in the same message. Do not pre-write a description of expected output before the
  output exists.
- NEVER claim something is "verified," "confirmed," or "done" without the actual
  evidence attached in that same message: an image, a test output number, or an
  ffprobe/ffmpeg result.
- If something was not actually tested or rendered, say so plainly ("not yet tested" /
  "rendering in progress") — do not imply completion before it's real.
- When a bug is found, state the actual root cause once confirmed (via logs, a traceback,
  or a direct test) — not a plausible-sounding guess offered as if it were the confirmed
  cause.

---

## 2. Per-Simulation Requirements

Before reporting any simulation as done, self-check ALL of the following:

1. **Equation caption**: matches THIS simulation's actual governing equation. Verify by
   reading the actual equation string being passed into the config/render call — do not
   assume a default or placeholder value is correct. Render via matplotlib mathtext;
   confirm no raw LaTeX (backslashes, braces) is visible in the output.
2. **Title**: Times New Roman, no box outline, sized and positioned consistently with
   the existing verified simulations.
3. **Readout**: positioned below the title, displaying this simulation's actual
   variable_log values for the frame shown — not zeroed out, not stuck at a constant,
   not missing. Label text must accurately describe what's being measured for THIS
   simulation (e.g. don't reuse a "Divergence (σ)" label borrowed from a
   multi-trajectory simulation if this simulation is single-trajectory). **Color: see
   Section 3.3 — this superseded the original flat-green rule with role-based coloring.**
4. **Color palette**: consistent role-based colors per the established universal palette
   — see Section 3.3 for the current (role-based) version of this rule, which supersedes
   the original red/blue/green/orange-to-magenta/coral/cool-blue scheme described here.
   No random or ad-hoc per-simulation colors.
5. **Duration**: computed via the correct category logic and clamped 10-30s (unless a
   category-specific exception applies — fractal zoom pacing, one full drift cycle for
   parametric curves, bounding-box plateau for multi-trajectory ODEs). Never a hardcoded
   guess with no stated reasoning.
6. **TEST_SPEC**: declared for this simulation, and tests/run_physics_tests.py passes
   for it — run and confirm BEFORE generating any render for visual review. Report the
   actual pass output with real numbers, not just "PASS."
7. **Auxiliary curves** (if used): every declared component curve actually renders and
   grows correctly, each labeled at its live endpoint, matching the working Euler POC
   pattern. Do not assume a new simulation's aux curves work just because Euler's did —
   verify directly.
8. **Equation-to-code correspondence**: the displayed equation caption must be verified
   against the ACTUAL implementation, not just the config string. State explicitly:
   "the equation shown is [X], and the code computes [specific line/function reference]
   which implements [X]" — a direct, explicit correspondence check, not an assumption
   that the config's LaTeX string was written correctly when the module was first built.
   Any time a simulation's underlying dynamics/parameters are modified, this check must
   be redone — a code change to the physics without a corresponding equation-string
   check is exactly the kind of silent mismatch this item exists to catch.

### Before Reporting Any Fix or Batch as Complete
- Re-render the affected simulation(s) fresh — never reuse old output files as evidence
  of a fix.
- Extract start/mid/end frames at genuinely different, correctly-ordered timestamps
  (verify Start < Mid < End in actual simulation progress, not just in filename) and
  attach them directly to the report.
- Where a claim can be checked with a number (duration via ffprobe, energy drift via the
  test suite, etc.), report that number — don't rely on visual impression alone where a
  concrete check exists.

---

## 3. Visual Styling Conventions

### 3.1 Layout
- Black background (`#000000`), unchanged — do not desaturate the background.
- Times New Roman titles, no box, white.
- Plot region: full-bleed within its frame area — do not shrink/pad the plot to
  make room for other elements (tried once for a subtitle tier, reverted —
  see 3.2).
- Equation caption: matplotlib mathtext, centered, at the bottom of frame.

### 3.2 Typography Hierarchy — no on-canvas subtitle
An on-canvas subtitle tier (plain-language description below the title) was
tried and reverted. Context/framing beyond the title itself belongs in the
**YouTube title field** at publish time (e.g. "Gradient Descent — Navigating a
Multi-Modal Loss Surface"), not baked into the rendered video. Reasons:
competes for space with the readout line, often redundant with the title,
and doesn't match the reference layouts that motivated the styling upgrade
(none combined subtitle + dense metrics + annotation at once).

### 3.3 Color System

> **Supersedes the original palette rule in Section 2.3–2.4.** The original
> convention specified flat green for all readout text and a fixed
> red/blue/green/orange-magenta/coral/cool-blue scheme. That was replaced during
> the visual-upgrade work with the role-based system below, verified end-to-end
> on `gradient_descent` (color-split confirmed via direct pixel sampling, not
> just visual review). Older simulations not yet part of the rollout still use
> the original flat-green scheme until they're updated — Section 2's readout
> and palette rules should be read as "current for un-migrated simulations,
> superseded once a simulation goes through the rollout described in Section 7."

**Structural roles** — tied to a specific plot element, fixed unique color each:

```python
ROLE_COLORS = {
    "primary": "#E85D4A",      # soft red — main trajectory/body
    "secondary": "#5DA8E8",    # powder blue — comparison trajectory
    "auxiliary": "#7FAE6B",    # sage green — derived/aux curve
    "control": "#D4C24A",      # muted yellow — tunable parameter
    "static": "#CFCFCF",       # off-white — fixed parameter
}
```

**Scalar metrics** — reserved for aggregate/summary statistics with **no single
corresponding colored element** (e.g. gradient_descent's Avg Loss/Avg Gradient,
which describe all four trajectories collectively). Pulled from an ordered
palette by index:

```python
METRIC_COLORS = [
    "#E86B5D",  # coral — metric 0
    "#B87FC9",  # dusty purple — metric 1
    "#C97FA0",  # muted rose — metric 2
    "#7FC9B0",  # muted teal — metric 3
]
```

> **Rule — `METRIC_COLORS` vs `ROLE_COLORS` for readout text:** if a scalar
> value has a clear 1:1 correspondence to one specific colored plot element (a
> single trajectory, a single body, a single curve), it must use that element's
> existing `ROLE_COLORS` hex directly — not a `METRIC_COLORS` index. This
> preserves the founding principle behind color-coding in the first place (a
> readout functions as a legend for the plot — see the original RK4/sigmoid
> reference analysis, where `k1`–`k4` were yellow specifically because the
> yellow curve was the RK4 approximation). `METRIC_COLORS` exists only for the
> case where no such correspondence is possible — a true aggregate across
> multiple elements.
>
> Before assigning a color to any metric on a not-yet-migrated simulation,
> classify it first: does this value describe one specific colored curve/body
> (→ use that element's `ROLE_COLORS` hex), or is it a genuine aggregate with no
> single corresponding element (→ use `METRIC_COLORS`)? Flag ambiguous cases
> rather than defaulting to `METRIC_COLORS`. This classification must appear in
> the Section 6.1 pre-implementation checklist for each simulation, before any
> render is sent for review.

- Every `variable_log` entry must have a `role`. If `role == "metric"`, it must
  also have a `metric_index` (see schema, Section 5).
- No simulation should retain pure-saturated RGB (`#FF0000`/`#00FF00`/`#0000FF`)
  anywhere in trajectory/trail rendering — verify via direct pixel sampling
  against the hex values above, not visual impression alone (muted colors can
  still look vivid against a near-black background at thin line widths — this
  is expected, not a bug; confirm via pixel value, not perception).
- Metric-separator pipes (`|`) between readout values: neutral light grey
  `#B0B0B0` (same tone as axis lines, Section 3.5), not inherited text color.

### 3.4 Pedagogical Annotations
- Optional per simulation — only add where there's a genuine equation-to-visual
  connection (e.g. gradient_descent's clip-bound circle, epicycloid's R/r radii,
  double_pendulum's pivot angle arcs). Do not force a contrived annotation onto
  a simulation with no natural candidate (e.g. mandelbrot, julia, random_walk) —
  skip cleanly instead.
- Thin lines (1–1.5px), never competing visually with the primary curve.
- Labels colored to match the role of the annotated element.
- Implementation: `generate()` may return an optional 4th value, `annotations`
  — a list of dicts: `{type: 'bracket'|'line'|'point'|'angle'|'circle', coords, label, color}`.
  Compositor applies a generic drawing pass consuming this list, after the main
  trajectory, before the text overlay.
- **Annotations must match the literal spec given, not an approximation of it.**
  ("draw and label the radii R and r" means visible R/r text labels exist —
  not just an unlabeled line. "draw an angular arc" means an arc, not a straight
  line between two points.) See Section 6.1 for how this gets verified before
  review.

### 3.5 Axis Styling (where applicable)
- Thin (1px), muted gray-white (`#B0B0B0`).
- Open axes with arrow-tipped ends, not a closed box border.
- Sparse ticks, no gridlines.
- Italic serif for single-letter math axes (x, y, t); plain word labels for
  descriptive axes (Time, Probability, Feature X).
- Not every simulation has traditional axes (e.g. gradient_descent's heatmap) —
  note explicitly when this doesn't apply rather than skipping silently.

### 3.6 Frame Bounds / Bounding Box Computation

> Added after cropping was found across multiple simulations — trajectories/
> plots extending past frame edges, varying by simulation.

- Axis/plot bounds must be computed from the **full trajectory extent across
  the entire render**, not from initial conditions or a partial/early sample.
  A trajectory that grows, drifts, or reaches new extremes later in the render
  will outgrow bounds set before the render started — this is the default
  failure mode and must be explicitly ruled out for each simulation, not
  assumed fine.
- For simulations using a `BoundingBoxPlateauDetector` (chaotic ODEs — lorenz,
  three_body, rossler): the same bounds used to detect the duration plateau
  must also be the bounds used for the plot's axis limits. A mismatch between
  "when rendering stops" and "what area is rendered" is a likely source of
  edge-cropping specifically on these simulations.
- **Annotation geometry counts toward the bounding box**, not just the
  trajectory/trail itself. A radius circle, vector line, or label that extends
  beyond the trajectory's own extent (e.g. gradient_descent's clip-bound
  circle, epicycloid's radius lines) must be included when computing what area
  needs to fit in frame.
- Add a fixed margin (5–10% of computed range) around the final bounding box
  on all sides — trajectories/annotations that touch the edge of their own
  computed extent should not visually clip against the frame boundary.
- This check applies to every simulation, including new ones built from
  scratch (see Section 8) — bounds should be computed correctly from the
  start, not retrofitted after cropping is spotted in review.

---

## 4. Rendering Architecture

### 4.1 Streaming, Not Buffering
`generate(config)` yields frames one at a time (generator), rather than
returning a fully-materialized list. This is required — buffering all frames
for a long/high-resolution render causes OOM crashes (confirmed on Rössler at
900 frames / 1080×1080×4).

- `renderer.py` consumes the generator and pipes raw bytes directly into
  ffmpeg's stdin (`-f rawvideo`), rather than writing intermediate PNGs to disk
  or buffering in memory.
- `variable_log` may still accumulate as a normal list — it's small scalar
  data per frame, not the memory risk.
- Peak memory should stay in the low hundreds of MB regardless of render
  duration — verify via `psutil` peak RSS tracking, not just visual confidence
  that "it didn't crash."
- ffmpeg `stderr` is logged to a file (`{output_path}.ffmpeg.log`), never routed
  to `DEVNULL` — silently discarding stderr trades away real diagnostic
  capability to dodge a pipe deadlock that has a cleaner fix.

### 4.2 `generate()` Return Contract
Always return a consistent shape — do not branch return type based on whether
optional elements (aux curves, annotations) are present:

```python
def generate(config) -> tuple[Generator[Frame, None, None], list[VariableLogEntry], Optional[list], Optional[list]]:
    """Returns (frame_generator, variable_log, auxiliary_curves, annotations)."""
```

`auxiliary_curves` and `annotations` are `None` when not applicable — never a
different tuple arity depending on what's present.

---

## 5. `variable_log` Schema Contract

```python
from typing import TypedDict, Optional

class VariableLogEntry(TypedDict):
    name: str
    value: str          # pre-formatted, human-readable (e.g. "12.25 J")
    role: str            # one of ROLE_COLORS keys, or "metric"
    metric_index: Optional[int]  # required if role == "metric", else None
```

- Enforced with a runtime assertion at the point of use in the compositor:
  ```python
  assert isinstance(entry["value"], str), f"variable_log entry.value must be str, got {type(entry['value'])}: {entry}"
  ```
- This exists specifically to prevent raw dict/tuple structures leaking into
  rendered text (confirmed failure mode: `Energy: {'value': '12.25 J', ...}`
  rendered directly onto a frame when the compositor wasn't updated for a
  schema change).

---

## 6. Error Prevention Process

### 6.1 Pre-Implementation Self-Check (required before any render is sent for review)

Before rendering for review, produce a written checklist comparing what was
asked against what was implemented — line by line:

```
SPEC vs IMPLEMENTATION CHECK
- [Requirement, quoted from the prompt] → [what was implemented] → MATCH / DEVIATION (reason)
```

Any deviation — including simplifications, omissions, or "close enough"
substitutions — must be flagged explicitly with a reason here, not discovered
by the reviewer after the fact. This catches scope drift that no automated
frame-level check can catch, since the output may be technically valid and
still not match what was asked.

**For any simulation with a live scalar readout:** the checklist must also
classify each metric per the `METRIC_COLORS` vs `ROLE_COLORS` rule in Section
3.3 — state explicitly whether each value is tied to one specific colored
element (→ uses that element's `ROLE_COLORS` hex) or is a genuine aggregate
(→ uses `METRIC_COLORS`), before implementing the color assignment.

### 6.2 Unit Tests for Formatting Logic
Fast, direct tests for compositor formatting functions, run before any render
is attempted — separate from the full `test_visuals.py` suite:

```python
def test_format_readout_entry_basic():
    entry = {"name": "Energy", "value": "12.25 J", "role": "metric", "metric_index": 0}
    assert format_readout_entry(entry) == "Energy: 12.25 J"

def test_format_readout_entry_rejects_bad_shape():
    with pytest.raises(AssertionError):
        format_readout_entry({"name": "Energy", "value": {"nested": "dict"}, "role": "metric"})
```

### 6.3 Automated Accuracy-Check Layer (post-render)
- **Deterministic assertions**: trend-across-render checks (e.g. divergence
  must trend monotonically across the full sequence, not just at sampled
  endpoints), not just single-point validation.
- **Provenance sidecars**: per-frame JSON with git commit hash, render
  timestamp, frame index, input parameters, and (for OCR) the actual pixel
  bounding box of title/equation text as drawn by the compositor — read
  dynamically by tests, never a hardcoded crop region.
- **Visual diffing**: pHash/SSIM against the same-simulation-same-timestamp
  baseline (not "last render" — early frames of a correct chaotic-sim render
  can legitimately resemble each other).
- **OCR verification** (Tesseract): title exact match; equation checked via
  LaTeX-stripped fuzzy match (`difflib.SequenceMatcher`, global threshold
  `0.35`) **plus** a strict core-token rule — Latin function/operator names
  (`clip`, `sin`, `cos`, `log`, `dx`, `dy`, `dz`, `dt`, etc.) must appear in the
  OCR output or the check fails outright, regardless of overall fuzzy ratio.
  Greek-letter LaTeX command names (`theta`, `sigma`, `rho`, `alpha`, etc.) are
  excluded from the core-token rule, since Tesseract cannot read rendered Greek
  glyphs as their command names.
  - Per-simulation ratio overrides are allowed only when documented with a
    specific reason (e.g. `electric_field: 0.05`, justified by Tesseract's
    inability to parse stacked fraction/vector notation) — never used to
    paper over an unexplained failure.

### 6.4 Division of Labor
Automate: data shape/type correctness, spec-vs-implementation checklist
production, formatting function correctness, frame-level checks (6.3).
Keep human-reviewed: whether an annotation is pedagogically useful, whether a
color reads correctly in context, whether a design tradeoff is the right call.

---

## 7. Process Conventions

- Batch rollouts (e.g. applying a change across all 16 simulations) proceed in
  small batches (3–4 simulations), with real per-batch verification sent before
  moving to the next batch — not one bulk summary at the end.
- All exports for review must be actual attached files (drag-and-drop/upload),
  never screenshots of a screen (picks up OS/browser UI chrome) and never a
  `file://` path claimed as "attached."
- Claims of numeric results (memory usage, trend values, pixel colors) should
  be backed by the actual measurement shown, not just described — e.g. peak
  memory via `psutil`, color match via direct pixel sampling and hex comparison,
  not visual description alone.

---

## 8. New Simulations (built from scratch, not retrofitted)

Most of Sections 3–7 were written while retrofitting existing simulations, but
they're intended to apply from the start to any new simulation, not just be
checked afterward. When building a new simulation:

- Follow the `generate()` return contract (Section 4.2) and `variable_log`
  schema (Section 5) from the first implementation — don't build first and
  retrofit the shape later.
- Compute frame bounds correctly from the start (Section 3.6) — this was found
  as a retrofit issue across existing simulations; a new simulation should not
  repeat it.
- Classify any live metrics (`ROLE_COLORS` vs `METRIC_COLORS`, Section 3.3)
  and any candidate annotations (Section 3.4) as part of initial design, not
  as an afterthought.
- Run the Section 6.1 pre-implementation checklist and Section 6.2 unit tests
  before the first render, same as any other change.

**Known scaling limits to check before assuming the current framework covers
a new case:**
- `ROLE_COLORS` currently has 5 fixed slots (primary, secondary, auxiliary,
  control, static). A simulation with 3+ trajectories of equal narrative
  importance (e.g. `three_body`) may already be stretching this — confirm how
  it's currently being handled before assuming a new simulation with a similar
  shape has a clear answer.
- `METRIC_COLORS` currently has 4 predefined slots. A simulation needing more
  than 4 distinct simultaneous aggregate metrics has no defined behavior yet —
  decide (extend the palette vs. consolidate metrics) before hitting this in
  practice, rather than mid-implementation.
