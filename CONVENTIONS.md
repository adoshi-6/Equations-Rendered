# Rendering Engine Conventions — Standing Reference

## Reporting Rules
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

## Per-Simulation Requirements
Before reporting any simulation as done, self-check ALL of the following:

1. **Equation caption**: matches THIS simulation's actual governing equation. Verify by
   reading the actual equation string being passed into the config/render call — do not
   assume a default or placeholder value is correct. Render via matplotlib mathtext;
   confirm no raw LaTeX (backslashes, braces) is visible in the output.
2. **Title**: Times New Roman, no box outline, sized and positioned consistently with
   the existing verified simulations.
3. **Readout**: green text, positioned below the title, displaying this simulation's
   actual variable_log values for the frame shown — not zeroed out, not stuck at a
   constant, not missing. Label text must accurately describe what's being measured for
   THIS simulation (e.g. don't reuse a "Divergence (σ)" label borrowed from a
   multi-trajectory simulation if this simulation is single-trajectory).
4. **Color palette**: consistent role-based colors per the established universal palette
   — red/blue/green for primary bodies/pivots, orange-to-magenta gradient for
   multi-trajectory trails, coral/cool-blue for auxiliary curve pairs. No random or
   ad-hoc per-simulation colors.
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

## Before Reporting Any Fix or Batch as Complete
- Re-render the affected simulation(s) fresh — never reuse old output files as evidence
  of a fix.
- Extract start/mid/end frames at genuinely different, correctly-ordered timestamps
  (verify Start < Mid < End in actual simulation progress, not just in filename) and
  attach them directly to the report.
- Where a claim can be checked with a number (duration via ffprobe, energy drift via the
  test suite, etc.), report that number — don't rely on visual impression alone where a
  concrete check exists.
