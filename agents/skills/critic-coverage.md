---
skill: critic-coverage
description: Critic — scores whether the page covers the topic at the right breadth and depth for its layer (curriculum vs arc-step vs arc-index). Catches both under- and over-scoping.
applies_to: [review]
triggers: [all pages]
---

# Critic: Coverage

You are scoring **one dimension only**: is the page covering the topic at the right scope for its layer?

Two failure modes:
- **Under-scoped** — page is a stub-sized treatment of a major concept.
- **Over-scoped** — page tries to teach the entire field; should have been split into multiple pages or moved to an arc.

## How to score (step-by-step, do NOT one-shot it)

See `reasoning-scaffolding.md` for the audit pattern. For coverage specifically:

1. **Locate the layer** — check frontmatter `page_type` (`core-concept` / `arc-step` / `arc-index`). Apply the corresponding standard from the table below.
2. **Count words rendered** — strip frontmatter and code blocks, count remaining words. Compare to the layer's expected range.
3. **Walk required sections** — go down the layer's required-section list in order, marking each present / thin / missing.
4. **Tally deductions** — apply each deduction from the "Specific deductions" section literally; don't paraphrase.
5. **Compose** — score = 1.0 minus tallied deductions, clamped to [0, 1]. Issues list names the specific gaps; fix_suggestions lists the section additions.

## Layer-specific standards

### Curriculum page (`page_type: core-concept`)
- 1500–3500 words rendered.
- Covers: opening intuition, why it matters at the frontier, core concepts (5–8 bullet definitions), mathematical foundations (3–5 equations with symbol explanations), key algorithms/techniques (5–10), essential reading (3–5 papers across seminal/test-of-time/SotA), current SotA, what's happening now, in production (3–5 named systems), MVBs (6 persona variants), open questions (3+), connected topics (3+).
- Does NOT cover: full tutorials, step-by-step training scripts, hyperparameter tables longer than ~10 rows.

### Arc step page (`page_type: arc-step`)
- 800–2000 words rendered.
- Covers: this step's MVB (one variant, persona declared), what this step compounds with (prev_artifact + produced artifact named), the 1–3 specific readings that motivate this step, sanity checks, expected output shapes + numbers, failure modes, the open question this step makes answerable.
- Does NOT cover: the general theory (defer to curriculum), other arcs' steps, exhaustive citation lists.

### Arc index page (`page_type: arc-index`)
- 1000–2500 words rendered.
- Covers: destination statement, why this arc exists, prereqs, the chapters (3–4 with 2–3 steps each), curated readings per chapter (1–3 papers), the compounding-trajectory table, where this arc leads to other arcs.
- Does NOT cover: implementation details, MVB recipes (those belong on step pages).

## Scoring rubric (0.0 – 1.0)

| Score | What it means |
|---|---|
| 1.0 | Word count in range. All required sections present. No section is empty or one-line. No section repeats material from another. |
| 0.85 | Word count off by 10–20%, but content covers the right ground. One section is thin but present. |
| 0.7 | Two or three required sections are missing OR one major section is over-padded with filler. |
| 0.5 | Skeletal — under 60% of expected word count, or several core sections empty. |
| 0.0 | Stub-sized or over-bloated past 2× the expected range. |

## Specific deductions

- Missing "Mathematical foundations" section on a curriculum page that has a known derivation (DDPM, transformer, GP, etc.) → −0.15
- "Current SotA" section without specific benchmark numbers or model names → −0.15
- "In production" section absent on a topic with known production deployments → −0.10
- MVB has fewer than 4 persona variants on a curriculum page → −0.10 (this overlaps with `critic-human-centered`; both still count)
- Arc step page with no `compounding_artifact` declared in frontmatter → −0.20
- Arc index without a compounding-trajectory table → −0.20
- Over-scoping: page tries to teach more than the topic's natural boundary (e.g., a "Diffusion Models" page that also tries to teach flow matching) → −0.15

## Special case: topics outside the canonical 10 tracks

For tracks 11–15 (robotics, physics-AI, graph-AI, biology, ML theory): coverage standards still apply but the topic boundary tends to be narrower. A "Foundation models for robotics" page should not also try to teach RL fundamentals. Score normally.

## Output format

```json
{
  "score": 0.0,
  "issues": [
    "Page is 800 words — under-scoped for a curriculum page on diffusion models",
    "No 'In production' section; latent diffusion has multiple major production deployments (Stable Diffusion, FLUX, etc.)"
  ],
  "fix_suggestions": [
    "Expand the page to 1500+ words; key gap is 'Mathematical foundations' and 'In production'",
    "Add 3–5 named production deployments with one-sentence context each"
  ]
}
```
