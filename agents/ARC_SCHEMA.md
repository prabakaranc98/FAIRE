# Arc Page Schema

Arc pages are pedagogical guides — narrative, sequential, directive. They tell a learner exactly what to learn, in what order, and why. They are human-authored, not agent-generated.

The arc index pages live at `docs/arcs/[arc-slug]/index.md`.

---

## Frontmatter

```yaml
---
title: "Arc: [Name] — [subtitle]"
arc: [arc-slug]
super_domain: [A-Foundations | B-Modeling | C-Decision | D-Perception | E-Science]
tracks: [track1, track2, ...]
estimated_depth: "N-M weeks, ~X papers"
prereqs: [specific topics — not "ML basics"]
---
```

---

## Page body

```markdown
# Arc: [Name]
> **What this arc builds:** one sentence — the mental model or capability you develop.

## Why this arc exists
[2 paragraphs: what connected idea this arc unpacks; why this particular sequence;
what you gain from the arc that you couldn't get by reading topics in isolation]

## Prerequisites
[Specific prerequisites — name actual topics, not "ML basics".
e.g., "Backpropagation, cross-entropy loss, Adam. Familiarity with Gaussian distributions."]

## The sequence

**[Section heading]**

1. **[Concept]** (foundational) — [one sentence: what it is and why it comes first]
2. **[Concept]** (applied) — [one sentence + optional arXiv link]
...
N. **[Concept]** (frontier) — [current frontier result — named paper, year]

[4 depth tags: (foundational) (applied) (theoretical) (frontier)]

## Key figures
- **[Name]** ([affiliation]) — [key contribution to this arc's topic]
[3-6 researchers who defined this area]

## Essential reading sequence
> Read in this order. Each paper builds on the previous.
1. [Paper](arxiv/edu link) — [Authors, Year] — [one sentence: what it establishes in the sequence]
2. ...
[6-10 papers forming the reading spine]

## Current frontier anchors
> As of [YYYY-MM-DD]
- **[System / Paper]** — [one sentence on what it achieves and why it's frontier]
[3-6 anchors representing where the arc terminates today]

## What you'll know when done
[3-5 concrete outcomes. Not "understand X" — but "Implement X", "Explain why Y works",
"Describe Z's tradeoffs", "Read a new paper and identify which problem it addresses"]

## Branch points to other arcs
- **→ [Arc Name] arc**: [specific concept that bridges — one sentence]
- **→ [Arc Name] arc**: [specific concept that bridges — one sentence]

## Where to go next
[1-2 recommended next arcs with links]
```

---

## Design principles for arc pages

- **20-28 nodes** — enough depth to be transformative, not so much it's overwhelming
- **Sequences, not lists** — every node should explain why it comes at this point
- **Named frontier anchors** — the arc ends at specific, named systems/papers, not vague "current research"
- **Specific prerequisites** — "backpropagation and PyTorch basics" not "ML knowledge"
- **Concrete outcomes** — "implement a speculative decoding loop" not "understand speculative decoding"
- **Branch points are first-class** — listed explicitly, not footnoted
