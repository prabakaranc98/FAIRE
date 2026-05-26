---
title: FAIRE Structure v2 — The Guiding Skeleton
description: Canonical 3-layer content structure, 5 page types, and growth frontmatter. The design contract for the rewrite.
status: active-design
authored: 2026-05-26
---

# FAIRE Structure v2 — The Guiding Skeleton

> The canonical structure for FAIRE going forward. Established 2026-05-26 to replace the flat 15-track DL taxonomy with a 3-layer tree that scales, makes the **Decision Engineering convergence** explicit, and lets the four product pillars (Wikipedia + roadmaps.sh + Papers with Code + MVB nudge) land on every page.

## Why a new structure

The previous tree (`docs/curriculum/01-ai` … `15-ml-theory-foundations`) had three load-bearing problems:

1. **Wrong scope.** All 15 tracks are narrow DL/ML subdomains. The user's curriculum includes humanities (Philosophy, Psych & Behavioral Econ, Writing & Thinking, History of Intelligence) — currently zero coverage.
2. **No convergence.** Tracks are parallel silos with no structural pointer to the **Decision Engineering** thesis they're supposed to feed.
3. **Wrong unit.** Pages are concept-only. There is no first-class representation of authors (Pearl, Olah, Karpathy, Kahneman, Munger…), learning arcs (roadmaps.sh-style), or builds (MVB recipes / paperswithcode-style projects).

The fix is structural, not prose-level. No amount of writer-prompt tuning will produce a Decision-Engineering-converging, author-anchored, humanities-integrated wiki out of a flat technical taxonomy.

## Current scope — the 10 canonical tracks (user decision 2026-05-26)

The user has scoped v2 down to the 10 canonical tracks from pracha.me/curriculum. **Co-curriculum and the Decision Engineering thesis are deferred** — they remain in memory as future scope but are not part of the current build.

```
docs/curriculum-v2/
│
└── core/                              ◀── pracha.me canonical 10
    ├── 01-ai/
    │   ├── index.md                Overview · reading order · arc map
    │   ├── concepts/                  Encyclopedic, self-contained pages
    │   ├── authors/                   Person-anchored reading guides
    │   ├── arcs/                      Learning paths
    │   └── builds/                    MVB recipes / projects
    ├── 02-generative-modeling/
    ├── 03-representation-learning/
    ├── 04-neural-networks-deep-learning/
    ├── 05-statistical-probabilistic-ml/
    ├── 06-reinforcement-learning/
    ├── 07-attention-memory-reasoning-continual/
    ├── 08-causal-statistical-inference/
    ├── 09-algorithms-systems-for-ai/
    └── 10-complexity-cognition-natural-intelligence/
```

### Deferred (not in current build)

- **co/** — humanities + meta-skills (Philosophy, Psych/BE, History of Intelligence, Writing & Thinking, Economics & Markets). Documented in `agents/memory/faire_user_curriculum.md` (Layer 2). To revisit once core/ is stable.
- **thesis/decision-engineering/** — the unified structural stack (Systems Eng · Contextual AI/ML · MIRO · X). Documented in `agents/memory/faire_decision_engineering.md`. To revisit once core/ is stable and the convergence story is clearer.

The `feeds_de_pillar:` frontmatter field is **kept** on concept pages — it lets future Decision Engineering work bolt onto the existing tree without restructuring.

### Layer 1 — Core (10 canonical tracks)

The publicly-listed tracks at [pracha.me/curriculum](https://pracha.me/curriculum). These are the "range" of FAIRE's four-layer architecture (Curriculum → Arcs → Projects → Capstone).

| # | Track | Slug |
|---|---|---|
| 01 | AI | `01-ai` |
| 02 | Generative Modeling | `02-generative-modeling` |
| 03 | Representation Learning | `03-representation-learning` |
| 04 | Neural Networks & Deep Learning | `04-neural-networks-deep-learning` |
| 05 | Statistical & Probabilistic ML | `05-statistical-probabilistic-ml` |
| 06 | Reinforcement Learning | `06-reinforcement-learning` |
| 07 | Attention, Memory, Reasoning, Continual | `07-attention-memory-reasoning-continual` |
| 08 | Causal & Statistical Inference | `08-causal-statistical-inference` |
| 09 | Algorithms & Systems for AI | `09-algorithms-systems-for-ai` |
| 10 | Complexity, Cognition & Natural Intelligence | `10-complexity-cognition-natural-intelligence` |

The legacy tracks 11–15 (robotics, physics-scientific-ai, graph-relational, biology, ml-theory) are absorbed back into their canonical parents:

| Legacy | Absorbed into |
|---|---|
| 11 Robotics & Embodied AI | 06 RL · 10 Complexity/Cognition |
| 12 Physics & Scientific AI | 10 Complexity/Cognition (with arcs into 04, 05, 09) |
| 13 Graph & Relational AI | 04 NN & DL · 03 Representation Learning |
| 14 Biology & Life Sciences | becomes an *arc*, not a track (cross-cutting application) |
| 15 ML Theory & Foundations | 04 NN & DL · 05 Stat/Prob ML |

### Layer 2 — Co-curriculum (5 subjects)

Humanities and meta-skills that deepen the technical 10. **Do not compete for time** with core — they frame and contextualize it. Structurally required because they feed pillar 4 of Decision Engineering (X — Domain Context).

| Slug | Anchored in |
|---|---|
| `history-theory-of-intelligence` | Consciousness, philosophy of mind, history of AI as ideas |
| `philosophy` | Stoicism, existentialism, philosophy of science, ethics under uncertainty |
| `psychology-behavioral-econ` | Kahneman, Cialdini, Duhigg, Frankl, Ariely |
| `economics-and-markets` | Smith, Hayek, Christensen, Acemoglu, Munger |
| `writing-and-thinking` | Orwell, Pinker, Olah, Karpathy, Montaigne, DFW, Feynman |

### Layer 3 — Thesis (Decision Engineering)

The capstone. Source: [pracha.me/decision-engineering/decision-engineering-unified-structural-stack.html](https://pracha.me/decision-engineering/decision-engineering-unified-structural-stack.html).

Four pillars:

1. **Systems Engineering** — MBSE, system dynamics, organizational boundaries, feedback loops.
2. **Contextual AI/ML** — Modular algorithmic components, not monolithic end-to-end.
3. **MIRO Stack** — Modeling · Inference · Reasoning & Simulation · Optimization. Each is a subdirectory under `pillar-3-miro/`.
4. **X — Domain Context Variable** — Domain theory, design thinking for adoption, tailored execution context. Fed primarily by the co-curriculum.

This is where the curriculum converges. Every `concept` page in core/ and co/ must declare which pillar it feeds via `feeds_de_pillar:` frontmatter.

## The five page types

| Type | Filename pattern | Purpose |
|---|---|---|
| `subject-overview` | `index.md` (one per subject) | Where to start. Reading order, key authors, arc map. |
| `concept` | `concepts/<slug>.md` | Olah/Distill-grade self-contained encyclopedic article. The bread-and-butter. |
| `author` | `authors/<lastname>.md` | Person-anchored reading guide: bio · arc of their thinking · key works · builds inspired by them. |
| `arc` | `arcs/<slug>.md` | Roadmaps.sh-style learning path. Ordered sequence of concept pages with rationale at each step. |
| `build` | `builds/<slug>.md` | MVB recipe / project. Reproducible, persona-tagged, runs end-to-end. |

### How the five page types map to the four product pillars

| Product pillar | Carried by |
|---|---|
| Wikipedia (depth, citations, neutrality) | `concept` + `subject-overview` |
| roadmaps.sh (learning paths) | `arc` |
| Papers with Code (SotA, reproducible code) | `concept`'s Current SotA section + `build` |
| MVB nudge (push to building) | `build` + every `concept`'s "What can you build next" block |

This is what dissolves validation fatigue: each pillar has a *home page type*, not a "make sure every page does all four things" reviewer instruction.

## Growth frontmatter (canonical)

Every page carries this frontmatter. New fields are additive — old pages stay valid.

```yaml
---
title: <Human-readable title>
slug: <kebab-case>
layer: core | co | thesis
subject: <subject-slug>             # e.g. 01-ai, philosophy, decision-engineering
page_type: concept | author | arc | build | subject-overview
state: stub | drafted | reviewed | approved
authors_anchored: [pearl, kahneman] # who the page leans on; null for author pages themselves
feeds_de_pillar:                    # MANDATORY on concept pages; null elsewhere
  - miro-modeling | miro-inference | miro-reasoning | miro-optimization
  - systems-engineering | contextual-aiml | x-domain-context
arc_position:                       # only for pages embedded in an arc
  arc: <arc-slug>
  prev: <slug>
  next: <slug>
mvb_personas: [cs-student, applied-engineer, applied-researcher, frontier-researcher]
prereqs: [<concept-slug>, ...]
tags: []
updated: YYYY-MM-DD
---
```

`feeds_de_pillar` is the **convergence enforcement** field. The reviewer fails any `concept` page missing it. This makes the "everything converges on Decision Engineering" principle structural, not aspirational.

## Quality bar (the readability rule)

Every `concept` page must be **self-contained**: a motivated reader landing cold should come away understanding the topic without chasing links. Olah/Distill grade, not Wikipedia-stub-with-links. Length floor: 1500 words; expected sweet spot 2000–3500 words for meaty concepts. Cross-links enrich; they do not replace explanation.

This is non-negotiable. See `agents/memory/feedback_self_contained_pages.md`.

## Migration approach — Freeze + Parallel

User-confirmed 2026-05-26: build the new tree at `docs/curriculum-v2/` while leaving `docs/curriculum/` intact. The autonomous loop stays **frozen** until:

1. The skeleton in `curriculum-v2/` is built (this doc + empty directories + `index.md` stubs).
2. `agents/SCHEMA.md` is updated to v2 (5 page types, growth frontmatter, convergence rule).
3. Writer + reviewer prompts in `agents/server.py` are updated to read the new schema.
4. At least one **exemplar concept page** has been hand-built end-to-end under the new schema as proof.
5. The reviewer + critic panel has been re-tuned against the exemplar to confirm validation fatigue is resolved.

Only then does the loop resume — pointing at `curriculum-v2/`. The old `curriculum/` is preserved as reference until enough v2 pages exist to publish the cutover.

## What stays the same

- The 4 reader personas (cs-student / applied-engineer / applied-researcher / frontier-researcher).
- The source policy (arxiv, .edu, huggingface, official docs only).
- The arc / project / capstone framing from `faire_canon`.
- The closed-loop observer / supervisor / scheduler architecture.
- Budget control and the model-routing config.

## What changes

- Tree layout (flat 15 → 3-layer 10+5+1).
- Page-type taxonomy (1 → 5).
- Frontmatter (adds `layer`, `feeds_de_pillar`, `authors_anchored`, `page_type`).
- Reviewer enforcement (the convergence rule + readability bar become hard fails).
- Author pages become first-class generation targets.

---

*See `agents/SCHEMA.md` (v2) for the page-body templates. See `agents/memory/faire_structure_v2.md` for the live decision log.*
