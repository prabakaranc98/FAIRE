---
title: What FAIRE is
description: The canonical sense of this wiki — what we are building, for whom, and what counts as "right." Read by the editorial agents on every page they touch.
---

# What FAIRE is

> A wiki + curated reading list + nudge toward doing something — three layers stitched into one site, generated and maintained by editorial agents, sourced only from primary literature.

This page is the canonical brief. It is read by the supervisor and writer agents on every cycle. Editing this page changes how every future page gets written.

## The pedagogical bet — compounding learning

Wikipedia gives you knowledge but no journey — you read a page and you're done. roadmap.sh gives you a path but no compounding — it's a list of links to random courses, each one starting from scratch. paperswithcode gives you implementations but no pedagogy — you can run code without knowing why. Tutorials give you a moment of progress but no continuity — finish one, forget it, start another.

FAIRE's bet is the **compounding journey: curriculum → arcs → MVBs.**

```
   CURRICULUM           ARCS                  MVBs
   (range)       →      (depth)         →     (proof)
   ─────────            ──────                ──────
   Read a page.         Walk an opinionated   Build something.
   Build context.       sequence of 6–10      Each step's artifact
                        steps from a          is the input to the
                        starting concept      next step's build.
                        to a frontier         The final artifact
                        capability.           is the capstone.
   ↑                                          ↑
   ↑                                          ↑
   ↑─────── nudges back into curriculum ──────↑
            for the prereqs each step needs
```

What compounds:

- **Context compounds across curriculum pages.** Read VAEs → score matching → DDPM and by page 3 you don't re-explain what a latent variable is. Backlinks make this explicit.
- **Builds compound across arc steps.** Step 1's checkpoint is what step 5 loads. Step 7's latent space is the one step 8 trains a flow on. The dependencies are literal, not metaphorical.
- **Artifacts compound across arcs.** A finished latent-diffusion arc gives you the architecture you walk into the protein-structure arc with. Arc destinations connect to arc entries elsewhere.

This is the journey neither Wikipedia, roadmap.sh, paperswithcode, nor tutorial blogs provide. It is the single thing this wiki is built around.

## What we are — and what we aren't

The bet is one sentence: **a wiki that nudges you toward getting valuable builds done.**

That positions us against four nearby things — we borrow from each but become none of them:

| Adjacent thing | What we take from it | Where we depart |
|---|---|---|
| **Wikipedia** | Encyclopedic voice, neutral tone, structured pages | We are **curated, not exhaustive.** Primary sources only. No Medium, no Substack, no Wikipedia as a citation. |
| **roadmap.sh** | Visible learning paths — the "arc" concept | Every step ends in a **named MVB artifact**, not an abstract "learn X." Arcs are compounding builds, not reading checklists. |
| **paperswithcode** | Papers tied to working implementations | We add **intuition, math, persona-shaped MVBs, and open questions.** Implementations are linked, not pasted. |
| **A tutorial blog** | Concrete recipes the reader can follow | We are **reference, not tutorial.** No "we" or "let's." The reader does the building; the page points at what's worth building. |

What we explicitly are not:

- **Not tutorial-chasing.** We don't teach you how to install Python or copy-paste a notebook. We point at the right paper, the right library, the right ablation, and the right open question.
- **Not bookmark-stacking.** Three citation categories — seminal, test-of-time, current SotA — and nothing else. If you can't decide which of the three a link is, it doesn't go in.
- **Not a pile of information.** Every page ends with a directed nudge: one named artifact per persona, one specific "what's next" link. No "here are 50 more resources."

## The defining principle

> **Proof of work says "look how much I've done." An arc of work says "look how I think."**

Each page must support an arc of work — reproduce something real, extend it, then point at where someone could originate next. Pages that just summarize what exists for its own sake do not belong here.

## For whom

Seven readers, every page. The same article serves all of them by routing each persona to their entry section and giving each their own MVB variant.

| # | Reader | Comes to do | Time | Compute | What they build |
|---|---|---|---|---|---|
| 1 | Curious learner | Build a mental model | 30 min – 1 hr | Browser / free Colab | A notebook that *shows* the concept |
| 2 | CS student / tinkerer | Reproduce on a laptop | 4 hr – 1 day | RTX 3060/4070, M-series | A small training run that hits a target metric |
| 3 | Applied / production engineer | Ship at quality + latency | 3 days – 1 week | A10 / L4 / cloud | A real checkpoint served with measured latency |
| 4 | Applied researcher | Run one focused experiment | 3 days – 1 week | A100 × few | An ablation with a stated hypothesis and a comparison |
| 5 | Theory student | Derive from first principles | 4 hr – 1 day | CPU | A derivation verified on toy data; one plot |
| 6 | Frontier researcher | Find an open problem to push | 1 week+ | Varies | A probe of an open question, with a falsifier named |
| 7 | PM / decision-maker | Decide whether to invest | 30 min | None | (synthesis only, no build) |

**Each pivotal page carries up to 6 MVB variants — one per persona except the PM** — sharing the same concept underneath but diverging in artifact, time, and success metric.

## Three layers, one site

- **Curriculum** (`docs/curriculum/`) — one page per concept, organized across 15 tracks (10 canonical + 5 extensions). Range. No builds — those live in arcs. Each page backlinks to the arc steps that use it.
- **Arc Index** (`docs/arcs/{arc}/index.md`) — opinionated paths from a starting concept to a frontier capability. Chapters, curated readings, compounding-trajectory table.
- **Arc Step** (`docs/arcs/{arc}/step-NN-*.md`) — one build per page. Each step's artifact is literally what the next step loads. **MVBs (Minimum Valuable Builds) live here as milestones.**

## Source policy (enforced by the reviewer)

Citations come from:

- `arxiv.org` — papers and preprints
- `*.edu` — university lecture notes and course pages
- `huggingface.co` — model cards and dataset cards
- Official library docs (PyTorch, JAX, Diffusers, etc.)
- *"In production" sections only:* official frontier-lab engineering blogs

**Never cite:** Medium, Towards Data Science, Substack, personal blogs, Wikipedia.

## What "essential" means

When picking readings, three categories — and only three:

| Category | Criterion | How many per page |
|---|---|---|
| **Seminal** | Introduced the idea | 1–2 |
| **Test-of-time** | Cited for 5+ years, still load-bearing | 1–2 |
| **Current SotA** | Published 2024+ on arXiv, has benchmark numbers | 1–3 |

Anything outside these is filler.

## The nudge

Every full page ends with one specific invitation to do something next. Not "try implementing diffusion." Rather: "Train DDPM on CIFAR-10 with a 4-block UNet; the checkpoint is what Step 5 (DDIM sampler) loads."

## The four objectives

Every sprint, every experiment, every page should serve at least one (from PRINCIPLES.md):

1. **Discovery** — find what you didn't expect.
2. **Evidence** — ground a claim in numbers, plots, ablations.
3. **Inference** — produce a claim you'd be willing to be wrong about.
4. **Optimization** — change something based on what you learned, and by how much.

## Operating constraint

Run only **2 active arcs at a time** at the program level. Curriculum has no such cap — range is the goal there. The supervisor enforces this when proposing new arcs.

## Curriculum precedes arcs

Arcs are **derived from the curriculum that exists**, not invented from nothing.

1. **Curriculum enhancement is continuous.** Each cycle, stubs become real pages and existing pages get refreshed with new SotA. References get added over time, never removed.
2. **Arcs propose only when curriculum has range.** Once track coverage clears the threshold (default 60%), the supervisor scans the curriculum and proposes candidate arcs.
3. **Arc count is bounded by budget.** Cost per arc ≈ (1 index + N step pages) × $0.20. The supervisor proposes only what fits in `remaining_budget × 0.5`, leaving headroom for curriculum maintenance and review revisions.
4. **Arcs are optimized for MVB completion**, not arc count. Better to ship 1 arc end-to-end with 6 persona-tagged MVBs than start 3 arcs that stall halfway.

## Voice

Reference voice. Encyclopedic. Calm. No "we" or "let's." No marketing adjectives. No filler. If a sentence can be deleted without changing meaning, delete it.

## Where this comes from

The canonical statements live at **pracha.me/frontier/faire** (the FAIRE program) and **pracha.me/curriculum** (the 10 canonical tracks). This wiki is the agentic implementation of that canon.

The internal version of this brief, written for the agents to read on every cycle, is `agents/skills/faire-sense.md`. The two should stay in sync — when one changes, update the other.
