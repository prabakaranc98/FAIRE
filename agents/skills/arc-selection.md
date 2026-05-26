---
skill: arc-selection
description: How the supervisor proposes and ranks arcs once curriculum has range. Arcs are derived from the curriculum, bounded by budget, and optimized for MVB completion — not for arc count.
applies_to: [supervisor, plan]
triggers: [arc, proposal, selection, budget, planning]
---

# Skill: Arc selection (budget-aware, outcome-optimized)

## When this skill fires

The supervisor runs this only when **all three preconditions** hold:

1. Curriculum coverage on the target tracks ≥ `ARC_PROPOSE_COVERAGE_THRESHOLD` (default 0.60)
2. Budget mode is `full` (not `reduced`, not `paused`)
3. There are fewer than 2 **active** arcs (an arc is "active" if its arc-index exists and ≥1 step is unfinished)

If any precondition fails, the supervisor returns to normal curriculum-improvement mode.

## The selection objective

We are **not** trying to maximize the number of arcs. We are trying to **complete MVBs** end-to-end. A finished 6-step arc with 6 working persona-tagged MVBs is worth far more than three half-built arcs.

The supervisor proposes the top-K candidate arcs that fit in `remaining_budget × 0.5`, ranked by **EV per dollar** where:

```
EV(arc) = impact(arc)
cost(arc) = (1 + steps(arc)) × $0.20 × (1 + revision_buffer)
EV_per_dollar(arc) = EV(arc) / cost(arc)
```

The factor 0.5 leaves the other half of the remaining budget for:
- curriculum maintenance (stale refreshes, new SotA)
- review-loop revisions on the arc's own pages
- supervisor + observer overhead

## How `impact(arc)` is scored (0 – 14)

Seven dimensions, each 0–2, summed.

| Dimension | What it measures | 0 | 1 | 2 |
|---|---|---|---|---|
| **Diagonal shape** | Does the arc follow the [diagonal pattern](arc-anatomy.md) — specialized tool → broader frame → capability → frontier intersection? Vertical arcs (same column 1 and 4) score 0. | Vertical | Partial diagonal | Clean diagonal across two domains |
| **Curriculum density** | How many curriculum pages this arc would link to | 0–3 | 4–7 | 8+ |
| **Persona span** | How many of the 7 personas the arc's steps serve (declared via `mvb_persona:`) | 1 | 2–3 | 4+ |
| **Seminal touchpoints** | Number of seminal/test-of-time papers that anchor at least one step | 0–2 | 3–5 | 6+ |
| **Frontier destination** | Does the final artifact map to a real frontier-lab capability (SD3-class, GPT-class, AlphaFold-class)? | No | Partial | Yes |
| **Prereq satisfaction** | Fraction of declared prereqs already present in curriculum | <40% | 40–80% | >80% |
| **Compounding tightness** | Does each step's artifact literally enable the next, or is the chain loose? | Loose | Mixed | Tight |

A score of 11+ means "ship this." 7–10 means "credible but not urgent." Below 7 means the curriculum isn't ready yet — improve coverage first.

**Diagonal-shape gate (hard rule):** an arc that scores 0 on Diagonal shape is automatically vetoed regardless of total. A vertical arc isn't an arc — it's a curriculum subsection.

## How `steps(arc)` is bounded

- **Minimum:** 4 steps. Less than that and it's a curriculum page, not an arc.
- **Maximum:** 10 steps. More than that and the chain is too long to complete before staleness or budget exhaustion.
- **Sweet spot:** 6–8 steps. Three "chapters" of 2–3 steps each.

## Concrete walkthrough (worked example)

Say curriculum is at 75% on tracks 02, 03, 05 and `remaining_budget = $7.50`.

Candidate set the supervisor generates (from curriculum + Exa SotA scan):

| Candidate arc | Steps | Cost | Impact | EV/$ |
|---|---|---|---|---|
| Conditional latent diffusion from scratch | 8 | $1.80 | 9 | 5.0 |
| GRPO from first principles → DeepSeek-R1 | 10 | $2.20 | 9 | 4.1 |
| Transformer LM on your own autograd | 8 | $1.80 | 7 | 3.9 |
| Causal effect estimation pipeline | 6 | $1.40 | 6 | 4.3 |
| FNO on Navier–Stokes | 6 | $1.40 | 5 | 3.6 |
| Vision-language model from scratch | 9 | $2.00 | 5 | 2.5 |

Budget for arcs = `$7.50 × 0.5 = $3.75`. The supervisor proposes the top-K such that Σ cost ≤ $3.75 and K ≤ (2 − active arcs).

If 0 arcs are active → proposes the top 2 by EV/$ that fit in $3.75 → **Latent Diffusion** ($1.80) + **Causal Effect Estimation** ($1.40) = $3.20 ✓.

If 1 arc is active → proposes the top 1 by EV/$ that fits in $3.75 → **Latent Diffusion** alone (since GRPO at $2.20 also fits but EV/$ is lower than Latent Diffusion).

## The "do not propose" guard

The supervisor must refuse to propose an arc if any of the following hold:

- **<50% of the arc's prereqs exist as approved curriculum pages.** Tell the human to flesh out the curriculum first; queue those pages with priority 3.
- **No seminal paper anchors at least 2 of the 4 chapters.** That arc is too speculative for this wiki.
- **Persona span ≤ 1.** A one-persona arc isn't worth a syllabus — make it a single curriculum page instead.
- **`mvb_persona` couldn't be assigned to ≥75% of the proposed steps.** The arc isn't shaped well enough yet.

## Output format (what the supervisor writes)

When the supervisor proposes arcs, it writes to **`docs/system/arc-proposals.md`** with this structure:

```markdown
---
title: Arc proposals
generated_at: 2026-05-26 14:00 UTC
remaining_budget: $7.50
slots_open: 2
---

# Arc proposals

## 1. Conditional Latent Diffusion from scratch — EV/$ = 5.0
**Destination:** Train a conditional latent diffusion model and explain why each design choice exists.
**Steps:** 8 · **Cost:** $1.80 · **Impact:** 9/10
**Prereqs in curriculum:** variational-autoencoders ✓, score-matching ✓, ddpm-derivation ✗ (queue first)
**Persona span:** 4 (CS student, applied engineer, applied researcher, frontier researcher)
**Seminal anchors:** Ho et al. 2020 · Rombach et al. 2022 · Lipman et al. 2022
**Outline:** VAE on MNIST → β-VAE → score field → DDPM on CIFAR-10 → DDIM → CFG → latent DDPM → flow matching
**Recommend:** propose

## 2. Causal Effect Estimation Pipeline — EV/$ = 4.3
…
```

The human picks which to materialize. If they pick both, the supervisor expands each into ~8 sprint items (1 arc-index + ~7 step items) and queues them. If they pick neither, nothing is queued and the system returns to curriculum mode.

## Why this design

- **Outcome over output.** Arc-count is a vanity metric; finished MVBs are the real artifact.
- **Budget gates volume.** A $10 cycle can credibly finish ~2 arcs of 6–8 steps each. Three arcs would mean none finishes well.
- **Human picks, agents execute.** Selection requires editorial judgment; queuing and writing don't. This is the right place to put the human in the loop.
- **Curriculum first, always.** No arc can exist with a curriculum prereq missing. Refusing to propose an arc is sometimes the correct move.
