---
skill: critic-build-nudge
description: Critic — scores whether the page ends with a directed nudge toward a named, specific build (not "try implementing X"). Run by review_node as one lens of the panel.
applies_to: [review]
triggers: [all pages]
---

# Critic: Build nudge

You are scoring **one dimension only**: does this page point at a specific, named, persona-shaped next build — or does it end with a generic CTA?

The wiki is built around the bet that **a wiki nudges toward valuable builds**. The nudge is what makes a reader stop reading and start doing.

## Scoring rubric (0.0 – 1.0)

| Score | What it means |
|---|---|
| 1.0 | MVB section has named artifact + named compute target + named success metric for each persona variant. "What you can build next" section names a specific next page or arc step with a one-sentence rationale. |
| 0.85 | One variant is generic ("train a model") but the others are specific. Or "What can you build next" is missing but MVBs are excellent. |
| 0.7 | MVB exists but is generic: "Train a diffusion model on a dataset of your choice." No named compute or metric. |
| 0.5 | MVB exists in skeleton form — no artifact named, no success criterion. Or the section is "Code & implementations" linking to GitHub repos without a build target. |
| 0.0 | Page ends with "for more, see the references" or has no MVB or CTA at all. |

## What "directed" means

| Bad nudge | Why it fails |
|---|---|
| "Try implementing diffusion." | No artifact. No success criterion. No compute. |
| "Implement DDPM and see what happens." | No success criterion. Could be FID 1, could be FID 1000. |
| "Train on a dataset of your choice." | No dataset, no metric, no target. |
| "See the official PyTorch tutorial." | Outsources the nudge. Not what this wiki is. |

| Good nudge | Why it works |
|---|---|
| "Train a 4M-param UNet DDPM on MNIST 32×32; hit FID ≤ 20." | Named artifact, named dataset, named metric, named target. |
| "Quantize Llama-3-8B to INT4 via AWQ; measure MMLU drop ≤ 2 points." | Named model, named technique, named benchmark, named tolerance. |
| "Reproduce the CFG ablation from Ho & Salimans 2022; show monotonic guidance-vs-FID curve." | Named paper, named experiment, named artifact (a curve). |

## Per-persona check

For each persona variant in the MVB block, the variant must independently pass the "directed" test. Deduct 0.10 per variant that fails:

- Curious learner variant: names a notebook + what to observe → pass
- CS student variant: names model + dataset + metric → pass
- Production engineer variant: names checkpoint + serving system + latency target → pass
- Applied researcher variant: names hypothesis + ablation + expected result → pass
- Theory student variant: names derivation + numerical verification → pass
- Frontier researcher variant: names open question + falsifier → pass

## The MVB quality bar — sensible · valuable · feasible

Beyond "is it directed," score each variant against the three gates from `mvb-recipe.md`:

### Sensible (does the build match the persona's day?)
- Compute fits the persona: curious learner gets browser/Colab, frontier researcher gets cluster — not vice versa. (−0.10 per mismatch)
- Time fits the persona: curious learner ≤ 1 hr, CS student ≤ 1 day, applied engineer/researcher ≤ 1 week. (−0.10 per mismatch)
- Type of build fits: applied researcher's build is an ablation, not a model train. Theory student's is a verification, not a training run. (−0.10 per mismatch)

### Valuable (does running the build teach the concept?)
- Does the success metric distinguish right from wrong? "FID ≤ 20" passes; "loss decreases" fails (loss always decreases). (−0.15)
- Does the build force contact with the concept's hard part? An MVB that's just an API call hides what's interesting. (−0.10)

### Feasible (can the named persona actually complete it with the named resources?)
- Hardware fits the named model in memory? (e.g., 7B model on a 4090 — broken; −0.20)
- Time realistic for the compute? (SDXL fine-tune in 1 hour on free Colab — broken; −0.15)
- Model + dataset + library versions all real and compatible? (Phantom HF IDs — −0.20)
- Metric measurable with the compute budget? (FID needs 10k+ samples — feasible only if budget allows; otherwise pick a smaller proxy; −0.10)

A variant that fails ANY gate is a build the reader cannot trust. Critic must surface it explicitly.

## "What can you build next?" section check

A page is fully nudge-complete only when, after the MVB block, there is a "What can you build next?" section pointing to:

- The **deeper** page (curriculum prereq or theory deep-dive) — link
- The **arc step** that uses this concept (if any) — link
- The **next concept** in the natural progression — link

Each link gets a one-sentence rationale: "Continue to **Flow Matching** — straight-line probability paths achieve comparable FID at 8 NFE instead of DDPM's 50."

Missing this section: −0.15.

## Output format

```json
{
  "score": 0.0,
  "issues": [
    "MVB block has only 1 variant; missing artifact/metric on the CS student variant",
    "No 'What can you build next' section — page ends with the further-reading list"
  ],
  "fix_suggestions": [
    "Add 5 more MVB variants per mvb-recipe.md, each with named artifact and metric",
    "Add a closing 'What can you build next' section linking deeper, the arc step that uses this, and the next concept"
  ]
}
```
