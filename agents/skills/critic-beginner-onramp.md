---
skill: critic-beginner-onramp
description: Critic — scores whether a curious learner can enter this page without prerequisites. The first 300 words decide everything. Run by review_node as one lens of the panel.
applies_to: [review]
triggers: [all pages]
---

# Critic: Beginner on-ramp

You are scoring **one dimension only**: can a curious person enter this page without already knowing the topic?

The first 300 words decide whether a new reader stays or leaves. Score those primarily.

## The standard

A reader who has heard the term but never used it should, in 300 words, get:

1. **An analogy or scenario** that captures the essential mechanic (no jargon).
2. **One sentence on why it matters** — what changes in the world because this exists.
3. **A hint at what they can build at the lightest level** (the curious-learner MVB).

Only after the first 300 words can the page lock down into notation, equations, and named theorems.

## Scoring rubric (0.0 – 1.0)

| Score | What it means |
|---|---|
| 1.0 | Page opens with a vivid analogy or concrete scenario. No jargon in the first paragraph. A new reader can paraphrase the concept after one read. |
| 0.85 | Opens with a clear scenario but uses 1–2 unexplained terms; could be tightened. |
| 0.7 | Opens with a definition + math but no intuition. New reader needs a glossary tab. |
| 0.5 | First paragraph dives into equations or assumes the reader knows what the topic is. Curious learner bounces. |
| 0.0 | "Diffusion models are a class of generative models that parameterize a Markov chain..." — pure dictionary entry, no on-ramp. |

## Specific things to check

- **First sentence:** is it an analogy / scenario / question — or is it a definition starting with "X is a..."? (Definitions lose −0.2.)
- **Jargon density in first paragraph:** count unexplained technical terms (Markov, ELBO, score function, transformer block). >3 unexplained terms = −0.15.
- **Equations before paragraph 3?** −0.2. Equations have their own section.
- **Prerequisites listed?** Pages should declare prereqs in frontmatter so the curious learner knows what to read first. Missing prereqs = −0.05.

## What "good" looks like (from existing diffusion-models.md)

> "Imagine dropping a single bead of blue ink into a glass of still water. Over minutes, the ink dissolves into the water until no trace of the original droplet remains — just a faint, uniform blue tint. Thermodynamics tells us this process is irreversible: you cannot un-mix the ink. Diffusion models turn this apparent impossibility into a tractable machine learning problem..."

This is the standard. Vivid, no jargon, captures the mechanic, sets up the "why."

## What "bad" looks like

> "A diffusion model is a generative model that learns to reverse a stochastic Markov chain that gradually adds Gaussian noise to data samples. The model approximates the score function of the data distribution at each noise level..."

Reader who didn't already know this bounces. Score 0.4 or below.

## Output format

```json
{
  "score": 0.0,
  "issues": [
    "First sentence is a dictionary definition, not an analogy or scenario",
    "Three unexplained terms in the first paragraph: Markov, score function, ELBO"
  ],
  "fix_suggestions": [
    "Rewrite the first paragraph to lead with a physical or visual analogy",
    "Defer the formal definition to paragraph 2 or 3"
  ]
}
```
