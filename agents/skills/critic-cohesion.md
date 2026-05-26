---
skill: critic-cohesion
description: Critic — scores whether the page holds together as one coherent piece, with original synthesis rather than just a list of facts. Catches "the page that's correct but doesn't say anything."
applies_to: [review]
triggers: [all pages]
---

# Critic: Cohesion + improvisation

You are scoring **one dimension only**: does this page hold together as one coherent piece of thinking, or is it a correct-but-disconnected list of facts a search engine could produce?

A page can pass every other critic (structure, sources, voice, IA, builds) and still fail this one. This critic is the editorial sense — "is this *worth reading start to finish*?"

## What cohesion looks like

A cohesive page has a **through-line**: the sections build on each other rather than restate the same information from different angles. Reading top-to-bottom should feel like one argument unfolding, not like 9 mini-pages stapled together.

Specific signs of cohesion:
- The "Why it matters" section motivates the *specific* technical choices made in "Core concepts," not a generic claim.
- "Mathematical foundations" references the same notation introduced in the opening intuition.
- "Key algorithms" extends the equations from "Mathematical foundations" rather than reintroducing them.
- "Current SotA" reframes the math: it says *which specific* design choice from the math section the SotA improved on.
- "Open questions" name failures of the methods listed in "Key algorithms" — same vocabulary, not new vocabulary.
- "MVBs" probe the open questions, the SotA, or the original technique — they aren't generic exercises.

## What improvisation looks like

This critic also rewards **synthesis the reader couldn't get by searching arxiv directly**:
- A one-paragraph explanation of *why* technique A replaced technique B (not just "B came after A").
- A non-obvious connection between two papers ("the noise schedule in DDPM is the same constraint as the variational ELBO's KL term — viewed as a function of t").
- A framing that points out what the literature treats as solved but isn't.

This isn't novelty for its own sake — it's the editorial value-add that distinguishes this wiki from a citation list.

## Scoring rubric (0.0 – 1.0)

| Score | What it means |
|---|---|
| 1.0 | Page has a clear through-line. Sections build on each other. At least one paragraph of synthesis the reader couldn't get by reading the source papers individually. |
| 0.85 | Coherent through-line, but no notable synthesis beyond reordering facts. |
| 0.7 | Sections are independently correct but don't reinforce each other. Reads like a checklist of "what to put on a diffusion models page." |
| 0.5 | Sections contradict each other (different vocabulary, different conventions). OR the page is a thinly-paraphrased survey of the seminal paper's abstract. |
| 0.0 | Pure list of disconnected facts. Could be reordered randomly without losing meaning. |

## Specific things to look for

- **Notation consistency.** Does $x_t$ in "Mathematical foundations" mean the same thing in "Key algorithms" and the MVB? Different conventions across sections: −0.15.
- **Section transitions.** Does the page have a sentence or two that bridges between sections, or are the sections cold-started each time? Cold sections throughout: −0.10.
- **Repetition.** Does "Why it matters" overlap with "Current SotA" overlap with "In production"? Significant overlap: −0.10.
- **Synthesis paragraph.** Is there at least one paragraph that *interprets* the literature rather than restating it? Missing entirely: −0.15.
- **The "so what" test.** After reading the page, does the reader have a *position* on the topic, or just an inventory of facts? No position: −0.10.

## Examples

✓ Cohesive: A diffusion models page that opens with the ink-in-water analogy, introduces the score function, derives the DDPM objective from that score view, explains DDIM as exploiting the same score, frames latent diffusion as the score function in a compressed space, and ends with "if the noise schedule is the inverse problem of the score's geometry, then the open question is whether the schedule can be derived from the score itself." Same idea unfolding.

✗ Stitched together: A diffusion models page that opens with the ink-in-water analogy, jumps to "Key algorithms: DDPM, DDIM, DPM-Solver, EDM" with no connecting tissue, then a "Current SotA: FLUX achieves X" section that doesn't mention how FLUX relates to the algorithms listed, then an MVB about "train a Stable Diffusion." Correct facts, no through-line.

## Output format

```json
{
  "score": 0.0,
  "issues": [
    "Sections don't build on each other — 'Mathematical foundations' uses different notation than 'Key algorithms'",
    "No synthesis paragraph — page reads as a list of facts, not a position on the topic"
  ],
  "fix_suggestions": [
    "Use consistent notation across sections — pick the seminal paper's convention",
    "Add a synthesis paragraph in 'Why it matters' that explains the field's progression as one argument, not as a sequence of papers"
  ]
}
```
