---
skill: arc-context
description: Rules for arc pages — synthesis, capstone MVB, depth over breadth
applies_to: [write_draft, revise_draft]
triggers: [arc, depth, build, project, integration, capstone, end-to-end, pipeline, system]
---

# Skill: Arc Content

## What an arc is

An arc is a depth sequence: 3–7 curriculum nodes that together enable one meaningful project.
A curriculum page teaches a concept broadly. An arc uses several concepts narrowly and deeply
to build something that runs.

The distinction:
- Curriculum page on "Attention" → broad, covers scaled dot-product, multi-head, positional encodings, complexity, variants
- Arc page "Build a Character-Level Language Model" → narrow, uses only the attention variant needed, goes deep on the training loop, failure modes, and the exact working code

## Arc page structure

An arc page must have:

1. **What you'll build** — one sentence. A specific artifact, not a capability. "A working denoising diffusion model that generates 32×32 CIFAR-10 samples" not "An understanding of diffusion models."

2. **Why this arc** — the synthesis paragraph. What is non-trivial about combining these pieces? Where does understanding break down when you try to integrate them? What does the arc teach that the individual pages don't?

3. **The nodes** — table linking to the curriculum pages this arc uses. Each with a one-sentence note on what specifically it contributes to the build.

4. **The capstone MVB** — the full recipe (follows the same rules as curriculum MVB but integrates multiple concepts). This is the centerpiece of the arc. It earns more space than a curriculum MVB.

5. **What breaks in practice** — the hardest debugging section in the wiki. Real failure modes from integration, not from individual concepts. What you will spend 80% of your time on.

6. **Where to go deeper** — 2–3 pointers to push the build further. Not "learn more theory" — "here is the next harder version of the same build."

## The synthesis paragraph

This is the most important paragraph in the arc. It answers: "Why is this hard to put together?"

Bad: "This arc combines diffusion models with score matching and DDPM to enable image generation."
Good: "The gap between understanding score matching and running DDPM is the noise schedule. Every implementation makes a choice — linear, cosine, or learned — and that choice dominates the training curve more than model architecture. The arc is structured around understanding that choice empirically before touching the architecture."

## Capstone MVB rules

The capstone MVB follows all rules in the `mvb-recipe` skill, plus:
- Explicitly states which arc nodes it integrates
- Includes one "integration checkpoint" per node: a sanity check that the piece from that node is working before combining
- Uses a single codebase (no Jupyter notebooks split across files)
- Ends with a quantitative output: FID, accuracy, perplexity, or a clear qualitative description

## Depth markers

When writing arc content, apply these markers to calibrate depth:
- **Concept explanation**: ≤1 paragraph (reader already knows this from the curriculum page)
- **Integration detail**: 2–4 paragraphs (this is arc-specific, not covered elsewhere)
- **Code**: self-contained, runnable, includes assertions
- **Failure mode**: ≥1 concrete example with error message or wrong output shape

## What arc pages are NOT

- Not a table of contents for the curriculum
- Not a reading list ("read page A, then read page B, then read page C")
- Not a summary of what each curriculum page covers
- Not a motivation piece for why the topic matters

The arc page assumes the reader has read the curriculum pages. It starts where those pages end.
