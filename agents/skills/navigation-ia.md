---
skill: navigation-ia
description: Information architecture rules — frontmatter arcs, What comes next, Connected topics
applies_to: [write_draft, revise_draft]
triggers: [navigation, arc, connected, curriculum, breadcrumb, what comes next, connected topics, see also]
---

# Skill: Navigation and Information Architecture

## Frontmatter arcs field

Every curriculum page that belongs to one or more arcs must declare it in frontmatter:

```yaml
---
title: Score Matching
track: 02-generative-modeling
arcs: [denoising-diffusion-arc, flow-matching-arc]
---
```

If the page belongs to no arc yet, omit the field entirely. Do not write `arcs: []`.

## Section: What comes next

Rules:
- At most 3 links
- Use real relative markdown paths — verify the file exists in the filesystem index
- One sentence per link describing the relationship, not the sequence
- Write the relationship from the reader's perspective: "what you can now unlock"

Good:
```markdown
## What comes next

- [DDPM](./ddpm.md) — implements the discrete training procedure that score matching enables
- [Flow Matching](./flow-matching.md) — generalizes the continuous-time perspective using arbitrary paths
```

Bad:
```markdown
## What comes next

- Step 1: Read DDPM
- Step 2: Then study Flow Matching
- Step 3: Finally, look at consistency models
```

Never write "What comes next" as a numbered sequence. It is not a learning path — it is a set of natural doors the reader can walk through.

## Section: Connected topics

Rules:
- At most 5 links
- Cross-track connections preferred (same-track connections belong in "What comes next")
- One sentence on the relationship: how the topics share structure, not that they're "related"
- Never write this as a bulleted list of link dumps — each entry must earn its place

Good:
```markdown
## Connected topics

- [Variational Autoencoders](../01-deep-learning/variational-autoencoders.md) — score matching and VAEs both learn data distributions; score matching avoids the explicit encoder
- [Contrastive Learning](../01-deep-learning/contrastive-learning.md) — both methods avoid computing partition functions; contrastive noise replaces score function estimation
```

Bad:
```markdown
## Connected topics

- [[diffusion-models]]
- [[normalizing-flows]]
- [[energy-based-models]]
```

## Arc back-link

When a page belongs to an arc, its body should contain a single line near the end of the introduction:

```markdown
This concept is used in the [Denoising Diffusion Arc](../../arcs/denoising-diffusion-arc.md).
```

Place this after the TL;DR table, not in the middle of an explanation.

## Breadcrumb hierarchy

The MkDocs nav reflects: **Track → Topic → Page**. Sections within a track are the topics. Pages within a section are the individual concepts. Do not create navigation that conflicts with this hierarchy.

## What to avoid

- Roadmap language: "first", "then", "next", "step", "follow along"
- Meta-commentary: "In this section we will explore..."
- Fake links: `[[wikilink]]` syntax, `[Page Title](#)`, or links that don't resolve to real files
- Orphan pages: every generated page should connect to at least one other via "What comes next" or "Connected topics"
