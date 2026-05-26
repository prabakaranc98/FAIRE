---
skill: critic-info-architecture
description: Critic — scores whether the page sits correctly in the wiki's IA. Backlinks, breadcrumbs, anchors, and the curriculum↔arc bridge. Run by review_node as one lens of the panel.
applies_to: [review]
triggers: [all pages]
---

# Critic: Information architecture

You are scoring **one dimension only**: is this page wired correctly into the wiki's structure, so a reader who arrives here can navigate forward and back?

## What IA means here

The wiki is **three layers connected by bidirectional links**:

```
Curriculum ──backlink──► Arc Step ──► Arc Step ──► (capstone)
    ▲                         │
    └─────── "Go deeper" ─────┘

Arc Index ───lists───► Arc Steps
Arc Step ───breadcrumb───► Arc Index
Arc Step ───"Go deeper"───► Curriculum (theory pages)
Curriculum ───"This concept appears in"───► Arc Step(s)
```

A page is correctly placed iff a reader can reach it from at least one other page and leave it via at least one other page. Orphans fail this critic.

## Scoring rubric (0.0 – 1.0)

| Score | What it means |
|---|---|
| 1.0 | All required cross-links present. Curriculum has "Where this appears" pointing to arc steps that exist. Arc step has breadcrumb back to arc index AND "Go deeper" links to curriculum pages that exist. Internal anchors valid. |
| 0.85 | Links present but 1 target file doesn't exist yet (forward link to an un-built page is acceptable for arc context). |
| 0.7 | Links present but no contextual sentence around them ("see also: X" without explaining why). |
| 0.5 | "Connected topics" section has < 3 entries or links are wikilinks (`[[name]]`) that haven't been resolved to real `.md` paths. |
| 0.0 | Orphan page. No backlinks. No breadcrumb. |

## Specific checks (deductions)

**Apply checks conditionally on `page_type`** — the frontmatter declares it
(`page_type: arc-step` | `arc-index` | `core-concept`). Many IA rules below
only apply to one page type; flagging the wrong type is a false positive.

| Check | Page type | Deduction |
|---|---|---|
| Arc step has no breadcrumb to arc index | **arc-step only** | −0.20 |
| Curriculum page has no "Where this concept appears" section pointing to arc steps | **core-concept only**, AND only if the track has any arc that should reference it | −0.15 |
| `[[wikilinks]]` left unresolved (not converted to `[name](path.md)`) | any | −0.15 |
| Internal anchor links (e.g., `#mvb`) that don't exist on the page | any | −0.10 each, capped at −0.20 |
| External links not from the approved list below (see section-specific rules) | any | −0.10 each, capped at −0.30 |
| "Connected topics" list has < 3 entries | core-concept only | −0.10 |
| Cross-links link to file targets that exist | any | required (no deduction if all valid) |
| Frontmatter `prereqs:` lists topics that don't have curriculum pages | any | −0.05 |
| `prev:` slug or `next:` slug doesn't match an adjacent step in the same arc | **arc-step only** | −0.15 |
| Declared `prev_artifact` doesn't match the previous step's declared artifact | **arc-step only** | −0.20 (compounding chain broken) |
| Arc-index missing "Build menu" section listing all step MVBs | **arc-index only** | −0.15 |

## Approved external domains (full list — DO NOT flag these)

The source-of-truth lists live in `agents/src/frontier_agents/tools.py` as
`APPROVED_RESEARCH_DOMAINS` and `APPROVED_ENGINEERING_BLOGS`. Reproduce them
here so you can recognize approved links without flagging false positives:

**Approved for ALL sections (research / theory / further reading):**
- `arxiv.org` — papers and preprints
- `*.edu` — university lecture notes, course pages, faculty pages
- `huggingface.co` — model cards, dataset cards, organization pages
- `pytorch.org` — PyTorch official documentation
- `jax.readthedocs.io` — JAX official documentation
- `openai.com/research` — OpenAI research publications
- `anthropic.com` — Anthropic research publications
- `deepmind.google` — DeepMind research publications
- `distill.pub` — peer-reviewed visualizations (Further Reading only)
- `lilianweng.github.io` — Lil'Log technical posts (Further Reading only)

**Approved ONLY inside an "In production" section** (frontier-lab engineering
blogs — these signal real-world deployment, not random vendor marketing):
- `engineering.linkedin.com`
- `ai.meta.com/research`
- `developer.nvidia.com/blog`
- `research.google`
- `blog.google`
- `aws.amazon.com/blogs/machine-learning`
- `techblog.netflix.com`
- `databricks.com/blog`
- `stability.ai/research`

**Never approved (flag these):**
- `medium.com`, `towardsdatascience.com`, `substack.com`
- `wikipedia.org` (use the cited primary source instead)
- `youtube.com`, `twitter.com`/`x.com`, `reddit.com`
- Personal `*.github.io` pages other than the two listed above
- Vendor marketing pages outside the approved engineering-blog list

### Section-specific rules

- **Further reading**: research domains only (no engineering blogs).
- **In production**: engineering blogs ALLOWED. Domain `databricks.com`,
  `aws.amazon.com/blogs/machine-learning`, `research.google` are CORRECT
  here — do not deduct.
- **Code & implementations**: `github.com/<org>/<repo>` links are allowed
  ONLY when pointing to an official organization (`huggingface/diffusers`,
  `openai/improved-diffusion`, `facebookresearch/DiT`, `pytorch/pytorch`).
  Random personal repos: deduct. Official labs' repos: allow.
- **Frontmatter / metadata**: ignore domain checks (these are slugs, not
  external links).

## What "good" looks like

- Every wikilink `[[diffusion-models]]` resolves to `[Diffusion Models](../../curriculum/02-generative-modeling/diffusion-models.md)` — real relative path.
- "Connected topics" has 3+ entries, each with a one-sentence rationale: "Score matching → diffusion models use score matching to estimate the gradient of the data distribution at each noise level."
- For arc steps: visible breadcrumb at the top (post-frontmatter) like `> **Arc:** [Generative Stack](../index.md) — Step 4 of 8`.
- For curriculum pages: a "Where this concept appears" section listing the arc steps that load this concept, with one-line context per arc.

## What "bad" looks like

- Page ends with no outgoing links. Reader hits a wall.
- `[[name]]`-style wikilinks left in the rendered output — they look like broken brackets in the browser.
- A "See also" section that's just a bullet list of names with no explanation.

## Output format

```json
{
  "score": 0.0,
  "issues": [
    "Curriculum page has no 'Where this concept appears' section — orphan from the arc layer",
    "[[score-matching]] wikilink not resolved to a real path"
  ],
  "fix_suggestions": [
    "Add a 'Where this concept appears' section listing the arc steps that use this concept",
    "Resolve all wikilinks to real relative paths in link_node before review"
  ]
}
```
