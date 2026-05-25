# Canonical Wiki Page Schema

Every topic page generated or edited by agents must follow this schema exactly.
The reviewer agent enforces compliance before a page is committed.

---

## Frontmatter

```yaml
---
title: [Topic Name]
track: [NN-track-slug]
tags: [tag1, tag2, tag3]
depth: [applied | foundations | research | all]
prereqs: [topic-a, topic-b]
updated: [YYYY-MM-DD]
has_mvb: [true | false]   # true for pivotal pages with Minimum Valuable Build section
---
```

---

## Page body

```markdown
# [Topic Name]
> **TL;DR:** one sentence — what this is and why it matters at the frontier.

---

## For your reader type

| I am... | Start here | Goal |
|---|---|---|
| MS/applied practitioner | [Key algorithms](#key-algorithms--techniques) → [MVB](#minimum-valuable-build) | Build something that works |
| Curious generalist | [What it is](#what-it-is) → [Why it matters](#why-it-matters-at-the-frontier) | Build intuition |
| Math/theory student | [Core concepts](#core-concepts) → [Mathematical foundations](#mathematical-foundations) | Understand the mechanics |
| Researcher / frontier | [Current SotA](#current-sota) → [Seminal papers](#seminal-papers--test-of-time) | Know where the open problems are |

---

## What it is
[2-3 paragraphs. Clear, precise explanation — not a tutorial, not a definition dump.
What is this thing? What problem does it solve? What makes it work?
Write for someone who knows ML basics but has never seen this topic.]

## Why it matters at the frontier
[What open problems does this touch? Why do frontier labs care about this?
What has this unlocked or what is it currently blocking?
Mention specific labs, papers, or systems where relevant.]

## Core concepts
- **[Concept]** — concise definition
- **[Concept]** — concise definition
[5-8 key concepts, defined precisely in 1 sentence each. These are the vocabulary
a reader needs before the math makes sense.]

## Mathematical foundations
[LaTeX equations where relevant — key formulations, not derivations unless essential.
If the math is the point, show it. If it's incidental, summarize in words.
Annotate each equation: what does each variable mean? Why does this term exist?]

## Key algorithms / techniques
[Named algorithms with brief (1-2 sentence) descriptions of what they do and why they matter.
Order from foundational to modern. Include failure modes where known.]

## Essential reading
> These 2-4 papers are the minimum to understand this topic.

| Paper | Year | Authors | Why essential |
|---|---|---|---|
| [Title](arxiv link) | YYYY | Last et al. | What it establishes |

## Seminal papers & test-of-time
> Papers that defined the field and have held up over time.

| Paper | Year | Key contribution |
|---|---|---|
| [Title](arxiv or university link) | YYYY | What it introduced |

## Current SotA
> *Updated: YYYY-MM-DD*

[2-3 sentences on where the frontier is now. What's the best current result?
What system/paper represents the current state of the art? What's still open?
Include one research frontier, one engineering/systems frontier.]

## What's happening now
> *Research · Engineering · Systems*

**Research:** [1-2 sentences on the most active research direction — named papers or groups]

**Engineering & Systems:** [1-2 sentences on how this is being implemented at scale — production systems, optimizations, tooling]

**Open problems:** [1-2 specific open questions — not vague "we need more research" but "X is unsolved because Y"]

## Minimum Valuable Build
> *[Only on pivotal pages — omit if `has_mvb: false` in frontmatter]*

A practical recipe: something real you can build with what's on this page. Not a toy — something
with a genuine use case, built with tools that actually exist.

**What you're building:** [one sentence — specific, concrete project]

**Why this is valuable:** [to the learner / to a user / to the industry — make it honest]

**Stack:**
- **Model:** [HuggingFace model card](huggingface.co link) or describe the architecture
- **Dataset:** [HuggingFace dataset](huggingface.co/datasets link) or describe the data
- **Framework:** PyTorch / JAX / Diffusers / Transformers / etc.

**The recipe:**
1. [Step 1 — specific, one actionable sentence]
2. [Step 2]
3. [Step 3 — produces the valuable output]
4. [Step 4 — optional stretch]

**Expected outcome:** [what you have at the end — something you can show, deploy, or build on]

**Stretch goals:**
- [How to push beyond minimum — something publishable, deployable, or shareable]
- [Alternative application of the same technique]

## Code & implementations
- [repo or huggingface link](URL) — what it implements and why it's worth reading

## Connected topics
- [[topic-name]] — how it connects
- [[topic-name]] — how it connects

## Further reading
- [Title](arxiv / university URL only) — one line on what it adds
```

---

## Multi-audience writing guidelines (for agents)

Pages must serve four reader types simultaneously. Use this checklist before approving a draft:

1. **Applied reader** (MS Data Science, industry practitioner): Can they understand the Key algorithms section without reading the math? Does the MVB section give them something to run today?
2. **Foundational reader** (curious generalist): Does "What it is" explain the *why* before the *what*? Is there a concrete analogy or mental model?
3. **Theoretical reader** (math/CS graduate): Is the mathematical notation clean and annotated? Are the core concepts defined precisely before use?
4. **Frontier reader** (researcher, PhD): Does "Current SotA" name specific papers with dates? Does "What's happening now" identify open problems precisely — not vaguely?

**Anti-patterns to avoid:**
- Defining a term by restating its name ("diffusion models are models that use diffusion")
- Math without annotation (equations where no variable is explained)
- SotA that's vague ("recent work has shown...") without naming the work
- MVB that's just "follow the tutorial" — it must synthesize understanding into a build

---

## MVB (Minimum Valuable Build) selection policy

Not every page needs an MVB. Add `has_mvb: true` and the section when:
- The concept is central to a learning arc (a "spine" node)
- A real build exists with public models/data (HuggingFace, arXiv code)
- The build produces something with genuine value (not just "verify the algorithm runs")

**Pivotal pages that should always have MVB:**
- Diffusion models, Flow matching, VAEs — generative stack
- Transformer, BERT, GPT — architecture spine
- RLHF, DPO, PPO — alignment methods
- FlashAttention, KV cache — systems
- AlphaFold, FNO, PINNs — scientific AI
- Mamba / SSMs — architecture alternatives
- CLIP, LLaVA — multimodal

---

## Source policy (enforced by reviewer agent)

Every URL must be one of:
- `arxiv.org` — papers and preprints
- `*.edu` — university course pages, lecture notes
- `huggingface.co` — model cards, datasets, spaces
- Official library documentation (pytorch.org, jax.readthedocs.io, docs.anthropic.com, etc.)

**Rejected sources:** Medium, Towards Data Science, personal blogs, Substack, YouTube, Wikipedia.

---

## Stub format

When a page is created but agent content is pending, use:

```markdown
---
title: [Topic Name]
track: [NN-track-slug]
tags: []
depth: foundations
prereqs: []
updated: YYYY-MM-DD
has_mvb: false
---

# [Topic Name]
> **TL;DR:** [one-sentence description]

> 🚧 Agent-generated content pending. See [track index](../index.md) for context.

## What it is
[stub]

## Why it matters at the frontier
[stub]

## Essential reading
[stub]

## Connected topics
[stub]
```

---

## Arc page schema

Arc pages follow a different, more narrative schema. See `ARC_SCHEMA.md`.
