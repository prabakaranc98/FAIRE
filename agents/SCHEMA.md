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

The one real thing a reader can build after this page. Not a verification that the algorithm runs —
something that changes what the reader can do or make. Honest about compute, honest about difficulty.

**What you're building:** [one sentence — specific output, real use case]

**Why this is valuable:** [honest — to the learner's trajectory AND to a real user or problem]

**Stack:**
- **Model:** [HuggingFace model ID](https://huggingface.co/...) — downloads count, real model card
- **Dataset:** [HuggingFace dataset ID](https://huggingface.co/datasets/...) — well-known, documented
- **Framework:** [specific library + key dependency]
- **Compute:** [GPU VRAM needed / free Colab tier / estimated time]

**The recipe:**
1. [Install + load — exact packages, one command]
2. [Data — specific preprocessing with why]
3. [Train/fine-tune — key hyperparameters, expected loss curve behavior]
4. [Evaluate — specific metric, expected ballpark number]
5. [What you now have — the artifact]

**Expected outcome:** [a model, a demo, a result table — something you can show or deploy]

**Stretch goals:**
- [One step toward publishable or production-deployable]
- [One alternative application of the same technique]

---

> *If this build worked for you — a ⭐ on [GitHub](https://github.com/prabakaranc98/FAIRE) is the only signal we collect.*

---

## What can you build next?
> *Your arc of work continues here.*

[1-2 sentences: what does building this unlock? What's the natural next question or build?]

**Go deeper on this concept:**
→ [[related-concept]] — [one sentence: what it adds to your understanding]

**Build a system with this:**
→ [[systems-or-applied-topic]] — [one sentence: how this scales or deploys]

**The arc this page belongs to:**
→ [Arc Name](../../arcs/arc-slug/index.md) — [one sentence on where this leads in the arc]

## Code & implementations
- [repo or huggingface link](URL) — what it implements and why it's worth reading

## Connected topics
- [[topic-name]] — how it connects
- [[topic-name]] — how it connects

## Further reading
- [Title](arxiv / university URL only) — one line on what it adds
```

---

## The philosophy this schema encodes

**FAIRE wiki is a wiki that nudges people toward building arc of work.**

Not a textbook. Not a reading list. A mentor who says: here's what's real, here's what you can build,
here's where this leads. Every page exists to answer one question for the reader: *what can I do with
this understanding?*

The **Minimum Valuable Build** is the cornerstone — not an optional section. It is the reason the
page exists. If the MVB is weak, the page has failed. If someone reads the page, builds the thing,
and it works — that page has succeeded.

The **"What can you build next?"** section closes the arc. No one should finish a page wondering where
to go. They should know: here's the next build, here's the arc you're in, here's where it leads.

The **GitHub star** is the only reward signal we collect. We ask for it once, after the MVB, without
pressure. If someone found the recipe and built something real — a star means "this worked." Nothing
else is asked.

**FAIRE = Frontier AI Research Encyclopedia.** It covers the 360° of contemporary AI from fundamentals
to frontier. Not "run a notebook" — a structured atlas that builds understanding arc by arc.

---

## Multi-audience writing guidelines (for agents)

Pages must serve four reader types simultaneously. Use this checklist before approving a draft:

1. **Applied reader** (MS Data Science, industry practitioner): Can they find the MVB in under 30 seconds? Does the recipe give them something runnable on a consumer GPU or free Colab?
2. **Foundational reader** (curious generalist): Does "What it is" explain the *why* before the *what*? Is there a concrete mental model before any math?
3. **Theoretical reader** (math/CS graduate): Is every LaTeX variable annotated? Are core concepts defined precisely before use? Is the derivation sketch honest about what it skips?
4. **Frontier reader** (researcher, PhD): Does "Current SotA" name specific papers with dates and metrics? Does "What's happening now" name specific open problems, not "more research is needed"?

**Writing with empathy — what this means in practice:**
- Meet the reader where they are: assume ML basics, assume no familiarity with *this specific topic*
- The first paragraph of "What it is" must be readable by a curious non-expert
- Every analogy should be grounded: don't say "like a filter" — say "like a high-pass filter on audio, removing low-frequency structure"
- Be honest about what's hard: "this is technically simple but took years to get right in practice because..."
- Be honest about limitations: "this works well for X but breaks on Y because..."
- The tone is confident but not authoritative: "here's what the evidence says" not "this is how it works"

**Anti-patterns to avoid:**
- Defining a term by restating its name ("diffusion models are models that use diffusion")
- Math without annotation (equations where no variable is explained)
- SotA that's vague ("recent work has shown...") without naming the work
- MVB that just says "follow the tutorial" — it must produce a real artifact
- "Further reading" that is just a bibliography dump — every item needs one honest line on what it adds
- "Connected topics" with no relationship description — "related" means nothing; explain how

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
