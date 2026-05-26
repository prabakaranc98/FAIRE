# Canonical Wiki Page Schema — v2

Every page generated or edited by agents must follow this schema. The reviewer
agent enforces compliance before commit.

> **v2 is the narrative walk-through schema.** Pages must read as essays that
> guide the reader through the field — not as bulleted shopping lists of
> definitions, tables, and references. See `docs/system/structure-v2.md` for
> the full tree design.

---

## Page types

| Type | Filename | Purpose |
|---|---|---|
| `subject-overview` | `_subject.md` → `index.md` at subject root | Where to start. Reading order, key authors, arc map. |
| `concept` | `concepts/<slug>.md` | Olah/Distill-grade self-contained encyclopedic article. The bread-and-butter. |
| `author` | `authors/<lastname>.md` | Person-anchored reading guide. Bio · arc of their thinking · key works · builds inspired by them. |
| `arc` | `arcs/<slug>.md` (or step files in same dir) | Roadmaps.sh-style learning path. |
| `build` | `builds/<slug>.md` | MVB recipe. Runnable, persona-tagged, real artifact at the end. |

This document specifies the `concept` page in detail (most common). Other page
types follow the same frontmatter + quality bar; their body templates are
documented separately as they mature.

---

## Frontmatter (canonical, v2)

```yaml
---
title: [Topic Name]
slug: [kebab-case-slug]
layer: core                              # core | co | thesis
subject: [NN-track-slug]                 # e.g. 01-ai, 08-causal-statistical-inference
page_type: concept                       # concept | author | arc | build | subject-overview
state: drafted                           # stub | drafted | reviewed | approved
authors_anchored: [pearl, kahneman]      # who the page leans on (kebab-case lastnames)
feeds_de_pillar: []                      # MANDATORY future field; leave [] for now
arc_position:                            # optional — only for pages embedded in an arc
  arc: [arc-slug]
  prev: [prev-slug]
  next: [next-slug]
mvb_personas: [cs-student, applied-engineer, applied-researcher, frontier-researcher]
prereqs: [topic-a, topic-b]              # other concept slugs the reader should know first
tags: []
updated: YYYY-MM-DD
has_mvb: true                            # true for pivotal pages
---
```

---

## The non-negotiable quality bar

Every `concept` page must satisfy **all three**:

1. **Self-contained.** A motivated reader landing cold should come away
   understanding the topic without chasing links. Substance lives ON the page.
2. **Narrative walk-through.** The page reads as a guided essay — hook, then
   territory, then mechanism, then frontier, then arc-forward. Tables and lists
   are enrichment, not skeleton.
3. **Convergent.** Every page must hint at how the concept plugs into the
   reader's larger arc of work (the MVB or the "Where to read next" pointer).

If a page passes a technical check but reads like a bulleted shopping list,
it fails. The reviewer must reject and the writer must rewrite.

Length: floor 1500 words. Expected sweet spot 2000–3500 words for meaty
concepts. The narrative form naturally fills this range because connective
prose can't be skipped.

---

## Page body — the 7-section narrative template

Use these section headings exactly. Each section opens with a prose lead-in;
none should start with a bullet list or table.

```markdown
# [Topic Name]

[**Hook — no heading, ~150 words.** Open with a question, a misconception, a
striking observation, or a metaphor. Frame why this matters and what the
reader will be able to think about by the end. NOT "Topic X is a method
that..." — open with the human problem this concept solves.]

## The territory

[**~300 words.** Where this concept sits in the field. What problem it
answers. The shape of the answer. What family of techniques it belongs to.
End with a transition into the mechanism: "How does it actually work?"
or similar.]

## How it works

[**~800–1500 words. This is where the page earns its weight.** The mechanism,
narrated. Math is embedded in prose: "the key idea is to rewrite the objective
as [eq] — the term on the left captures …, the term on the right is what
makes the optimization tractable." Annotate every variable inline.

Sub-headings (### Level 3) are allowed if the mechanism has natural stages,
but each sub-section must open with a transition sentence. NEVER open with a
bullet list.

Worked examples and intuitions belong here. Failure modes belong here.
"This is the key tension..." paragraphs belong here.]

## Where the field is now

[**~400 words.** Current SotA in narrative form. Name specific papers in
prose: "the 2024 result from Yu et al. (DAPO) showed that…". A table can
appear here AS EVIDENCE for a paragraph claim — never as the section spine.

Include one research frontier and one engineering/systems frontier. Be
specific: benchmark numbers, named models, named labs.]

## What's still open

[**~250 words.** The honest frontier. What's broken, contested, or unknown.
Specific enough that a researcher could write a paper on each open question.
Not "more research is needed" — name the question.]

## Where to read next

[**One paragraph, not a bibliography.** Inline links:

If you want the engineering side, → [[related-concept]] explains how this
scales in production. If you want the theory, → [[theory-concept]] gives the
proof sketch. If you want the historical arc, → [[arc-name]] walks the field
from its first paper to the current frontier.

This is the page's connective tissue to the rest of the wiki.]

## Build it

[**Numbered MVB recipe — the ONE section where list form is correct.**
Only present when `has_mvb: true`. See MVB Selection Policy below.

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

**Variants per persona (one per active mvb_personas entry):**
- **CS student:** [one-line tweak to the recipe — what they should do differently]
- **Applied engineer:** [the production-flavoured variant]
- **Applied researcher:** [the ablation/hypothesis variant]
- **Frontier researcher:** [the falsifier-driven extension]

> *If this build worked for you — a ⭐ on [GitHub](https://github.com/prabakaranc98/FAIRE) is the only signal we collect.*
```

That's the entire page. **Seven sections.** No "Core concepts" bullet
list. No "Essential reading" table at the bottom. No "Connected topics"
bullets. The reading flows.

Citations live INSIDE the prose (e.g. "DDPM (Ho et al. 2020)
[arxiv:2006.11239](https://arxiv.org/abs/2006.11239) showed that…"). The
prose itself is the bibliography, anchored to primary sources.

---

## The philosophy this schema encodes

**FAIRE is a wiki for building valuable things at the frontier of AI.**

Not a textbook. Not a Wikipedia stub-with-links. Not a tutorial chase. A
mentor in essay form who says: here's the territory, here's how it works,
here's what's open, here's what you can build, here's where this leads.

The reading experience is the product. If the page can't be read end-to-end
in one sitting and leave the reader changed, it has failed — even if every
citation is correct.

The Minimum Valuable Build is the closing act, not an appendix. It's the
moment where reading turns into doing.

---

## Multi-audience writing guidelines (for agents)

Pages must serve four reader types simultaneously. Use this checklist:

1. **Applied practitioner** (MS Data Science, industry engineer)
   - Wants: "What can I build with this TODAY?"
   - Lives in: the Build section + the engineering frontier paragraph in "Where the field is now"
   - Failure: "Use a diffusion model for image generation" — too vague
   - Success: "Fine-tune `stabilityai/stable-diffusion-2-1` with LoRA (~4GB VRAM, 1hr)"

2. **Curious generalist** (smart, limited ML background)
   - Wants: "What IS this and why do people care?"
   - Lives in: the hook + The territory + the first paragraph of How it works
   - Failure: Opening with "formally, given a probability distribution..."
   - Success: Opening with a concrete scenario that explains WHY the problem is hard

3. **Math/theory student**
   - Wants: "What are the actual equations? What's the proof sketch?"
   - Lives in: the deeper paragraphs of How it works (math embedded inline)
   - Failure: "The ELBO objective is \\[L = ...\\]" with no annotation
   - Success: Each variable annotated in the same sentence

4. **Frontier researcher**
   - Wants: "What are the open problems? What just changed?"
   - Lives in: Where the field is now + What's still open
   - Failure: "Recent work has shown improvements..."
   - Success: "DAPO (Yu et al. 2025) achieves 50 pts on AIME 2024 using decoupled clipping…"

---

## Prose rules (mandatory)

**NO BULLET DUMPS.** "Here are 5 things:" followed by 5 bullets is not
writing — it is a lazy outline. Transform every bullet dump into connected
prose. Allowed bullets: (a) the model/dataset/framework stack inside Build it,
(b) per-persona variants inside Build it, (c) the recipe steps (numbered) inside
Build it. That's it.

**NO NESTED LISTS.** Ever. A bullet inside a bullet is a writing failure.

**OPEN WITH THE HUMAN PROBLEM, not the technical definition.**
  BAD: "Diffusion models are latent variable models that..."
  GOOD: "Imagine you could take any photograph and gradually dissolve it into
        noise — then train a neural network to run that process in reverse.
        That's the core idea. The reason it works…"

**CONNECT PARAGRAPHS WITH CAUSALITY.** Show the reader how ideas follow from
each other. Use: "This is why…", "The consequence is…", "That insight led
directly to…", "Here's where it gets interesting…", "This is the key tension…"

**WRITE WITH EMPATHY.** The reader is smart but arriving new to this topic.
They don't need condescension; they don't need your assumptions about what they
know. They need a guide pointing out what matters.

**EMBED MATH IN PROSE.** Don't have a separate "Mathematical foundations"
section. Math appears where the explanation needs it, with every variable
annotated in the same sentence or the one immediately after.

---

## MVB selection policy

Not every page needs an MVB. Add `has_mvb: true` and the Build it section when:

- The concept is a "spine" node in a learning arc
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

For pages without MVB, the "Where to read next" paragraph should explicitly
point at the parent page that hosts the build: *"For a hands-on build with
this concept, see [[parent-concept]]."*

---

## Source policy (enforced by reviewer)

Every URL must be one of:
- `arxiv.org` — papers and preprints
- `*.edu` — university course pages, lecture notes
- `huggingface.co` — model cards, datasets, spaces
- Official library documentation (pytorch.org, jax.readthedocs.io, docs.anthropic.com, etc.)
- Official engineering blogs from frontier labs (only in "Where the field is now")

**Rejected sources:** Medium, Towards Data Science, personal blogs, Substack,
YouTube, Wikipedia.

---

## Stub format

When a page is created but agent content is pending, use:

```markdown
---
title: [Topic Name]
slug: [kebab-case-slug]
layer: core
subject: [NN-track-slug]
page_type: concept
state: stub
authors_anchored: []
feeds_de_pillar: []
mvb_personas: [cs-student, applied-engineer, applied-researcher, frontier-researcher]
prereqs: []
tags: []
updated: YYYY-MM-DD
has_mvb: false
---

# [Topic Name]

🚧 Agent-generated content pending. This concept sits in [[../index.md|<subject>]]
and will be developed into a full narrative walk-through (~2500 words) following
the v2 schema. See `agents/SCHEMA.md` for the template.
```

When the agent picks up a stub, it replaces the entire body with a v2 narrative
walk-through and bumps `state:` to `drafted` or `reviewed`.

---

## Reviewer enforcement summary

The reviewer must REJECT a page if any of the following are true:

| Failure | Why |
|---|---|
| Word count < 1500 | The narrative form can't fit; the page is a skeleton |
| Any of the 5 main sections missing (territory, how-it-works, field-now, still-open, read-next) | Skeleton check |
| Page opens with "## What it is" or any heading before the hook | Wrong template |
| Math equation without inline variable annotation | Theory-reader fails |
| Bulleted list under "Core concepts" or "Connected topics" headings | Old schema markers — page is on the wrong template |
| Bare URL in any link (no anchor text) | Source-policy fail |
| Any banned domain (medium, towardsdatascience, substack, wikipedia, youtube) | Source-policy fail |
| `has_mvb: true` but Build it section is generic ("follow the tutorial") | MVB integrity |
| Citation in prose without arxiv/edu/huggingface URL | Source-policy fail |
| Hook reads as a definition rather than a question/scenario/observation | Voice fail |

Pass = page is approvable. Send to git commit pipeline. State → `approved`.
Fail = page is sent back to writer with the failure list. Revision count incremented.
