# Frontier Wiki

A 360° AI/ML knowledge base built around one question: *what can you actually build with this?*

Not a tutorial series. Not a reading list. A structured knowledge substrate where every page
is the first step in someone's arc of work — from the intuition, through the math, to a
concrete artifact.

---

## What is an Arc of Work?

An arc of work is a deliberate sequence of concepts, builds, and insights that takes you from
"I've heard of this" to "I understand how it works and I've built something real with it."

Each arc has ~20–28 ordered concept nodes. Each node is a page. Three to five nodes per arc
have a **Minimum Valuable Build** — a concrete, runnable project that produces a real artifact.
The MVB on the VAE node lets you train a latent-space model from scratch. The MVB on the
Diffusion node lets you train a conditional image generator. Each build unlocks a new class
of artifact you couldn't make before. Sub-concept pages (UNet, DDPM sampler) don't get their
own MVB — they feed the build at the parent page.

---

## How to enter the wiki

**[Curriculum](docs/curriculum/index.md)** — breadth first. Fifteen tracks covering the entire
field. Enter at any topic. See how it connects. Find your footing.

**[Arcs](docs/arcs/index.md)** — depth first. Focused, sequential journeys with a clear
purpose. Read the arc page first — it's the map.

**[System](docs/system/status.md)** — live generation status: which pages are agent-generated,
reviewer confidence scores, changelog.

---

## How the system is built

### Agent pipeline

Every page is generated and maintained by a local LangGraph editorial agent. The pipeline runs
on your machine — never deployed, never automated without your approval.

```
START
  → load_persona       per-track expert persona (YAML)
  → read_stub          existing content if the page already exists
  → research           Exa: foundational papers + 2024+ SotA + production deployments
                       HuggingFace: relevant models + datasets
  → plan               deliberate planning pass (RESEARCH_MODEL — fast):
                         core insight · opening analogy · MVB yes/no ·
                         3 essential papers · specific open problem
  → scratch            working-memory compiler (RESEARCH_MODEL):
                         verified citations · key equations · production examples ·
                         MVB stack · opening scenario · the open problem
                         (writer never sees raw search results — only this fact sheet)
  → write_draft        3 sequential WRITER_MODEL calls:
                         chunk 1 — frontmatter + TL;DR + What it is + Why it matters
                         chunk 2 — Core concepts + Math + Algorithms + Reading + SotA + In production
                         chunk 3 — MVB + Code + What comes next + Connected topics
                         each chunk reads plan + scratch_pad + previous chunks
  → review             REVIEWER_MODEL: schema · source policy · prose quality ·
                         technical accuracy · MVB executability → confidence 0–1
  → [conf ≥ 0.8]  → write_file → commit → log_run → END
  → [conf < 0.8, rev<2] → revise_draft → review
  → [conf < 0.8, rev≥2] → flag_human_review → log_run → END
```

### Why this architecture

**Plan before you write.** The planning step forces the agent to answer five specific questions
before prose generation begins: what's the core insight, what analogy opens it, does this page
earn an MVB, which three papers are essential, what is the specific open problem. Agents that
skip this produce vague, generic output.

**Scratch pad as working memory.** Raw search results (8 papers × 600 chars + SotA highlights
+ production summaries) would flood the writer context. The scratch node compiles this into a
clean fact sheet: verified citations only, typed-out equations with annotations, production
examples with scale numbers, the exact HuggingFace model/dataset IDs for the MVB. The writer
reads the fact sheet, not the raw data.

**Chunked writing for coherence.** A full wiki page (~5000 tokens of markdown) in one LLM call
truncates and loses coherence across sections. Three chunks of ~1500 tokens each, each reading
the previous chunks, produces a page where the prose in "Why it matters" references the analogy
set up in "What it is," and the MVB uses the exact model IDs established in the scratch pad.

**Confidence-gated commit.** The reviewer scores 0–1. Pages commit to git automatically when
confidence ≥ 0.8 (configurable). If `GIT_AUTO_PUSH=true`, the commit is also pushed —
meaning a full run ends with the page live on GitHub Pages.

### Self-improving loop (48h cycle)

```
uv run python server.py          # start — 48h cycle
uv run python server.py --run-now  # run one cycle immediately, then schedule
```

Each cycle: **audit** (scan all pages for quality regressions) → **sprint** (run pipeline on
items in `agents/sprints/current.md`) → **changelog** (confidence delta per touched page).

API at `http://localhost:8765`: `GET /status` · `POST /trigger` · `GET /changelog`

---

## Setup checklist

```bash
# 1. Clone and navigate
git clone https://github.com/prabakaranc98/FAIRE.git && cd FAIRE

# 2. Install agent dependencies
cd agents && uv sync

# 3. Configure environment
cp .env.example .env
# Fill in: OPENROUTER_API_KEY, EXA_API_KEY
# Models are pre-configured: claude-opus-4.7 writer, gemini-3.1-pro-preview reviewer

# 4. Generate a page
uv run python generate.py generate \
  --topic score-matching \
  --track 02-generative-modeling \
  --page-type core-concept \
  --depth-emphasis theoretical

# 5. Check coverage
uv run python generate.py status

# 6. Start the self-improving server (optional)
uv run python server.py --run-now

# 7. Preview the wiki locally
cd .. && mkdocs serve
```

---

## Source policy

Every link in this wiki comes from:
- `arxiv.org` · `*.edu` — papers and lecture notes
- `huggingface.co` — model cards, datasets, spaces
- Official library docs (pytorch.org, jax.readthedocs.io, etc.)
- `distill.pub` · `lilianweng.github.io` — cited for attribution, never reproduced verbatim

"In production" sections also allow official engineering blogs from top labs.
No Medium. No Towards Data Science. No personal blogs. No Wikipedia.
The reviewer enforces this on every run.

---

## Models

| Role | Model | Context |
|---|---|---|
| Writer | `anthropic/claude-opus-4.7` | 1M |
| Reviewer | `google/gemini-3.1-pro-preview` | 1M |
| Research / Planning / Scratch | `google/gemini-3.5-flash` | 1M |
| Fallback | `anthropic/claude-sonnet-4.6` | 1M |

All model calls route through OpenRouter. The wiki is generated via local API calls —
no cloud agent infrastructure, no external orchestration.

---

See [PRINCIPLES.md](PRINCIPLES.md) for the operating philosophy.
See [docs/about.md](docs/about.md) for the full architecture walkthrough.
