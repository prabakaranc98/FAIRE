---
title: About Frontier Wiki
description: How this wiki is built, maintained, and why it exists
---

# About Frontier Wiki

Frontier Wiki is a 360° AI/ML knowledge base designed around one question: *what can you
actually build with this?* Not a course. Not a reading list. A structured knowledge substrate
where every page is the first step in someone's arc of work.

---

## The Arc-of-Work Philosophy

An **arc of work** is a deliberate sequence of concepts, builds, and insights that takes you
from "I've heard of this" to "I understand how this works and I've built something real with it."

Most learning resources answer "what is this?" The wiki answers that too — but every arc ends
with something you can make. Three to five nodes per arc have a **Minimum Valuable Build**: a
concrete, runnable project that produces a real artifact, not a verification exercise.

The VAE node in the Generative Stack arc earns an MVB because training a VAE from scratch is
the first moment latent-space intuition becomes tactile. DDPM earns one because it's the first
time you can generate images you didn't explicitly specify. UNet doesn't get one — it's the
mechanism inside DDPM, and its MVB lives at the DDPM node.

---

## How the Agent System Works

Every page in this wiki is generated and maintained by a local LangGraph editorial agent
pipeline. The pipeline runs on your machine — it is never deployed or automated without your
explicit command.

### The pipeline (per page)

```
START
  → load_persona       load the per-track expert persona (YAML)
  → read_stub          read existing content if the page already exists
  → research           Exa: foundational papers + 2024+ SotA + production deployments
                       HuggingFace: relevant models + datasets
  → plan               deliberate planning pass before writing
                       (core insight, opening analogy, MVB judgment, 3 essential papers,
                        specific open problem)
  → write_draft        Claude Opus 4.7 writes the full page using the plan
  → review             Gemini Pro reviews: schema, source policy, prose quality,
                       technical accuracy, MVB executability
                       assigns a confidence score 0.0–1.0
  → [if conf ≥ 0.8]  → write_file → commit → log_run → END
  → [if conf < 0.8,   
     count < 2]       → revise_draft → review (revision loop, max 2)
  → [if conf < 0.8,   
     count ≥ 2]       → flag_human_review → log_run → END
```

The **planning step** is what separates this from a naive "prompt → output" loop. Before the
writer touches the page, it answers five specific questions: what is the single most important
insight? what analogy opens the explanation? does this page earn an MVB? what are the three
essential papers? what is the specific unsolved problem to name? That plan is baked into the
writing prompt — the output is more coherent because the agent thinks before it writes.

### How commit decisions work

The agent makes the commit decision, not a static config flag. A page is committed to git only
if the reviewer's confidence score meets the threshold (default: 0.8). If `GIT_AUTO_PUSH=true`
in your `.env`, the commit is also pushed to origin — so a full run ends with the page live on
GitHub Pages without manual intervention.

This means: if the reviewer isn't confident, the page doesn't land in the wiki. Human review
is flagged explicitly in `agents/runs/runs.jsonl` and in the [changelog](system/changelog.md).

### Source policy

Every link in this wiki comes from:
- `arxiv.org` — papers
- `*.edu` — university course pages, lecture notes
- `huggingface.co` — model cards, datasets, spaces
- Official library documentation (PyTorch, JAX, etc.)

"In production" sections also allow official engineering blogs from top labs (ai.meta.com,
research.google, developer.nvidia.com/blog, etc.). No Medium. No Towards Data Science.
No personal blogs. No Substack. No Wikipedia. The reviewer agent enforces this on every run.

---

## The Self-Improving Loop

The wiki has two layers:

**Outer layer** — what you're reading now. The deployed wiki on GitHub Pages: polished,
verified, sourced.

**Inner layer** — the agent system that generates and maintains it. Runs locally on a schedule
(default: every 48 hours). Each cycle:

1. **Audit** — scans all pages for nested lists, banned domains, stale SotA (>6 months),
   missing sections. Writes issues to `agents/runs/last_audit.json`.
2. **Sprint** — parses `agents/sprints/current.md` (a Markdown checklist you edit). Runs the
   full pipeline for each unchecked item. Marks items done.
3. **Changelog** — appends a quality-delta table to this site: confidence before vs. after for
   every touched page.

To queue work for the next cycle, edit `agents/sprints/current.md`:
```markdown
- [ ] score-matching | 02-generative-modeling | core-concept | theoretical
- [ ] grpo | 06-reinforcement-learning | core-concept | applied
```

To start the server:
```bash
cd agents
uv run python server.py               # 48h cycle
uv run python server.py --run-now     # run one cycle immediately, then schedule
uv run python server.py --interval 1  # every hour (for testing)
```

The server exposes a local HTTP API (`http://localhost:8765`):

| Endpoint | Description |
|---|---|
| `GET /status` | Sprint queue, last audit, next scheduled run |
| `GET /audit` | Run quality scan now |
| `POST /trigger` | Force a full cycle immediately |
| `GET /runs` | Last N run records |
| `GET /changelog` | Full changelog |

---

## Running the agent

```bash
# Setup
cd agents && uv sync && cp .env.example .env  # fill in API keys

# Generate one page
uv run python generate.py generate \
  --topic diffusion-models \
  --track 02-generative-modeling \
  --page-type arc-entry \
  --depth-emphasis applied

# Improve an existing page
uv run python generate.py improve \
  --topic transformer \
  --track 07-attention-memory-reasoning

# Check coverage
uv run python generate.py status
```

---

## Models used

| Role | Model | Why |
|---|---|---|
| Writer | Claude Opus 4.7 | Richest prose, strongest academic reasoning |
| Reviewer | Gemini 3.1 Pro Preview | Independent review, 1M context, structured feedback |
| Research / Planning | Gemini 3.5 Flash | Fast + capable; cheaper for the pre-writing pass |
| Fallback | Claude Sonnet 4.6 | When Opus is unavailable |

All model calls route through [OpenRouter](https://openrouter.ai) for unified access and
fallback. The wiki itself is generated via local API calls — no cloud agent infrastructure,
no external orchestration.

---

*The GitHub star is the only engagement metric we collect.*
