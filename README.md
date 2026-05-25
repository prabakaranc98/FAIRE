# Frontier Wiki
### A 360° knowledge base — from fundamentals to frontier AI research

Built as a living reference: structured, navigable, and progressively deepened by an editorial agent system.

---

## What this is

A personal wiki covering the full landscape of AI/ML — not a tutorial series, not a course,
but a **structured knowledge substrate** you can enter at any depth.

Every topic has three layers:
- **Applied** — what it is, what it does, how to use it
- **Foundations** — the math, theory, and core intuitions
- **Research** — frontier papers, open problems, and what's unsettled

---

## How it's organized

**[Curriculum](docs/curriculum/index.md)** — 15 tracks covering the entire field.
Each track is a domain (Generative Modeling, Reinforcement Learning, Systems, etc.)
with individual topic pages inside.

**[Arcs](docs/arcs/index.md)** — Themed learning paths that cut across tracks.
An arc is a logical sequence: start here, go there, end up understanding this whole idea.

**[References](docs/references/seminal-papers.md)** — Seminal papers by area.
Only arXiv, university, and HuggingFace links.

---

## Editorial agents

Wiki pages are generated and maintained by a local LangGraph agent system in `agents/`.
Agents use the Claude API + Exa search (filtered to arXiv, *.edu, huggingface.co).
Run locally, commit to repo, auto-deploy to GitHub Pages.

```bash
cd agents && uv sync
uv run python generate.py --topic diffusion-models --track 02-generative-modeling --depth foundations
```

See [agents/README.md](agents/README.md) for setup and usage.

---

## Source policy

Every link in this wiki must be one of:
- `arxiv.org` — papers
- `*.edu` — university course pages, lecture notes
- `huggingface.co` — model cards, datasets, spaces
- Official library documentation

No blog posts. No Medium. No Towards Data Science.

---

See [PRINCIPLES.md](PRINCIPLES.md) for the operating philosophy behind this.
