# Frontier Wiki

A 360° AI/ML knowledge base — from fundamentals to frontier research, built around one question:
*what can you actually build with this?*

Not a tutorial series. Not a reading list. A structured knowledge substrate where every page
is the first step in someone's arc of work.

---

## What is an Arc of Work?

An arc of work is a deliberate sequence of concepts, builds, and insights that takes you from
"I've heard of this" to "I understand how it works and I've built something real with it."

The wiki is organized around arcs. Each arc has ~20-28 ordered concept nodes. Each node is
a page that answers: what is this, why does it matter at the frontier, and what can you build
with it? Three to five nodes per arc have a **Minimum Valuable Build** — a concrete,
runnable project that produces something real, not just a verification exercise.

The MVB on the Generative Stack arc's VAE node lets you train a latent-space model from scratch.
The MVB on the Diffusion node lets you train a conditional image generator. The MVB on the Latent
Diffusion node lets you fine-tune Stable Diffusion on a custom domain. Each build unlocks a new
class of artifact you couldn't make before.

---

## How to enter the wiki

**[Curriculum](docs/curriculum/index.md)** — breadth first. Fifteen tracks covering the entire
field: Generative Modeling, Reinforcement Learning, Representation Learning, Causal Inference,
Systems, and more. Enter at any topic. See how it connects. Find your footing.

**[Arcs](docs/arcs/index.md)** — depth first. Focused, sequential journeys with a clear purpose.
The arc tells you why this sequence, what you're building toward, and what you know at each step.
Read the arc page before diving into topic pages — it's the map.

---

## The agent system

Wiki pages are generated and maintained by a local LangGraph editorial agent system in `agents/`.
Agents run on your machine only — never deployed, never automated without your approval.

The pipeline: research (Exa + HuggingFace) → deliberate planning → write → review → log.
Every run is tracked in `agents/runs/runs.jsonl`. Check coverage with `uv run python generate.py status`.

```bash
# Setup
cd agents && uv sync && cp .env.example .env  # fill in your keys

# Generate a single page
uv run python generate.py generate \
  --topic diffusion-models \
  --track 02-generative-modeling \
  --page-type arc-entry \
  --depth-emphasis applied

# Arc-aware generation (tells agent where this page sits in the sequence)
uv run python generate.py generate \
  --topic score-matching \
  --track 02-generative-modeling \
  --page-type core-concept \
  --depth-emphasis theoretical \
  --arc generative-stack \
  --arc-position 4 \
  --prev-node ddpm \
  --next-node flow-matching

# Check coverage
uv run python generate.py status
```

---

## Source discipline

Every link in this wiki comes from one of:
- `arxiv.org` — papers
- `*.edu` — university course pages, lecture notes
- `huggingface.co` — model cards, datasets, spaces
- Official library documentation (PyTorch, JAX, etc.)

"In production" sections also allow official engineering blogs from top labs
(ai.meta.com/research, research.google, developer.nvidia.com/blog, etc.).

No Medium. No Towards Data Science. No personal blogs. No Substack. No Wikipedia.

---

## Quality signal

The reviewer agent scores every page 0.0–1.0 for: schema compliance, source policy,
prose quality (no nested lists, narrative flow), technical accuracy, and MVB executability.
Pages below 0.8 confidence are flagged for human review before writing to disk.

The GitHub star is the only engagement metric we collect.

---

See [PRINCIPLES.md](PRINCIPLES.md) for the operating philosophy behind this.
