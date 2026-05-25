# Frontier Wiki — Agent System

The editorial agent system that generates, reviews, and maintains all wiki topic pages. **This is the core engine of the wiki** — humans define structure and guardrails, agents generate content at scale.

## Architecture

```
agents/
├── generate.py                    ← CLI entry point
├── pyproject.toml                 ← uv project (Python 3.11+)
├── .env.example                   ← API key config
├── SCHEMA.md                      ← canonical page schema (agents must follow this)
└── src/frontier_agents/
    ├── state.py                   ← WikiPageState TypedDict
    ├── tools.py                   ← exa_search, hf_search, read_stub, write_file, git_commit
    ├── nodes.py                   ← research, write_draft, review, mvb_recipe, commit nodes
    ├── graph.py                   ← LangGraph StateGraph (two graphs: full + mvb-only)
    ├── cli.py                     ← Click CLI (generate command)
    └── personas/                  ← per-track editor configs (YAML)
        ├── 02-generative-modeling.yaml
        ├── 04-neural-networks-dl.yaml
        ├── 06-reinforcement-learning.yaml
        ├── 07-attention-memory-reasoning.yaml
        ├── 08-causal-statistical-inference.yaml
        ├── 09-algorithms-systems-ai.yaml
        ├── 11-robotics-embodied-ai.yaml
        ├── 12-physics-scientific-ai.yaml
        ├── 13-graph-relational-ai.yaml
        ├── 14-biology-life-sciences.yaml
        └── ...
```

## Agent pipeline

```
Full page (default):
  START → load_persona → read_stub → research (Exa) → write_draft (Opus 4.7)
        → review (Haiku 4.5) ──[approved]──→ write_file → git_commit → END
                              └─[rejected, retry<2]──→ revise_draft → review
                              └─[rejected, retry≥2]──→ flag_human_review → END

MVB-only (--mvb-only flag):
  START → load_persona → read_stub → mvb_recipe (HF search + Opus 4.7)
        → merge_mvb → review (Haiku 4.5) → write_file → git_commit → END
```

**Four agent types:**
- **Editorial agent** (`claude-opus-4-7`) — researches + writes full topic pages with all schema sections
- **Pedagogical agent** (`claude-opus-4-7`) — curates reading lists, arc sequences, SotA summaries
- **MVB recipe agent** (`claude-opus-4-7`, on-demand) — generates Minimum Valuable Build sections with real HuggingFace models/datasets
- **Reviewer agent** (`claude-haiku-4-5`) — schema compliance + source policy + confidence scoring

## Setup

```bash
cd agents

# Install uv (if needed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Configure API keys
cp .env.example .env
# Edit .env with your ANTHROPIC_API_KEY and EXA_API_KEY
```

## Usage

```bash
# Generate a full topic page
uv run python generate.py generate \
  --topic diffusion-models \
  --track 02-generative-modeling

# Generate with specific depth
uv run python generate.py generate \
  --topic transformer \
  --track 07-attention-memory-reasoning \
  --depth all

# Generate only the MVB section for an existing page
uv run python generate.py generate \
  --topic rlhf \
  --track 06-reinforcement-learning \
  --mvb-only

# Generate all stubs in a track (batch mode)
uv run python generate.py generate \
  --track 02-generative-modeling \
  --all-stubs

# Dry run (preview without writing to disk)
uv run python generate.py generate \
  --topic flow-matching \
  --track 02-generative-modeling \
  --dry-run
```

## What agents generate

Every topic page follows the canonical schema in `SCHEMA.md`:

1. **For your reader type** — table routing 4 reader types (applied, foundational, theoretical, frontier)
2. **What it is** — 2-3 paragraph clear explanation serving the curious generalist
3. **Why it matters at the frontier** — connects to open problems and frontier labs
4. **Core concepts** — 5-8 key ideas defined precisely in 1 sentence each
5. **Mathematical foundations** — annotated LaTeX equations for theory readers
6. **Key algorithms / techniques** — named methods for applied practitioners
7. **Essential reading** — 2-4 papers that are the minimum to understand this topic
8. **Seminal papers & test-of-time** — signal filtered from noise
9. **Current SotA** — named papers/systems, not vague "recent work"
10. **What's happening now** — Research · Engineering · Systems (specific, named)
11. **In production** — real deployments at real scale with real company links
12. **Minimum Valuable Build** — concrete, runnable recipe with HuggingFace models/datasets
13. **Code & implementations** — curated links to official repos
14. **Connected topics** — wikilinks for graph navigation
15. **Further reading** — additional resources (arXiv/edu only)

## Source policy

Enforced by the reviewer agent at commit time:

| Section | Allowed domains |
|---|---|
| All sections (default) | arxiv.org, *.edu, huggingface.co, official library docs |
| "In production" section | + official engineering blogs (engineering.linkedin.com, ai.meta.com, developer.nvidia.com/blog, research.google, openai.com/research) |
| Rejected everywhere | Medium, Towards Data Science, personal blogs, Substack, YouTube, Wikipedia |

## Adding a new track

1. Create the track directory: `docs/curriculum/NN-track-name/`
2. Add a persona YAML: `agents/src/frontier_agents/personas/NN-track-name.yaml`
3. Create topic stubs with the stub format from `SCHEMA.md`
4. Run: `uv run python generate.py generate --track NN-track-name --all-stubs`

## Reviewer confidence scores

| Score | Action |
|---|---|
| ≥ 0.9 | Auto-approved, committed |
| 0.8–0.9 | Approved, committed |
| 0.6–0.8 | Revision loop (max 2 rounds) |
| < 0.6 | Flagged for human review |
