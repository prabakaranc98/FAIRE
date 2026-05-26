---
skill: exa-search-deep
description: Exa AI search idioms FAIRE's research and supervisor agents must follow. type-of-query rules, find_similar usage, source-policy domains, query phrasing. Read by scratch/plan/research nodes and the supervisor's explorer phase.
applies_to: [scratch, plan, research, supervisor]
triggers: [all pages]
---

# Skill: Exa search idioms for FAIRE

This skill is the agents' durable knowledge of Exa AI — how to query it well, when to switch search types, and how source policy gets enforced at search time (not post-hoc). Source of truth: `agents/src/frontier_agents/tools.py`.

## The three search functions and when each fires

| Function | When | What it returns |
|---|---|---|
| `exa_search_papers(query, foundational=True)` | Looking for the seminal paper that introduced an idea | List of {url, title, text, highlights, domain, published_date, source_type="paper"} |
| `exa_search_sota(query)` | Looking for current SotA (auto-constrains to 2024+) | List of {url, title, highlights, domain, published_date, source_type="sota"} |
| `exa_search_production(query)` | Looking for production deployments (engineering blogs only) | List of {url, title, summary, domain, source_type="production"} |
| `exa_find_similar(seed_url)` | Once you have one good paper, widen the net | List of similar papers |

`research_node` in `nodes.py` fires all three in parallel via `ThreadPoolExecutor` (`RESEARCH_WORKERS` env var, default 7) and merges. Use the dedicated functions; do not call raw `Exa.search` from a node.

## `type` parameter — when to use which

- **`type="auto"`** (default for SotA, production): Exa picks neural vs keyword based on the query shape. Use when you don't know whether the model name is well-indexed or whether the query is conceptual.
- **`type="neural"`** (default for foundational papers): semantic search. Better when the query is "the paper that introduced/proved/showed [conceptual claim]" — Exa finds papers by what they *contributed*, not by what words they contain.
- **`type="keyword"`** (default for production blogs): exact string match. Better when the query is a named system or company (e.g., "Stability AI Movie Gen production deployment").

**Phrasing rules:**

| Search type | Good query | Bad query |
|---|---|---|
| Foundational (neural) | "A paper that introduced the noise-prediction objective for diffusion models" | "DDPM Ho 2020" |
| SotA (auto) | "Current state-of-the-art FID on ImageNet 256 class-conditional 2024" | "best diffusion model" |
| Production (keyword) | "Stable Diffusion 3.5 production deployment Stability AI" | "how diffusion models work in production" |

Neural queries: phrase as a *claim about contribution*, not as keywords.
Keyword queries: include exact names — model, benchmark, company, year.

## `category="research paper"` — when to set it

Always set `category="research paper"` for the foundational and SotA searches. It dramatically improves academic precision by filtering out blog posts, tutorials, and aggregator pages even when they happen to be on arxiv.

Don't set it for production blog searches — `include_domains` does the filtering there.

## `contents` parameter — highlights vs text

- **`text` with `max_characters=3000`**: full body excerpt. Use for foundational papers where the contribution is in the abstract + section 1.
- **`highlights={"num_sentences": 3, "highlights_per_url": 2}`**: 2-3 sentence excerpts from across the paper, picked by Exa for relevance. Use for SotA (you want the benchmark number, not the whole paper).
- **`summary={"query": "..."}`**: LLM-generated summary of the page, focused on a question. Use for production blogs ("what system did this company build, at what scale?").

`tools.py` already picks the right `contents` mode per function. Don't override unless you have a specific reason.

## `find_similar` — the cheap expansion trick

Once `exa_search_papers` returns a strong seed (e.g., DDPM paper for "diffusion"), `find_similar(seed_url, num_results=5)` returns related papers without you having to phrase another neural query. It's a different Exa endpoint and counts separately in rate limits.

When to use: after the primary search returns 1-3 good hits but you suspect the field is wider. Common in arc construction (the explorer playbook calls find_similar in step 1 of the survey phase).

## Source policy enforced at search time

The function-level `include_domains` parameter restricts results before they leave Exa's servers. This is more reliable than filtering after the fact (which can fail silently if Exa returns results in unexpected formats).

- `exa_search_papers` / `exa_search_sota`: `include_domains=["arxiv.org", ".edu"]`
- `exa_search_production`: `include_domains=APPROVED_ENGINEERING_BLOGS` (see `tools.py` for the full list — also documented in `critic-info-architecture.md`)
- The legacy `exa_search(section=...)` function combines based on `section="default"` vs `section="in_production"`.

If you need a domain not currently in either list, do not add it ad-hoc inside a node — edit `tools.py::APPROVED_RESEARCH_DOMAINS` or `APPROVED_ENGINEERING_BLOGS` and propagate to `critic-info-architecture.md` so the IA critic doesn't flag it later.

## `start_published_date` — required for SotA

`exa_search_sota` always sets `start_published_date="2024-01-01"`. Without it, an "SotA" query returns historical papers because Exa's relevance score doesn't account for recency.

For arc construction's explorer phase, also set `start_published_date` when surveying branching directions — you want the active research arms, not the historical state.

## Failure modes

- **Empty results**: `exa_search_*` returns `[]` rather than raising. Always check `len(results) == 0` before consuming. If empty, the query is probably too specific — drop a constraint and retry.
- **Network failure**: tools.py wraps these so the calling node sees an empty list, not an exception. The `_safe()` wrapper in `research_node` handles this.
- **Rate limit (429)**: not common but possible during parallel fanout. The research-workers pool defaults to 7 concurrent searches per page; lower if you hit limits.

## Query budget — don't be wasteful

Each Exa call costs API credits. The research node makes 5–8 calls per page (foundational × 2 + SotA × 2 + production × 1 + find_similar × 1). Don't call Exa from skills or critic prompts; route everything through `research_node` and its scratch_pad. The writer never re-searches.

## Related skills

- `source-policy.md` — what citations count as approved
- `critic-info-architecture.md` — enforces the same domain list on rendered pages
- `arc-exploration.md` — uses survey-then-find_similar to map arc branches
- `agent-runtime.md` — `RESEARCH_WORKERS` env var, etc.
