"""LangGraph node functions for the Frontier Wiki editorial agent system.

Graph flow:
  START → load_persona_node → read_stub_node → research_node
        → [mvb mode: mvb_recipe_node → merge_mvb_node → review_node]
        → [full mode: write_draft_node → review_node]
        → [approved: write_file_node → commit_node → END]
        → [rejected, retry < 2: revise_draft_node → review_node]
        → [rejected, retry == 2: flag_human_review_node → END]
"""

from __future__ import annotations

import os
from pathlib import Path

from .state import WikiPageState
from .tools import (
    exa_search,
    hf_search_datasets,
    hf_search_models,
    load_persona,
    read_stub,
    update_arc_json,
    write_file,
    git_commit,
)

# ---------------------------------------------------------------------------
# Model config
# ---------------------------------------------------------------------------

WRITER_MODEL = os.getenv("WRITER_MODEL", "claude-opus-4-7")
REVIEWER_MODEL = os.getenv("REVIEWER_MODEL", "claude-haiku-4-5-20251001")
MVB_MODEL = os.getenv("MVB_MODEL", "claude-opus-4-7")
DOCS_DIR = os.getenv("WIKI_DOCS_DIR", "../docs")
ARCS_JSON = os.getenv("WIKI_ARCS_JSON", "../docs/arcs.json")

_PERSONAS_DIR = Path(__file__).parent / "personas"


def _get_anthropic_client():
    import anthropic
    return anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def load_persona_node(state: WikiPageState) -> WikiPageState:
    """Load the per-track editor persona."""
    persona = load_persona(state["track"], str(_PERSONAS_DIR))
    output_path = (
        f"{DOCS_DIR}/curriculum/{state['track']}/{state['topic']}.md"
    )
    return {**state, "persona": persona, "output_path": output_path}


def read_stub_node(state: WikiPageState) -> WikiPageState:
    """Read existing page content if it exists."""
    stub = read_stub(state["output_path"])
    return {**state, "existing_stub": stub, "revision_count": 0}


def research_node(state: WikiPageState) -> WikiPageState:
    """Search Exa for authoritative sources on this topic.

    Searches:
    - arXiv papers (methodology + theory)
    - HuggingFace (models, datasets, spaces)
    - .edu pages (lecture notes, course material)
    - Engineering blogs (for "In production" section)
    """
    topic = state["topic"].replace("-", " ")
    track = state.get("persona", {}).get("domain", topic)

    queries = [
        f"{topic} arxiv paper deep learning",
        f"{topic} {track} survey tutorial",
        f"{topic} huggingface implementation",
    ]
    if state.get("persona", {}).get("search_seeds"):
        queries += state["persona"]["search_seeds"][:2]

    all_results: list[dict] = []
    for q in queries[:4]:
        try:
            results = exa_search(q, num_results=5)
            all_results.extend(results)
        except Exception:
            pass

    # Also search for "In production" sources
    production_results: list[dict] = []
    try:
        production_results = exa_search(
            f"{topic} production deployment scale engineering",
            num_results=4,
            section="in_production",
        )
    except Exception:
        pass

    # Dedup by URL
    seen = set()
    deduped = []
    for r in all_results + production_results:
        if r["url"] not in seen:
            seen.add(r["url"])
            deduped.append(r)

    return {**state, "research_results": deduped}


def write_draft_node(state: WikiPageState) -> WikiPageState:
    """Write a full wiki page draft using Claude Opus 4.7.

    The draft follows the canonical schema from SCHEMA.md:
    - For your reader type table (4 reader types)
    - What it is / Why it matters / Core concepts / Math foundations
    - Key algorithms / Essential reading / Seminal papers / Current SotA
    - What's happening now (Research · Engineering · Systems)
    - In production (top labs/companies deployments)
    - Minimum Valuable Build (if has_mvb: true)
    - Code & implementations / Connected topics / Further reading
    """
    client = _get_anthropic_client()

    schema_path = Path(__file__).parent.parent.parent / "SCHEMA.md"
    schema = schema_path.read_text(encoding="utf-8") if schema_path.exists() else ""

    persona = state.get("persona", {})
    research = state.get("research_results", [])
    existing = state.get("existing_stub", "")

    research_summary = "\n\n".join(
        f"URL: {r['url']}\nTitle: {r['title']}\n{r['text'][:800]}"
        for r in research[:12]
    )

    system_prompt = f"""You are the Frontier Wiki editorial agent — a knowledgeable expert in {persona.get('domain', 'AI/ML')}.

Your job is to write a single, self-contained wiki page that serves FOUR reader types simultaneously:
1. **Applied practitioner** (MS Data Science, industry): wants to build something; needs clear algorithms + MVB recipe
2. **Curious generalist**: wants intuition; needs clear "what is this" + "why does it matter"
3. **Math/theory student**: wants rigorous foundations; needs precise definitions + LaTeX equations
4. **Frontier researcher**: wants to know the current state + open problems; needs named papers + specific open questions

CRITICAL DESIGN PRINCIPLES:
- This wiki makes people "get shit done" — it nudges learners toward building, exploring, and pushing to frontiers
- Every page should feel like a knowledgeable mentor who has read everything and filtered to signal
- The page must be self-contained: opening it gives enough context without needing to follow links first
- "What's happening now" must name specific papers/systems/companies — never vague "recent work"
- "In production" shows REAL deployments at REAL scale — not toy examples
- The MVB must be concrete, runnable, and genuinely valuable (not just "verify the algorithm runs")

SCHEMA TO FOLLOW (exact section names required):
{schema}

SOURCE POLICY (strictly enforced):
- Default sections: arxiv.org, *.edu, huggingface.co, official library docs only
- "In production" section: official engineering blogs allowed (engineering.linkedin.com, ai.meta.com, developer.nvidia.com/blog, research.google, openai.com/research)
- NEVER cite: Medium, Towards Data Science, personal blogs, Substack, Wikipedia

Track persona: {persona.get('domain', '')} | Seminal authors: {', '.join(persona.get('seminal_authors', []))} | Key venues: {', '.join(persona.get('key_venues', []))}
"""

    user_prompt = f"""Write a complete, production-quality Frontier Wiki page for the topic: **{state['topic']}**
Track: {state['track']} | Depth: {state.get('depth', 'all')}

{"EXISTING CONTENT TO IMPROVE:\n" + existing if existing else "No existing content — write from scratch."}

RESEARCH SOURCES (use these for accurate citations — verify URLs before including):
{research_summary}

Generate the complete markdown page following the canonical schema exactly. Include all sections. For the MVB section, use real HuggingFace model/dataset links. For "In production", use real company deployments with real links. Make the math section have annotated LaTeX equations. Make every reader type feel served.
"""

    response = client.messages.create(
        model=WRITER_MODEL,
        max_tokens=8000,
        messages=[{"role": "user", "content": user_prompt}],
        system=system_prompt,
    )

    draft = response.content[0].text
    return {**state, "draft": draft}


def mvb_recipe_node(state: WikiPageState) -> WikiPageState:
    """On-demand: generate only the Minimum Valuable Build section.

    Searches HuggingFace for relevant models + datasets, then generates
    a concrete, runnable build recipe with real HF links.
    """
    client = _get_anthropic_client()
    topic = state["topic"].replace("-", " ")

    # Search HuggingFace for models + datasets
    hf_models = []
    hf_datasets = []
    try:
        hf_models = hf_search_models(topic, limit=5)
        hf_datasets = hf_search_datasets(topic, limit=5)
    except Exception:
        pass

    model_list = "\n".join(
        f"- {m['model_id']} ({m['downloads']:,} downloads) — {m['url']}"
        for m in hf_models[:5]
    )
    dataset_list = "\n".join(
        f"- {d['dataset_id']} ({d['downloads']:,} downloads) — {d['url']}"
        for d in hf_datasets[:5]
    )

    prompt = f"""Generate a Minimum Valuable Build (MVB) section for a Frontier Wiki page on: **{topic}**

Available HuggingFace models (pick the most suitable):
{model_list or "Search HuggingFace manually for relevant models"}

Available HuggingFace datasets (pick the most suitable):
{dataset_list or "Search HuggingFace manually for relevant datasets"}

The MVB must follow this exact format:

## Minimum Valuable Build

**What you're building:** [one sentence — specific, concrete project with a real use case]

**Why this is valuable:** [honest value — to the learner, to a user, to industry — not "understand the concept"]

**Stack:**
- **Model:** [HuggingFace model card link — real model ID]
- **Dataset:** [HuggingFace dataset link — real dataset ID]
- **Framework:** [PyTorch / JAX / Diffusers / Transformers / etc.]

**The recipe:**
1. [Step 1 — specific, one actionable sentence]
2. [Step 2]
3. [Step 3 — produces the valuable output]
4. [Step 4 — stretch/optional]

**Expected outcome:** [what you have at the end — something you can show, share, or build on]

**Stretch goals:**
- [How to push beyond minimum — something publishable, deployable, or shareable]
- [Alternative application of the same technique]

REQUIREMENTS:
- Use REAL HuggingFace model/dataset IDs — not placeholder names
- The recipe must be runnable today with public tools
- "Valuable" means genuinely useful — not just "verify the algorithm works"
- Include the specific import or first code step in the recipe
"""

    response = client.messages.create(
        model=MVB_MODEL,
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}],
    )

    mvb_section = response.content[0].text
    return {**state, "mvb_section": mvb_section, "hf_models": hf_models, "hf_datasets": hf_datasets}


def merge_mvb_node(state: WikiPageState) -> WikiPageState:
    """Merge a new MVB section into an existing page draft."""
    existing = state.get("existing_stub", "")
    mvb = state.get("mvb_section", "")

    if not existing:
        # No existing page — wrap MVB in a minimal stub
        topic_title = state["topic"].replace("-", " ").title()
        merged = f"""---
title: {topic_title}
track: {state['track']}
has_mvb: true
updated: 2026-05-25
---

# {topic_title}

> **TL;DR:** See the build recipe below; read the full page for theory.

{mvb}
"""
    else:
        # Replace or append the MVB section in existing content
        if "## Minimum Valuable Build" in existing:
            before = existing.split("## Minimum Valuable Build")[0]
            after_parts = existing.split("## Minimum Valuable Build")[1]
            # Find the next section after MVB
            next_section_idx = after_parts.find("\n## ", 1)
            after = after_parts[next_section_idx:] if next_section_idx != -1 else ""
            merged = before + mvb + "\n" + after
        else:
            # Append before "## Code & implementations" or at the end
            if "## Code & implementations" in existing:
                merged = existing.replace(
                    "## Code & implementations",
                    mvb + "\n\n## Code & implementations",
                )
            else:
                merged = existing + "\n\n" + mvb

    return {**state, "draft": merged}


def review_node(state: WikiPageState) -> WikiPageState:
    """Review agent (Claude Haiku) checks schema compliance and source policy.

    Assigns a confidence score 0.0-1.0. Returns structured feedback.
    Pages with confidence < 0.8 are flagged for human review.
    """
    client = _get_anthropic_client()

    draft = state.get("draft", "")
    schema_path = Path(__file__).parent.parent.parent / "SCHEMA.md"
    schema = schema_path.read_text(encoding="utf-8") if schema_path.exists() else ""

    prompt = f"""You are the Frontier Wiki reviewer agent. Check this wiki page for:

1. **Schema compliance** — does it have all required sections from the schema?
2. **Source policy** — are all URLs from approved domains (arxiv.org, *.edu, huggingface.co, official docs)?
3. **Quality** — does each reader type (applied, foundational, theoretical, frontier) get value?
4. **Accuracy** — are paper titles, authors, and claims plausible? Flag any suspicious hallucinations.
5. **MVB quality** — if present, is the build recipe specific, runnable, and genuinely valuable?

SCHEMA:
{schema[:3000]}

DRAFT TO REVIEW:
{draft[:6000]}

Respond in this EXACT format:
PASS/FAIL: [PASS or FAIL]
CONFIDENCE: [0.0-1.0]
ISSUES:
- [issue 1 — specific, actionable]
- [issue 2]
SUGGESTIONS:
- [suggestion 1]
"""

    response = client.messages.create(
        model=REVIEWER_MODEL,
        max_tokens=1500,
        messages=[{"role": "user", "content": prompt}],
    )

    feedback = response.content[0].text

    passed = "PASS/FAIL: PASS" in feedback
    confidence = 0.8
    try:
        for line in feedback.split("\n"):
            if "CONFIDENCE:" in line:
                confidence = float(line.split(":")[1].strip())
    except (ValueError, IndexError):
        pass

    return {
        **state,
        "review_feedback": feedback,
        "review_pass": passed,
        "review_confidence": confidence,
        "approved": passed and confidence >= 0.8,
    }


def revise_draft_node(state: WikiPageState) -> WikiPageState:
    """Writer agent revises the draft based on reviewer feedback."""
    client = _get_anthropic_client()

    prompt = f"""Revise this Frontier Wiki page draft based on the reviewer's feedback.

REVIEWER FEEDBACK:
{state.get('review_feedback', '')}

CURRENT DRAFT:
{state.get('draft', '')}

Fix all ISSUES listed. Implement SUGGESTIONS where reasonable. Keep all correct content unchanged.
Return the complete revised page.
"""

    response = client.messages.create(
        model=WRITER_MODEL,
        max_tokens=8000,
        messages=[{"role": "user", "content": prompt}],
    )

    revised = response.content[0].text
    revision_count = state.get("revision_count", 0) + 1
    return {**state, "draft": revised, "revision_count": revision_count}


def write_file_node(state: WikiPageState) -> WikiPageState:
    """Write the approved draft to the wiki docs directory."""
    final = state.get("draft", "")
    path = state["output_path"]
    write_file(path, final)
    return {**state, "final_content": final}


def commit_node(state: WikiPageState) -> WikiPageState:
    """Git commit the newly written wiki page."""
    path = state["output_path"]
    topic = state["topic"]
    committed = git_commit(
        path=path,
        message=f"wiki: add {topic} page (agent-generated)\n\nCo-authored-by: Frontier Wiki Agent",
        docs_dir=DOCS_DIR,
    )
    return {**state, "committed": committed}


def flag_human_review_node(state: WikiPageState) -> WikiPageState:
    """Mark the page for human review when agent approval fails after max retries."""
    from rich.console import Console
    console = Console()
    console.print(
        f"[yellow]⚠ Human review required:[/yellow] {state['output_path']}\n"
        f"Confidence: {state.get('review_confidence', 0):.2f}\n"
        f"Feedback:\n{state.get('review_feedback', '')}"
    )
    return {**state, "approved": False, "committed": False}


# ---------------------------------------------------------------------------
# Routing functions (used by LangGraph conditional edges)
# ---------------------------------------------------------------------------

def route_after_review(state: WikiPageState) -> str:
    """Decide what to do after reviewer runs."""
    if state.get("approved"):
        return "write_file"
    if state.get("revision_count", 0) >= 2:
        return "flag_human_review"
    return "revise_draft"


def route_after_load(state: WikiPageState) -> str:
    """Route to the right writing node based on mode."""
    mode = state.get("mode", "full")
    if mode == "mvb-only":
        return "mvb_recipe"
    return "research"
