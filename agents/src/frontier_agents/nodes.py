"""LangGraph node functions — Frontier Wiki editorial agent system.

All LLM calls go through LangChain (ChatOpenAI via OpenRouter or ChatAnthropic direct).
The reviewer uses structured output (Pydantic) for reliable PASS/FAIL + confidence.

Graph flow:
  START → load_persona → read_stub → research → plan → scratch
        → [full] write_draft → review
        → [mvb-only] mvb_recipe → merge_mvb → review
        → [approved] write_file → commit → log_run → END
        → [rejected, count<2] revise_draft → review
        → [rejected, count≥2] flag_human_review → log_run → END
"""

from __future__ import annotations

import json as _json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from .llm import get_llm


def _coerce_text(content) -> str:
    """Robustly turn an LLM response.content into a string.

    Some OpenRouter models (notably the openai/gpt-5.x reasoning family and
    google/gemini-3.x via langchain_openai) return `content` as a list of
    content blocks instead of a flat string. Each block is either a dict
    with a `text` field (or `reasoning`/`thinking`), or a plain string.

    Returns the concatenated text, stripped of leading/trailing whitespace.
    Defensive: always returns a str, never raises.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                # Common keys across providers; skip reasoning/thinking blocks
                # since they're internal scratch the model used to plan
                if block.get("type") in ("thinking", "reasoning"):
                    continue
                parts.append(
                    block.get("text") or block.get("content") or ""
                )
        return "".join(parts)
    return str(content)
from .prompts import (
    MVB_SYSTEM,
    PLAN_SYSTEM,
    REVIEWER_SYSTEM,
    SCRATCH_SYSTEM,
    WRITE_INSTRUCTIONS,
    WRITE_INSTRUCTIONS_ARC_INDEX,
    WRITE_INSTRUCTIONS_ARC_STEP,
    WRITER_SYSTEM,
)
from .skills import context_tokens_from_state, format_skills_block, load_skills, select_skills
from .state import WikiPageState
from .tools import (
    exa_find_similar,
    exa_search_papers,
    exa_search_production,
    exa_search_sota,
    git_commit,
    hf_search_datasets,
    hf_search_models,
    load_persona,
    log_run,
    read_stub,
    write_file,
)

DOCS_DIR = os.getenv("WIKI_DOCS_DIR", "../docs")
_PERSONAS_DIR = Path(__file__).parent / "personas"
_SCHEMA_PATH = Path(__file__).parent.parent.parent / "SCHEMA.md"

# Loaded once at import time; safe because skills files are read-only at runtime.
_SKILLS = load_skills()


def _load_schema() -> str:
    return _SCHEMA_PATH.read_text(encoding="utf-8") if _SCHEMA_PATH.exists() else ""


# ---------------------------------------------------------------------------
# Pydantic model for structured reviewer output
# ---------------------------------------------------------------------------

class ReviewResult(BaseModel):
    """Structured review output with rubric-based dimensions.

    Approval uses a rubric: each dimension is scored independently.
    A page approves when ALL blocker dimensions pass and prose ≥ 0.6.
    This prevents a single minor issue from tanking an otherwise excellent page.
    """
    passed: bool = Field(description="True if the page passes schema + source policy checks")
    confidence: float = Field(ge=0.0, le=1.0, description="Overall quality confidence score")
    issues: list[str] = Field(description="Specific actionable issues to fix")
    suggestions: list[str] = Field(description="Optional improvement suggestions")
    # Rubric dimensions (each 0.0–1.0)
    schema_score: float = Field(
        default=1.0, ge=0.0, le=1.0,
        description="Schema compliance: all required sections present and named correctly (0=major missing, 1=complete)"
    )
    source_score: float = Field(
        default=1.0, ge=0.0, le=1.0,
        description="Source policy: only approved domains, no hallucinated citations (0=violations, 1=clean)"
    )
    prose_score: float = Field(
        default=0.7, ge=0.0, le=1.0,
        description="Prose quality: no nested lists, no course language, opens with scenario not definition (0=bad, 1=excellent)"
    )
    mvb_score: float = Field(
        default=0.8, ge=0.0, le=1.0,
        description="MVB quality: specific model IDs, realistic compute, actionable recipe, CTA present (0=missing/vague, 1=complete)"
    )
    frontier_citation_score: float = Field(
        default=0.8, ge=0.0, le=1.0,
        description="Frontier sections cite papers with author/year/URL inline; vague claims like 'recent work suggests' score 0 (0=uncited vague claims, 1=every frontier claim has author-year-URL)"
    )
    open_questions_score: float = Field(
        default=0.7, ge=0.0, le=1.0,
        description="Open questions section quality: three admonition blocks (researcher/engineer/open), each with a specific publishable-level question (0=absent/vague, 1=three specific questions present)"
    )
    backlink_score: float = Field(
        default=0.7, ge=0.0, le=1.0,
        description="Arc backlinks: curriculum pages link to arc steps (This concept appears in), arc steps link to curriculum (Go deeper) with context sentences (0=absent, 1=present with context)"
    )


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def load_persona_node(state: WikiPageState) -> WikiPageState:
    persona = load_persona(state["topic"], str(_PERSONAS_DIR))
    if persona.get("track") != state["track"]:
        persona = load_persona(state["track"], str(_PERSONAS_DIR))
    output_path = _resolve_output_path(state)
    run_id = str(uuid.uuid4())
    started_at = datetime.now(timezone.utc).isoformat()
    return {**state, "persona": persona, "output_path": output_path,
            "run_id": run_id, "started_at": started_at}


def _resolve_output_path(state: WikiPageState) -> str:
    """Route output to the v2 curriculum tree based on mode/page_type.

    v2 layout (see docs/system/structure-v2.md):
    - mode == "arc-step"   → docs/curriculum/core/{track}/arcs/{arc_id}/step-{pos:02d}-{slug}.md
    - mode == "arc-index"  → docs/curriculum/core/{track}/arcs/{arc_id}/index.md
    - page_type == "author" → docs/curriculum/core/{track}/authors/{slug}.md
    - page_type == "build"  → docs/curriculum/core/{track}/builds/{slug}.md
    - everything else (concept) → docs/curriculum/core/{track}/concepts/{slug}.md
    """
    mode = state.get("mode", "full")
    arc_ctx = state.get("arc_context") or {}
    arc_id = arc_ctx.get("arc_id", "")
    page_type = state.get("page_type", "concept")
    topic = state["topic"]
    track = state["track"]

    if mode == "arc-step" and arc_id:
        pos = int(arc_ctx.get("position") or 0)
        return f"{DOCS_DIR}/curriculum/core/{track}/arcs/{arc_id}/step-{pos:02d}-{topic}.md"
    if mode == "arc-index" and arc_id:
        return f"{DOCS_DIR}/curriculum/core/{track}/arcs/{arc_id}/index.md"
    if page_type == "author":
        return f"{DOCS_DIR}/curriculum/core/{track}/authors/{topic}.md"
    if page_type == "build":
        return f"{DOCS_DIR}/curriculum/core/{track}/builds/{topic}.md"
    return f"{DOCS_DIR}/curriculum/core/{track}/concepts/{topic}.md"


def read_stub_node(state: WikiPageState) -> WikiPageState:
    stub = read_stub(state["output_path"])
    return {**state, "existing_stub": stub, "revision_count": 0}


def research_node(state: WikiPageState) -> WikiPageState:
    """Three-phase Exa search: foundational papers, SotA (2024+), production deployments."""
    from concurrent.futures import ThreadPoolExecutor as _TPE, as_completed as _asc
    import os as _os

    topic = state["topic"].replace("-", " ")
    persona = state.get("persona", {})
    domain = persona.get("domain", "")
    seeds = (persona.get("search_seeds") or [])[:2]

    def _safe(fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception:
            return []

    # Build parallel search tasks using Exa neural query best practices.
    # Neural queries: phrase as "A paper that introduced/proved/showed [X]"
    # Keyword queries: use for specific named systems, benchmarks, companies
    search_tasks = [
        # Foundational papers: neural query describing the contribution
        ("paper", exa_search_papers,
         (f"A paper that introduced or defined {topic} as a method or concept",),
         {"foundational": True}),
        ("paper", exa_search_papers,
         (f"The original {topic} paper {domain} seminal contribution arxiv",),
         {"foundational": True}),
        # SotA: neural for contribution description, keyword for benchmark names
        ("sota",  exa_search_sota,
         (f"A paper showing state-of-the-art results using {topic} on benchmark 2024 2025",), {}),
        ("sota",  exa_search_sota,
         (f"{topic} best result {domain} benchmark metric performance 2025 arxiv",), {}),
        # Production: keyword search for company + system names
        ("prod",  exa_search_production,
         (f"{topic} production deployment {domain} engineering at scale real system",), {}),
        # HuggingFace
        ("hf_m",  hf_search_models,  (topic,), {}),
        ("hf_d",  hf_search_datasets,(topic,), {}),
    ]
    for seed in seeds:
        search_tasks.append(("paper", exa_search_papers,
                             (f"A paper about {seed}",), {"foundational": True}))

    paper_results: list[dict] = []
    sota_results:  list[dict] = []
    production_results: list[dict] = []
    hf_models: list[dict] = []
    hf_datasets: list[dict] = []

    max_workers = int(_os.getenv("RESEARCH_WORKERS", "7"))
    with _TPE(max_workers=max_workers, thread_name_prefix="research") as pool:
        futs = {
            pool.submit(_safe, fn, *args, **kwargs): bucket
            for bucket, fn, args, kwargs in search_tasks
        }
        for fut in _asc(futs):
            bucket = futs[fut]
            val = fut.result()
            if   bucket == "paper": paper_results.extend(val)
            elif bucket == "sota":  sota_results.extend(val)
            elif bucket == "prod":  production_results = val or production_results
            elif bucket == "hf_m":  hf_models = val or hf_models
            elif bucket == "hf_d":  hf_datasets = val or hf_datasets

    # Dedup paper results by URL
    seen: set[str] = set()
    deduped_papers = []
    for r in paper_results:
        if r["url"] not in seen:
            seen.add(r["url"])
            deduped_papers.append(r)

    # Expand with find_similar on the top 2 foundational papers (Exa best practice:
    # find_similar surfaces related work that keyword/neural queries miss).
    # Run sequentially — pool is closed above; only 2 calls so threading is not worth it.
    arxiv_seeds = [r["url"] for r in deduped_papers if "arxiv.org" in r.get("url", "")][:2]
    if arxiv_seeds and int(_os.getenv("EXA_USE_FIND_SIMILAR", "1")):
        for url in arxiv_seeds:
            for sim in (_safe(exa_find_similar, url, 3) or []):
                if sim["url"] not in seen:
                    seen.add(sim["url"])
                    deduped_papers.append(sim)

    return {
        **state,
        "research_results": deduped_papers,
        "sota_results": sota_results,
        "production_results": production_results,
        "hf_models": hf_models,
        "hf_datasets": hf_datasets,
    }


def _do_plan(state: WikiPageState) -> str:
    """Return the writing plan string (extracted from plan_node for parallel use)."""
    planner = get_llm("research", temperature=0.2)
    topic = state["topic"].replace("-", " ")
    page_type = state.get("page_type", "core-concept")
    depth_emphasis = state.get("depth_emphasis", ["applied"])
    arc_context = state.get("arc_context", {})

    papers_summary = "\n".join(
        f"- {r.get('title', 'Untitled')} ({r.get('url', '')})"
        for r in (state.get("research_results", []))[:8]
    )
    sota_summary = "\n".join(
        f"- {r.get('title', 'Untitled')}: {'; '.join((r.get('highlights') or [])[:1])}"
        for r in (state.get("sota_results", []))[:4]
    )
    prod_summary = "\n".join(
        f"- {r.get('title', 'Untitled')}: {str(r.get('summary', ''))[:200]}"
        for r in (state.get("production_results", []))[:3]
    )

    arc_info = ""
    if arc_context:
        arc_info = (
            f"\nArc: {arc_context.get('arc_id', 'unknown')} "
            f"(position {arc_context.get('position', '?')}), "
            f"prev: {arc_context.get('prev', 'none')}, "
            f"next: {arc_context.get('next', 'none')}"
        )

    user = f"""Produce a writing plan for this Frontier Wiki page.

Topic: {topic}
Page type: {page_type}
Depth emphasis: {', '.join(depth_emphasis)}{arc_info}

Foundational papers found:
{papers_summary or 'No foundational papers found in search.'}

SotA (2024+) found:
{sota_summary or 'No recent SotA results found.'}

Production deployments found:
{prod_summary or 'No production examples found.'}

{PLAN_SYSTEM}
"""
    return _coerce_text(planner.invoke([HumanMessage(content=user)]).content)


def _do_scratch(state: WikiPageState) -> str:
    """Return the scratch pad string (extracted from scratch_node for parallel use)."""
    compiler = get_llm("research", temperature=0.1)
    topic = state["topic"].replace("-", " ")
    writing_plan = state.get("writing_plan", "")

    def _fmt_papers(results: list[dict], max_items: int = 8) -> str:
        return "\n".join(
            f"- [{r.get('title', 'Untitled')}]({r.get('url', '')}) — "
            f"{r.get('text', '')[:300].strip()}"
            for r in results[:max_items]
        )

    def _fmt_sota(results: list[dict], max_items: int = 5) -> str:
        return "\n".join(
            f"- [{r.get('title', 'Untitled')}]({r.get('url', '')}) — "
            f"{'; '.join((r.get('highlights') or [])[:2])}"
            for r in results[:max_items]
        )

    def _fmt_prod(results: list[dict], max_items: int = 4) -> str:
        return "\n".join(
            f"- [{r.get('title', 'Untitled')}]({r.get('url', '')}) — "
            f"{str(r.get('summary', ''))[:200]}"
            for r in results[:max_items]
        )

    hf_models = state.get("hf_models", [])
    hf_datasets = state.get("hf_datasets", [])
    hf_block = (
        "Models: " + " | ".join(
            f"{m['model_id']} ({m['downloads']:,} dl)" for m in hf_models[:5]
        ) + "\nDatasets: " + " | ".join(
            f"{d['dataset_id']} ({d['downloads']:,} dl)" for d in hf_datasets[:5]
        )
    ) if hf_models or hf_datasets else "No HuggingFace results found."

    ctx_tokens = context_tokens_from_state(state)
    skills_block = format_skills_block(select_skills(_SKILLS, "scratch", ctx_tokens))

    user = f"""Compile a working-memory fact sheet for the Frontier Wiki page on: **{topic}**

Writing plan (use this to guide what's essential):
{writing_plan}

Raw research results — extract, verify, organise:

FOUNDATIONAL PAPERS:
{_fmt_papers(state.get("research_results", [])) or "None found."}

SOTA (2024+):
{_fmt_sota(state.get("sota_results", [])) or "None found."}

PRODUCTION DEPLOYMENTS:
{_fmt_prod(state.get("production_results", [])) or "None found."}

HUGGINGFACE (for MVB):
{hf_block}

{SCRATCH_SYSTEM}{skills_block}
"""
    return _coerce_text(compiler.invoke([HumanMessage(content=user)]).content)


def plan_and_scratch_node(state: WikiPageState) -> WikiPageState:
    """Run plan first, then pass it to scratch so the fact-sheet is plan-guided."""
    writing_plan = _do_plan(state)
    state_with_plan = {**state, "writing_plan": writing_plan}
    scratch_pad = _do_scratch(state_with_plan)
    return {**state, "writing_plan": writing_plan, "scratch_pad": scratch_pad}


def _format_checklist_block(checklist: dict | None) -> str:
    """Render the writing_checklist as a block the writer reads.

    The reviewer enforces this list deterministically (regex), so we make the
    requirements explicit in the prompt. Returns empty string if no checklist.
    """
    if not checklist:
        return ""
    parts = []
    papers = checklist.get("must_cite_papers") or []
    hf = checklist.get("must_use_hf_models") or []
    concepts = checklist.get("must_link_concepts") or []
    if papers:
        parts.append("MUST cite these papers inline (Author et al. YEAR + arxiv URL):")
        parts.extend(f"  • {p}" for p in papers)
    if hf:
        parts.append("\nMUST use these HuggingFace IDs in Build it (do NOT invent others):")
        parts.extend(f"  • {m}" for m in hf)
    if concepts:
        parts.append("\nMUST link these related concepts inline in Where to read next:")
        parts.extend(f"  • [[{c}]]" for c in concepts)
    if not parts:
        return ""
    return (
        "\n\n════════════════════════════════════════\n"
        "MANDATORY CHECKLIST — the reviewer rejects any page that misses items\n"
        "════════════════════════════════════════\n"
        + "\n".join(parts)
        + "\n"
    )


def build_writing_checklist_node(state: WikiPageState) -> WikiPageState:
    """Promote grounded facts from research into a mandatory writer checklist.

    Pure-Python derivation (no LLM call): pull the top-N citations from
    research_results/sota_results, the top HF model IDs from hf_models, and
    the prereq slugs from the persona. The writer treats these as required;
    the reviewer downgrades any page that omits items.

    This directly attacks two failure modes seen in production:
      - HF model ID hallucination (must_use_hf_models is pre-verified)
      - Citation vagueness (must_cite_papers is named author+url)
    """
    research = state.get("research_results", []) or []
    sota = state.get("sota_results", []) or []
    hf_models = state.get("hf_models", []) or []
    prereqs = (state.get("persona", {}) or {}).get("prereqs", []) or []

    def _cite_str(r: dict) -> str:
        url = r.get("url", "")
        title = (r.get("title", "Untitled") or "Untitled").strip()
        return f"{title[:80]} ({url})"

    # Citations: take arxiv/edu URLs from research + sota (top 4 by source quality)
    must_cite = []
    seen_urls = set()
    for r in research + sota:
        url = r.get("url", "")
        if not url or url in seen_urls:
            continue
        if "arxiv.org" in url.lower() or ".edu" in url.lower():
            must_cite.append(_cite_str(r))
            seen_urls.add(url)
            if len(must_cite) >= 4:
                break

    # HF models: take top 2 by downloads (only ones that actually exist)
    must_use_hf = []
    for m in sorted(hf_models, key=lambda x: x.get("downloads", 0), reverse=True):
        mid = m.get("model_id", "")
        if mid and m.get("downloads", 0) > 50:
            must_use_hf.append(mid)
            if len(must_use_hf) >= 2:
                break

    # Related concepts: prereqs + top 2 topic_slugs from research (if any)
    must_link = list(prereqs)[:3]

    checklist = {
        "must_cite_papers": must_cite,
        "must_use_hf_models": must_use_hf,
        "must_link_concepts": must_link,
    }
    return {**state, "writing_checklist": checklist}


def keep_best_draft_node(state: WikiPageState) -> WikiPageState:
    """Knockout selection — if the revised draft scored worse than the prior,
    restore the prior draft and its review state.

    Pattern from PerFine (arxiv 2510.24469, 2025). Prevents the well-known
    'revision makes it worse' regression where the writer's second pass
    introduces new problems while fixing the flagged ones.

    Tolerance: revised must be at least 0.02 confidence below prev to trigger
    a restore (small wiggle room for review noise).
    """
    current_conf = state.get("review_confidence", 0.0) or 0.0
    prev_conf = state.get("prev_review_confidence", 0.0) or 0.0
    prev_draft = state.get("prev_draft", "") or ""

    # First pass (no prev to compare) — just stash current as prev for next round
    if not prev_draft:
        return {
            **state,
            "prev_draft": state.get("draft", ""),
            "prev_review_confidence": current_conf,
            "prev_review_issues": list(state.get("review_issues", []) or []),
        }

    # Revised regressed vs prior — restore prior, log a note
    if current_conf + 0.02 < prev_conf:
        try:
            from rich.console import Console
            Console().print(
                f"[yellow]↩ Knockout: revised draft {current_conf:.2f} < prev "
                f"{prev_conf:.2f} — restoring prior draft.[/yellow]"
            )
        except ImportError:
            pass
        return {
            **state,
            "draft": prev_draft,
            "review_confidence": prev_conf,
            "review_issues": list(state.get("prev_review_issues", []) or []),
            # 'approved' stays whatever the prev pass yielded; route_after_review
            # reads review_confidence so the kept draft drives downstream routing
        }

    # Revised is as-good-or-better — stash it as the new prev for the next round
    return {
        **state,
        "prev_draft": state.get("draft", ""),
        "prev_review_confidence": current_conf,
        "prev_review_issues": list(state.get("review_issues", []) or []),
    }


def write_draft_node(state: WikiPageState) -> WikiPageState:
    """Write the complete wiki page in a single LLM call.

    The model has 200K context — a full page (~7500 tokens) plus research context
    fits comfortably in one call. Single-pass writing produces coherent output:
    the model maintains the full narrative arc, avoids repetition, and can cross-
    reference earlier sections naturally.
    """
    writer = get_llm("writer", temperature=0.3)
    schema = _load_schema()
    persona = state.get("persona", {})
    writing_plan = state.get("writing_plan", "")
    scratch_pad = state.get("scratch_pad", "")
    page_type = state.get("page_type", "core-concept")
    depth_emphasis = state.get("depth_emphasis", ["applied"])
    topic = state["topic"]
    track = state["track"]

    ctx_tokens = context_tokens_from_state(state)
    skills_block = format_skills_block(select_skills(_SKILLS, "write_draft", ctx_tokens))
    system = (
        WRITER_SYSTEM
        .replace("{domain}", persona.get("domain", "AI/ML"))
        .replace("{schema}", schema[:12000])
    ) + skills_block

    depth_note = (
        "Depth emphasis: " + ", ".join(depth_emphasis) + ". "
        + ("Lean applied: longer MVB, code patterns, production framing. "
           if "applied" in depth_emphasis else "")
        + ("Lean theoretical: derivation steps, formal definitions, proof intuitions. "
           if "theoretical" in depth_emphasis else "")
        + ("Lean frontier: benchmark numbers, named 2024–2025 papers, open problems as questions. "
           if "frontier" in depth_emphasis else "")
    )

    existing = state.get("existing_stub", "")
    improve_note = (
        "\n⚠ IMPROVE MODE — existing content below. Keep what's good, rewrite what's weak:\n"
        + existing[:3000]
        if existing.strip() and "🚧" not in existing else ""
    )

    checklist_block = _format_checklist_block(state.get("writing_checklist"))

    user = f"""Topic: **{topic}** | Track: {track} | Page type: {page_type}
{depth_note}
{checklist_block}
════════════════════════════════════
WRITING PLAN
════════════════════════════════════
{writing_plan}

════════════════════════════════════
WORKING MEMORY (verified facts — use ONLY these citations, equations, examples)
════════════════════════════════════
{scratch_pad}
{improve_note}

{WRITE_INSTRUCTIONS}
"""

    response = writer.invoke([
        SystemMessage(content=system),
        HumanMessage(content=user),
    ])
    draft = _coerce_text(response.content).strip()
    return {**state, "draft": draft}


def write_arc_step_node(state: WikiPageState) -> WikiPageState:
    """Write an arc step build page using the arc-step schema and instructions.

    Uses the same WRITER_SYSTEM persona + scratch_pad as write_draft_node, but
    applies WRITE_INSTRUCTIONS_ARC_STEP instead of the curriculum WRITE_INSTRUCTIONS.
    Arc step pages always have has_mvb: true and use the arc_context for breadcrumbs.
    """
    writer = get_llm("writer", temperature=0.3)
    persona = state.get("persona", {})
    writing_plan = state.get("writing_plan", "")
    scratch_pad = state.get("scratch_pad", "")
    page_type = state.get("page_type", "arc-step")
    depth_emphasis = state.get("depth_emphasis", ["applied"])
    topic = state["topic"]
    track = state["track"]
    arc_context = state.get("arc_context", {})

    ctx_tokens = context_tokens_from_state(state)
    skills_block = format_skills_block(select_skills(_SKILLS, "write_draft", ctx_tokens))
    system = (
        WRITER_SYSTEM
        .replace("{domain}", persona.get("domain", "AI/ML"))
        .replace("{schema}", "")  # arc step uses WRITE_INSTRUCTIONS_ARC_STEP, not SCHEMA.md
    ) + skills_block

    arc_info = ""
    if arc_context:
        arc_info = (
            f"\nArc: {arc_context.get('arc_id', 'unknown')} "
            f"— Step {arc_context.get('position', '?')} of {arc_context.get('total', '?')}\n"
            f"Arc title: {arc_context.get('arc_title', '')}\n"
            f"Previous step: {arc_context.get('prev', 'none')}\n"
            f"Next step: {arc_context.get('next', 'none')}\n"
            f"Output path: {state.get('output_path', '')}"
        )

    depth_note = (
        "Depth emphasis: " + ", ".join(depth_emphasis) + ". "
        + ("Lean applied: more specific recipe steps, exact hyperparameters. "
           if "applied" in depth_emphasis else "")
        + ("Lean theoretical: fuller theory section, equation derivation steps. "
           if "theoretical" in depth_emphasis else "")
    )

    existing = state.get("existing_stub", "")
    improve_note = (
        "\n⚠ IMPROVE MODE — existing content below. Keep what's good, rewrite what's weak:\n"
        + existing[:3000]
        if existing.strip() and "🚧" not in existing else ""
    )

    user = f"""Topic: **{topic}** | Track: {track} | Page type: {page_type}
{depth_note}
{arc_info}

════════════════════════════════════
WRITING PLAN
════════════════════════════════════
{writing_plan}

════════════════════════════════════
WORKING MEMORY (verified facts — use ONLY these citations, equations, examples)
════════════════════════════════════
{scratch_pad}
{improve_note}

{WRITE_INSTRUCTIONS_ARC_STEP}
"""

    response = writer.invoke([
        SystemMessage(content=system),
        HumanMessage(content=user),
    ])
    draft = _coerce_text(response.content).strip()
    return {**state, "draft": draft}


def write_arc_index_node(state: WikiPageState) -> WikiPageState:
    """Write an arc index page: destination, chapters, curated readings, compounding trajectory.

    Uses WRITE_INSTRUCTIONS_ARC_INDEX. Arc index pages have has_mvb: false — builds live
    in the step pages. The arc_context.chapters list drives the chapter structure.
    """
    writer = get_llm("writer", temperature=0.3)
    persona = state.get("persona", {})
    writing_plan = state.get("writing_plan", "")
    scratch_pad = state.get("scratch_pad", "")
    topic = state["topic"]
    track = state["track"]
    arc_context = state.get("arc_context", {})

    ctx_tokens = context_tokens_from_state(state)
    skills_block = format_skills_block(select_skills(_SKILLS, "write_draft", ctx_tokens))
    system = (
        WRITER_SYSTEM
        .replace("{domain}", persona.get("domain", "AI/ML"))
        .replace("{schema}", "")
    ) + skills_block

    chapters_info = ""
    if arc_context.get("chapters"):
        chapters_info = "\nArc chapters:\n" + "\n".join(
            f"  Chapter {c.get('number', i+1)}: {c.get('title', '')} "
            f"— steps {c.get('steps', [])}"
            for i, c in enumerate(arc_context["chapters"])
        )

    arc_info = (
        f"\nArc: {arc_context.get('arc_id', topic)}\n"
        f"Arc title: {arc_context.get('arc_title', topic.replace('-', ' ').title())}\n"
        f"Destination: {arc_context.get('destination', '')}\n"
        f"Total steps: {arc_context.get('total', '?')}\n"
        f"Tracks: {arc_context.get('tracks', [])}\n"
        f"Prerequisites: {arc_context.get('prereqs', [])}"
        + chapters_info
    ) if arc_context else ""

    existing = state.get("existing_stub", "")
    improve_note = (
        "\n⚠ IMPROVE MODE — existing arc index below. Keep structure, rewrite for new schema:\n"
        + existing[:3000]
        if existing.strip() and "🚧" not in existing else ""
    )

    user = f"""Topic: **{topic}** | Track: {track} | Page type: arc-index
{arc_info}

════════════════════════════════════
WRITING PLAN
════════════════════════════════════
{writing_plan}

════════════════════════════════════
WORKING MEMORY (verified citations and readings — use ONLY these)
════════════════════════════════════
{scratch_pad}
{improve_note}

{WRITE_INSTRUCTIONS_ARC_INDEX}
"""

    response = writer.invoke([
        SystemMessage(content=system),
        HumanMessage(content=user),
    ])
    draft = _coerce_text(response.content).strip()
    return {**state, "draft": draft}


def mvb_recipe_node(state: WikiPageState) -> WikiPageState:
    """On-demand: generate only the MVB section using HuggingFace search + LLM."""
    writer = get_llm("mvb", temperature=0.2)
    topic = state["topic"].replace("-", " ")

    hf_models = state.get("hf_models") or []
    hf_datasets = state.get("hf_datasets") or []

    if not hf_models:
        try:
            hf_models = hf_search_models(topic, limit=5)
            hf_datasets = hf_search_datasets(topic, limit=5)
        except Exception:
            pass

    model_list = "\n".join(
        f"- {m['model_id']} ({m['downloads']:,} downloads): {m['url']}" for m in hf_models[:5]
    ) or "Search huggingface.co/models manually"

    dataset_list = "\n".join(
        f"- {d['dataset_id']} ({d['downloads']:,} downloads): {d['url']}" for d in hf_datasets[:5]
    ) or "Search huggingface.co/datasets manually"

    prompt = f"""{MVB_SYSTEM}

---
Generate a Minimum Valuable Build (MVB) section for the Frontier Wiki topic: **{topic}**

Available HuggingFace models (sorted by downloads — prefer these):
{model_list}

Available HuggingFace datasets (sorted by downloads — prefer these):
{dataset_list}

Output ONLY the MVB section markdown (starting with ## Minimum Valuable Build).
"""

    response = writer.invoke([HumanMessage(content=prompt)])
    return {**state, "mvb_section": _coerce_text(response.content), "hf_models": hf_models, "hf_datasets": hf_datasets}


def merge_mvb_node(state: WikiPageState) -> WikiPageState:
    """Inject MVB section into existing page or create a minimal wrapper."""
    existing = state.get("existing_stub", "")
    mvb = state.get("mvb_section", "")

    if not existing or "🚧" in existing:
        title = state["topic"].replace("-", " ").title()
        draft = f"""---
title: {title}
track: {state['track']}
has_mvb: true
updated: 2026-05-25
---

# {title}
> **TL;DR:** See the Minimum Valuable Build below. Full page content pending agent generation.

{mvb}
"""
    elif "## Minimum Valuable Build" in existing:
        before, rest = existing.split("## Minimum Valuable Build", 1)
        next_section = rest.find("\n## ", 1)
        after = rest[next_section:] if next_section != -1 else ""
        draft = before + mvb + "\n" + after
    else:
        marker = "## Code & implementations"
        if marker in existing:
            draft = existing.replace(marker, mvb + "\n\n" + marker, 1)
        else:
            draft = existing + "\n\n" + mvb

    return {**state, "draft": draft}


def _rubric_approve(result: ReviewResult) -> bool:
    """Rubric-based approval gate — replaces raw confidence threshold.

    A page approves when:
      - schema_score >= 0.8  (all required sections present)
      - source_score >= 0.8  (no banned URLs, no hallucinated citations)
      - prose_score  >= 0.6  (readable, no nested lists, no roadmap language)
      - mvb_score    >= 0.6  (if MVB expected — specific IDs, runnable recipe)
      - confidence   >= 0.65 (overall quality floor)

    This prevents one bad dimension from tanking an otherwise complete page,
    while blocking pages with genuine structural or source-policy violations.
    """
    return (
        result.schema_score >= 0.8
        and result.source_score >= 0.8
        and result.prose_score >= 0.6
        and result.mvb_score >= 0.6
        and result.frontier_citation_score >= 0.65
        and result.open_questions_score >= 0.4
        and result.confidence >= 0.65
    )


class CriticScore(BaseModel):
    """One critic's output from the review panel."""
    score: float = Field(ge=0.0, le=1.0)
    issues: list[str] = Field(default_factory=list)
    fix_suggestions: list[str] = Field(default_factory=list)


def _run_critic_panel(draft: str, state: WikiPageState) -> dict[str, CriticScore]:
    """Fan out all critic-* skills in parallel; one API call per critic.

    Each critic gets the full draft + its own skill prompt and returns a
    structured CriticScore. Results aggregate into a per-critic dict the
    reviewer's confidence is then derived from.
    """
    from .skills import load_skills
    from concurrent.futures import ThreadPoolExecutor as _TPE, as_completed as _asc

    skills = [s for s in load_skills() if "review" in s.applies_to and s.name.startswith("critic-")]
    if not skills:
        return {}

    # Critics use a dedicated non-reasoning model (see llm.py "critic" role).
    # Reasoning models eat the max_tokens budget on internal CoT before producing
    # JSON, which previously caused LengthFinishReasonError across the panel.
    critic_llm = get_llm("critic", temperature=0.0)
    critic_reviewer = critic_llm.with_structured_output(CriticScore)

    def _run_one(skill) -> tuple[str, CriticScore]:
        prompt = f"""You are running ONE focused critic on a wiki page. Your skill is:

{skill.body}

---
PAGE TO REVIEW:
{draft}

Return your CriticScore JSON. Score only the dimension your skill covers — leave the
broader review to other critics. Be specific in `issues` (point at the section or
the exact phrase you saw) and actionable in `fix_suggestions` (name the edit).
"""
        try:
            result: CriticScore = critic_reviewer.invoke([HumanMessage(content=prompt)])
            return skill.name, result
        except Exception as exc:
            return skill.name, CriticScore(
                score=0.7,
                issues=[f"critic call failed: {type(exc).__name__}"],
                fix_suggestions=[],
            )

    results: dict[str, CriticScore] = {}
    # 8 critics; OpenRouter handles concurrency fine. Cap workers at 8.
    with _TPE(max_workers=min(8, max(1, len(skills))), thread_name_prefix="critic") as pool:
        futures = {pool.submit(_run_one, s): s for s in skills}
        for fut in _asc(futures):
            name, score = fut.result()
            results[name] = score
    return results


def _aggregate_review(
    structured: ReviewResult,
    panel: dict[str, CriticScore],
    revision_count: int = 0,
    prev_panel_min: float | None = None,
) -> tuple[float, bool, list[str], dict[str, float]]:
    """Combine the structured rubric reviewer with the critic panel.

    Returns (confidence, approved, all_issues, per_dim_scores).

    Approval rule (replaces the old strict `all critics ≥ 0.6`):
      - panel_avg ≥ 0.65          (consensus, robust to one outlier)
      - panel_worst ≥ 0.4         (catastrophic floor — one critic at 0.3 still blocks)
      - rubric_ok                 (structured-rubric thresholds, unchanged)
      - Revision-aware escape: if revision_count ≥ 1 AND panel_min hasn't improved
        by ≥ 0.05 since the previous review, accept anyway — the revision-spiral
        won't break through; stop wasting tokens.

    Confidence is the average panel score (with worst-case floor) — gives the
    supervisor a smoother signal than the old min().
    """
    panel_scores = {name: cs.score for name, cs in panel.items()}
    panel_avg = (sum(panel_scores.values()) / len(panel_scores)) if panel_scores else 1.0
    panel_worst = min(panel_scores.values()) if panel_scores else 1.0
    # Composite confidence: average modulated by worst-case
    confidence = min(structured.confidence, panel_avg) if panel_scores else structured.confidence

    rubric_ok = _rubric_approve(structured)
    panel_ok = panel_avg >= 0.65 and panel_worst >= 0.4

    # Revision-spiral escape: if we revised but the worst critic didn't budge,
    # stop chasing it. Better to ship and move on than burn tokens revising
    # to the same flag.
    if (
        not panel_ok
        and revision_count >= 1
        and prev_panel_min is not None
        and panel_worst - prev_panel_min < 0.05
        and rubric_ok
    ):
        panel_ok = True  # accept with the floor it has — the next revision won't help

    approved = rubric_ok and panel_ok

    all_issues = list(structured.issues)
    for name, cs in panel.items():
        for iss in cs.issues:
            all_issues.append(f"[{name}] {iss}")

    per_dim = {
        "schema": structured.schema_score,
        "source": structured.source_score,
        "prose": structured.prose_score,
        "mvb": structured.mvb_score,
        "frontier_citations": structured.frontier_citation_score,
        "open_questions": structured.open_questions_score,
        "backlinks": structured.backlink_score,
        **panel_scores,
    }
    return confidence, approved, all_issues, per_dim


def review_node(state: WikiPageState) -> WikiPageState:
    """Multi-critic panel review.

    Runs in two halves, concurrently:
      1. Structured rubric reviewer (existing schema/source/prose/MVB/citations check)
      2. Critic panel: each critic-* skill spawns one parallel API call
         scoring its single dimension

    The two halves are combined: confidence = min(rubric, min(critic scores)).
    Approval requires both rubric_ok AND every critic >= 0.6.
    Any critic flagging a real problem blocks — this is the conservative gate.
    """
    from concurrent.futures import ThreadPoolExecutor as _TPE

    reviewer = get_llm("reviewer", temperature=0.0)
    draft = state.get("draft", "")

    structured_reviewer = reviewer.with_structured_output(ReviewResult)
    prompt = f"""{REVIEWER_SYSTEM}

---
PAGE TO REVIEW:
{draft}
"""

    # Run the structured rubric reviewer and the critic panel concurrently.
    with _TPE(max_workers=2, thread_name_prefix="review") as pool:
        rubric_future = pool.submit(
            lambda: structured_reviewer.invoke([HumanMessage(content=prompt)])
        )
        panel_future = pool.submit(_run_critic_panel, draft, state)

        try:
            structured: ReviewResult = rubric_future.result()
        except Exception:
            # Same fallback as before — unstructured reviewer
            fallback = reviewer.invoke([
                HumanMessage(content=prompt + (
                    "\n\nRespond in this exact format:\n"
                    "VERDICT: PASS or FAIL\n"
                    "CONFIDENCE: 0.0-1.0\n"
                    "ISSUES: bullet list of issues\n"
                ))
            ])
            text = fallback.content
            import re as _re
            passed = "PASS" in text.upper() and "FAIL" not in text.upper().split("PASS")[0]
            conf_match = _re.search(r"\b(0\.\d+|1\.0)\b", text)
            fallback_conf = float(conf_match.group(1)) if conf_match else 0.7
            # Still try to consume panel results for richer feedback
            try:
                panel = panel_future.result()
            except Exception:
                panel = {}
            panel_min = min((cs.score for cs in panel.values()), default=1.0)
            return {
                **state,
                "review_feedback": text + ("\n\nCritic panel:\n" + "\n".join(
                    f"  {name}: {cs.score:.2f} — {'; '.join(cs.issues[:2])}"
                    for name, cs in panel.items()
                ) if panel else ""),
                "review_pass": passed,
                "review_confidence": min(fallback_conf, panel_min),
                "approved": passed and fallback_conf >= 0.65 and panel_min >= 0.6,
                "critic_panel": {n: {"score": cs.score, "issues": cs.issues}
                                 for n, cs in panel.items()},
            }

        panel = panel_future.result()

    # Revision-aware accept: stash the previous panel_min so _aggregate_review
    # can detect "we revised but the critic didn't budge — accept and move on."
    prev_rubric = state.get("review_rubric") or {}
    prev_panel_min = None
    if prev_rubric:
        # panel scores were stored alongside structured-rubric in review_rubric
        # Pick out the critic-* keys and take their min.
        critic_keys = [k for k in prev_rubric if k.startswith("critic-")]
        if critic_keys:
            prev_panel_min = min(prev_rubric[k] for k in critic_keys)

    confidence, approved, all_issues, per_dim = _aggregate_review(
        structured, panel,
        revision_count=state.get("revision_count", 0),
        prev_panel_min=prev_panel_min,
    )

    # Deterministic checklist enforcement (no LLM call): downgrade for missed items.
    # The writer was given a mandatory list of papers / HF models / concept links;
    # we verify each appears in the draft. Penalty: -0.05 per miss, cap at -0.20.
    checklist = state.get("writing_checklist") or {}
    checklist_misses: list[str] = []
    draft_lower = draft.lower()
    for paper in (checklist.get("must_cite_papers") or []):
        # Extract URL from the citation string; we only need the arxiv ID or domain.
        import re as _re_local
        m = _re_local.search(r"https?://[^\s)]+", paper)
        url = m.group(0) if m else paper
        # Match by URL or by the arxiv id portion (last path segment)
        arxiv_id = url.rsplit("/", 1)[-1].lower()
        if url.lower() not in draft_lower and arxiv_id not in draft_lower:
            checklist_misses.append(f"Missing required citation: {paper[:120]}")
    for hf_id in (checklist.get("must_use_hf_models") or []):
        if hf_id and hf_id.lower() not in draft_lower:
            checklist_misses.append(f"Missing required HuggingFace ID: {hf_id}")
    for slug in (checklist.get("must_link_concepts") or []):
        if slug and f"[[{slug}".lower() not in draft_lower and f"/{slug}".lower() not in draft_lower:
            checklist_misses.append(f"Missing required concept link: [[{slug}]]")

    if checklist_misses:
        penalty = min(0.20, 0.05 * len(checklist_misses))
        confidence = max(0.0, confidence - penalty)
        all_issues = list(all_issues) + checklist_misses
        per_dim["checklist_score"] = max(0.0, 1.0 - penalty * 2)
        # If checklist misses pushed us under the approval bar, flip approved.
        if penalty >= 0.10:
            approved = False
    else:
        per_dim["checklist_score"] = 1.0

    summary = (
        f"PASS: {structured.passed}\nConfidence: {confidence:.2f} "
        f"(rubric={structured.confidence:.2f}, panel_min={min((cs.score for cs in panel.values()), default=1.0):.2f})\n"
        + " · ".join(f"{k}:{v:.2f}" for k, v in per_dim.items())
        + (f"\n\nChecklist misses ({len(checklist_misses)}):\n" + "\n".join(f"  - {m}" for m in checklist_misses) if checklist_misses else "")
        + ("\n\nCritic panel detail:\n" + "\n".join(
            f"  [{name}] {cs.score:.2f} — {'; '.join(cs.issues[:3]) or 'no issues'}"
            for name, cs in panel.items()
        ) if panel else "")
    )

    return {
        **state,
        "review_feedback": summary,
        "review_issues": all_issues,
        "review_pass": structured.passed,
        "review_confidence": confidence,
        "review_rubric": per_dim,
        "critic_panel": {n: {"score": cs.score, "issues": cs.issues,
                              "fixes": cs.fix_suggestions}
                          for n, cs in panel.items()},
        "approved": approved,
    }


def revise_draft_node(state: WikiPageState) -> WikiPageState:
    """Writer LLM revises the draft based on reviewer feedback.

    Includes the full WRITER_SYSTEM + scratch_pad context so the reviser
    has the same verified facts and prose rules as the original writer.
    """
    writer = get_llm("writer", temperature=0.2)
    schema = _load_schema()
    persona = state.get("persona", {})
    scratch_pad = state.get("scratch_pad", "")

    system = (
        WRITER_SYSTEM
        .replace("{domain}", persona.get("domain", "AI/ML"))
        .replace("{schema}", schema[:12000])
    )

    draft = state.get("draft", "")
    feedback = state.get("review_feedback", "")
    issues = state.get("review_issues", [])

    # Capture the pre-revision draft so keep_best_draft_node can knockout-compare
    # after the next review pass. Only stash on the FIRST revision (when prev
    # is empty) — subsequent revisions roll forward via keep_best_draft_node.
    stash_prev = {}
    if not state.get("prev_draft"):
        stash_prev = {
            "prev_draft": draft,
            "prev_review_confidence": state.get("review_confidence", 0.0) or 0.0,
            "prev_review_issues": list(issues or []),
        }

    # Detect if the page was truncated (missing v2 sections)
    v2_required = ["## The territory", "## How it works", "## Where the field is now",
                   "## What's still open", "## Where to read next"]
    missing = [s for s in v2_required if s not in draft]
    truncation_note = ""
    if missing:
        truncation_note = (
            "\n⚠ TRUNCATION DETECTED — these v2 sections are missing and must be written NOW:\n"
            + "\n".join(f"  {s}" for s in missing)
            + "\n"
        )

    writing_plan = state.get("writing_plan", "")
    plan_note = (
        f"\nORIGINAL WRITING PLAN (revision must not contradict this):\n{writing_plan[:400]}\n"
        if writing_plan else ""
    )

    checklist_block = _format_checklist_block(state.get("writing_checklist"))

    prompt = f"""You are revising a Frontier Wiki page. The reviewer flagged specific issues. Fix all of them.
{checklist_block}

REVIEWER FEEDBACK (fix every issue listed):
{feedback}

SPECIFIC ISSUES TO ADDRESS:
{chr(10).join(f"  - {i}" for i in issues) if issues else "(see feedback above)"}
{truncation_note}{plan_note}
WORKING MEMORY (verified facts — use ONLY these citations when adding new content):
{scratch_pad[:3000]}

CURRENT DRAFT TO REVISE:
{draft}

Return the COMPLETE revised page. Include ALL v2 sections (hook → The territory →
How it works → Where the field is now → What's still open → Where to read next →
Build it). Do not omit any section, even ones that don't need changes.
"""

    response = writer.invoke([SystemMessage(content=system), HumanMessage(content=prompt)])
    count = state.get("revision_count", 0) + 1
    return {
        **state,
        **stash_prev,
        "draft": _coerce_text(response.content),
        "revision_count": count,
    }


def _sanitize_draft(text: str) -> str:
    """Strip writer-model artifacts that prevent the file from starting with valid frontmatter.

    Handles three real failure modes seen in production:
      1. Writer wraps the whole page in ```yaml ... ``` or ```markdown ... ```.
      2. Writer prepends a preamble like "Here's the revised page:" or "I'll fix two issues:".
      3. Writer emits an opening fence on a line by itself with no language tag.

    The contract is simple: the saved file MUST start with `---\\n` (YAML frontmatter).
    Anything before the first `---\\n` that isn't whitespace is junk.
    """
    if not text:
        return text
    stripped = text.lstrip()
    # Drop opening code fence (```yaml, ```markdown, ```md, or bare ```)
    if stripped.startswith("```"):
        first_newline = stripped.find("\n")
        if first_newline != -1:
            stripped = stripped[first_newline + 1:]
    # Drop a trailing closing fence
    if stripped.rstrip().endswith("```"):
        stripped = stripped.rstrip()
        stripped = stripped[: stripped.rfind("```")].rstrip() + "\n"
    # If first non-whitespace content isn't frontmatter, scan forward and drop the preamble
    if not stripped.lstrip().startswith("---"):
        idx = stripped.find("\n---\n")
        if idx != -1:
            stripped = stripped[idx + 1:]
        else:
            # Last resort: also try `---` at the very start after dropping leading lines
            idx = stripped.find("---\n")
            if idx != -1:
                stripped = stripped[idx:]
    return stripped


def write_file_node(state: WikiPageState) -> WikiPageState:
    final = _sanitize_draft(state.get("draft", ""))

    # Inject arc breadcrumb immediately after frontmatter closing ---
    arc_context = state.get("arc_context", {})
    if arc_context and arc_context.get("arc_id"):
        arc_id = arc_context["arc_id"]
        position = arc_context.get("position", "?")
        arc_title = arc_context.get("arc_title", arc_id.replace("-", " ").title())
        arc_path = f"../../arcs/{arc_id}.md"
        total_str = f" of {arc_context['total']}" if arc_context.get("total") else ""
        breadcrumb = f"\n> **Arc:** [{arc_title}]({arc_path}) — Step {position}{total_str}\n"
        fm_start = final.find("---\n")
        fm_end = final.find("\n---\n", fm_start + 1)
        if fm_start != -1 and fm_end != -1:
            final = final[:fm_end + 4] + breadcrumb + final[fm_end + 4:]

    write_file(state["output_path"], final)
    return {**state, "final_content": final}


def commit_node(state: WikiPageState) -> WikiPageState:
    topic = state.get("topic", "unknown")
    page_type = state.get("page_type", "core-concept")
    confidence = state.get("review_confidence", 0.0)
    has_mvb = state.get("mvb_decision", False)
    revisions = state.get("revision_count", 0)

    message = (
        f"wiki: {topic} ({page_type}, conf={confidence:.2f}"
        f"{', mvb' if has_mvb else ''}"
        f"{f', rev={revisions}' if revisions else ''})"
    )
    committed = git_commit(
        path=state["output_path"],
        message=message,
        docs_dir=DOCS_DIR,
        confidence=confidence,
    )
    return {**state, "committed": committed}


def log_run_node(state: WikiPageState) -> WikiPageState:
    """Log this generation run to runs/runs.jsonl and update wiki_status.md."""
    try:
        log_run(state)
    except Exception:
        pass  # logging failure should never break the pipeline
    return state


def link_node(state: WikiPageState) -> WikiPageState:
    """Inject real internal links into the draft based on existing wiki pages.

    Runs after write_draft, before review. Scans the filesystem for pages that
    actually exist, uses RESEARCH_MODEL to find semantically related ones, and
    replaces placeholder [[wikilinks]] with real relative markdown links.

    Never creates broken links — only links to files that exist on disk.
    Also updates docs/system/backlinks.json for reverse navigation.
    """
    draft = state.get("draft", "")
    if not draft:
        return state

    try:
        docs_dir = Path(DOCS_DIR)
        existing_pages = _index_existing_pages(docs_dir)

        if not existing_pages:
            return state

        # Remove current page from candidates (no self-links)
        current_slug = f"{state['track']}/{state['topic']}"
        existing_pages.pop(current_slug, None)

        linked_draft = _inject_links(
            draft,
            existing_pages,
            state["track"],
            state["topic"],
            docs_dir,
        )
        return {**state, "draft": linked_draft}
    except Exception:
        return state  # link injection failure must never block publishing


def _index_existing_pages(docs_dir: Path) -> dict[str, dict]:
    """Scan curriculum pages and return {slug: {title, track, slug, abs_path}}.

    v2-aware: pages live at docs/curriculum/core/<track>/<page_type>/<slug>.md
    (where page_type ∈ {concepts, authors, arcs, builds}). Track is the
    directory two levels above the file. Falls back to parent.name for the
    legacy flat layout.

    Indexed by bare slug (not "track/slug") so the writer's inline
    [[wikilinks]] resolve cleanly without needing to know the track.

    Skips stub pages — pure filesystem scan, no LLM.
    """
    curriculum = docs_dir / "curriculum"
    pages: dict[str, dict] = {}

    if not curriculum.exists():
        return pages

    for page_file in curriculum.rglob("*.md"):
        if page_file.name == "index.md":
            continue

        # v2 path: .../curriculum/core/<track>/<page_type>/<slug>.md
        # legacy:  .../curriculum/<track>/<slug>.md
        parts = page_file.parent.parts
        if "core" in parts:
            ci = parts.index("core")
            track = parts[ci + 1] if ci + 1 < len(parts) else page_file.parent.name
            page_type = parts[ci + 2] if ci + 2 < len(parts) else "concepts"
        else:
            track = page_file.parent.name
            page_type = "concepts"

        slug = page_file.stem

        try:
            content = page_file.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        if "🚧" in content or "Agent-generated content pending" in content:
            continue

        title = slug.replace("-", " ").title()
        fm_title = re.search(r"^title:\s*(.+)$", content, re.MULTILINE)
        if fm_title:
            title = fm_title.group(1).strip().strip("\"'")

        # Last-write-wins on collisions; slugs SHOULD be unique across tracks
        pages[slug] = {
            "title": title,
            "track": track,
            "slug": slug,
            "page_type": page_type,
            "abs_path": page_file,
        }

    return pages


def _inject_links(
    draft: str,
    existing_pages: dict[str, dict],
    current_track: str,
    current_topic: str,
    docs_dir: Path,
) -> str:
    """v2 link resolver — turn the writer's inline [[wikilinks]] into real
    relative markdown links.

    The v2 writer prompts produce `[[slug]]` references inside the
    "## Where to read next" paragraph (and occasionally elsewhere). This
    function scans those, looks each one up in existing_pages (indexed by
    bare slug), computes the correct v2 relative path from the current
    file's location, and replaces them. References to pages that don't
    exist on disk are stripped to plain text (no broken links).

    DOES NOT add a "## Connected topics" section or any bibliography —
    that was a v1 pattern. The v2 "Where to read next" paragraph is the
    only connective tissue.

    Best-effort: returns the draft unchanged on any error.
    """
    if not existing_pages:
        return draft

    # Current file location (v2 layout): docs/curriculum/core/<track>/concepts/<topic>.md
    current_path = Path(DOCS_DIR) / "curriculum" / "core" / current_track / "concepts" / f"{current_topic}.md"

    pattern = re.compile(r"\[\[([a-z0-9][a-z0-9\-]*)\]\]")

    resolved_targets: list[str] = []  # for backlinks.json

    def _replace(match: re.Match) -> str:
        slug = match.group(1)
        if slug == current_topic:
            # Self-reference — drop the brackets, keep the word
            return slug.replace("-", " ")
        meta = existing_pages.get(slug)
        if not meta:
            # Page doesn't exist yet — leave the wikilink as-is so a later
            # generation can resolve it. Better than a broken markdown link.
            return f"[[{slug}]]"
        target_path: Path = meta["abs_path"]
        try:
            rel = os.path.relpath(target_path, start=current_path.parent)
        except ValueError:
            return f"[[{slug}]]"  # cross-volume on Windows
        resolved_targets.append(slug)
        return f"[{meta['title']}]({rel})"

    try:
        new_draft = pattern.sub(_replace, draft)
    except Exception:
        return draft

    if resolved_targets:
        _update_backlinks(docs_dir, current_track, current_topic, resolved_targets)

    return new_draft


def _update_backlinks(
    docs_dir: Path,
    current_track: str,
    current_topic: str,
    targets: list[str],
) -> None:
    """Persist forward + reverse link index to docs/system/backlinks.json."""
    system_dir = docs_dir / "system"
    system_dir.mkdir(parents=True, exist_ok=True)
    backlinks_path = system_dir / "backlinks.json"

    try:
        backlinks: dict = {}
        if backlinks_path.exists():
            backlinks = _json.loads(backlinks_path.read_text(encoding="utf-8"))

        source_slug = f"{current_track}/{current_topic}"
        backlinks.setdefault("forward", {})[source_slug] = targets

        for target in targets:
            refs = backlinks.setdefault("reverse", {}).setdefault(target, [])
            if source_slug not in refs:
                refs.append(source_slug)

        backlinks_path.write_text(
            _json.dumps(backlinks, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    except Exception:
        pass


def flag_human_review_node(state: WikiPageState) -> WikiPageState:
    """Land the page on disk as a 'drafted' artifact when review maxes out
    without approval. Never throw away the work — the draft is real content
    the user can polish manually, and a future cycle can re-attempt it."""
    try:
        from rich.console import Console
        Console().print(
            f"[yellow]⚠ Max revisions reached — landing as 'drafted':[/yellow] "
            f"{state['output_path']}\n"
            f"Confidence: {state.get('review_confidence', 0):.2f}\n"
        )
    except ImportError:
        print(f"Landing as drafted (max revs): {state['output_path']}")

    # Write the draft to disk even though it didn't pass full review.
    # The auto-commit downstream will skip git commit if confidence is low,
    # but the file lands so we don't lose work.
    from .tools import write_file as _write_file
    draft = state.get("draft", "")
    if draft:
        try:
            _write_file(state["output_path"], draft)
        except Exception:
            pass

    return {**state, "approved": False, "committed": False}


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def route_after_review(state: WikiPageState) -> str:
    """v2 routing — never discard a reasonable draft.

    - approved → write_file (full commit path)
    - revisions maxed BUT confidence >= 0.6 → write_file anyway as 'drafted'
      (auto-commit decides separately based on GIT_COMMIT_THRESHOLD)
    - revisions maxed AND confidence < 0.6 → flag_human_review (page still
      lands on disk via the new flag_human_review_node)
    - otherwise → revise_draft
    """
    if state.get("approved"):
        return "write_file"
    if state.get("revision_count", 0) >= 2:
        if state.get("review_confidence", 0) >= 0.6:
            return "write_file"
        return "flag_human_review"
    return "revise_draft"
