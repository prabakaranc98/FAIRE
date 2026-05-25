"""LangGraph node functions — Frontier Wiki editorial agent system.

All LLM calls go through LangChain (ChatOpenAI via OpenRouter or ChatAnthropic direct).
The reviewer uses structured output (Pydantic) for reliable PASS/FAIL + confidence.

Graph flow:
  START → load_persona → read_stub → research → plan
        → [full] write_draft → review
        → [mvb-only] mvb_recipe → merge_mvb → review
        → [approved] write_file → commit → log_run → END
        → [rejected, count<2] revise_draft → review
        → [rejected, count≥2] flag_human_review → log_run → END
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from .llm import get_llm
from .prompts import MVB_SYSTEM, PLAN_SYSTEM, REVIEWER_SYSTEM, WRITER_SYSTEM
from .state import WikiPageState
from .tools import (
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


def _load_schema() -> str:
    return _SCHEMA_PATH.read_text(encoding="utf-8") if _SCHEMA_PATH.exists() else ""


# ---------------------------------------------------------------------------
# Pydantic model for structured reviewer output
# ---------------------------------------------------------------------------

class ReviewResult(BaseModel):
    """Structured review output — eliminates fragile text parsing."""
    passed: bool = Field(description="True if the page passes schema + source policy checks")
    confidence: float = Field(ge=0.0, le=1.0, description="Quality confidence score")
    issues: list[str] = Field(description="Specific actionable issues to fix")
    suggestions: list[str] = Field(description="Optional improvement suggestions")


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def load_persona_node(state: WikiPageState) -> WikiPageState:
    persona = load_persona(state["topic"], str(_PERSONAS_DIR))
    if persona.get("track") != state["track"]:
        persona = load_persona(state["track"], str(_PERSONAS_DIR))
    output_path = f"{DOCS_DIR}/curriculum/{state['track']}/{state['topic']}.md"
    run_id = str(uuid.uuid4())
    started_at = datetime.now(timezone.utc).isoformat()
    return {**state, "persona": persona, "output_path": output_path,
            "run_id": run_id, "started_at": started_at}


def read_stub_node(state: WikiPageState) -> WikiPageState:
    stub = read_stub(state["output_path"])
    return {**state, "existing_stub": stub, "revision_count": 0}


def research_node(state: WikiPageState) -> WikiPageState:
    """Three-phase Exa search: foundational papers, SotA (2024+), production deployments."""
    topic = state["topic"].replace("-", " ")
    persona = state.get("persona", {})
    domain = persona.get("domain", "")
    seeds = (persona.get("search_seeds") or [])[:2]

    # Phase 1: foundational papers
    paper_results: list[dict] = []
    for q in [
        f"{topic} foundational paper",
        f"{topic} {domain} original contribution",
        *seeds,
    ]:
        try:
            paper_results.extend(exa_search_papers(q, foundational=True))
        except Exception:
            pass

    # Phase 2: SotA (2024+ papers with highlights)
    sota_results: list[dict] = []
    for q in [
        f"{topic} state of the art 2024 2025 benchmark",
        f"{topic} best result arxiv 2025",
    ]:
        try:
            sota_results.extend(exa_search_sota(q))
        except Exception:
            pass

    # Phase 3: production deployments (engineering blogs only)
    production_results: list[dict] = []
    try:
        production_results = exa_search_production(
            f"{topic} production deployment engineering at scale"
        )
    except Exception:
        pass

    # HuggingFace models/datasets for MVB research
    hf_models, hf_datasets = [], []
    try:
        hf_models = hf_search_models(topic)
        hf_datasets = hf_search_datasets(topic)
    except Exception:
        pass

    # Dedup paper results by URL
    seen: set[str] = set()
    deduped_papers = []
    for r in paper_results:
        if r["url"] not in seen:
            seen.add(r["url"])
            deduped_papers.append(r)

    return {
        **state,
        "research_results": deduped_papers,
        "sota_results": sota_results,
        "production_results": production_results,
        "hf_models": hf_models,
        "hf_datasets": hf_datasets,
    }


def plan_node(state: WikiPageState) -> WikiPageState:
    """Deliberate planning step before writing — forces the agent to think first.

    Uses RESEARCH_MODEL (fast, capable) since this is a reasoning/synthesis task,
    not prose generation. Output becomes writing_plan in state.
    """
    planner = get_llm("research", temperature=0.2)
    topic = state["topic"].replace("-", " ")
    page_type = state.get("page_type", "core-concept")
    depth_emphasis = state.get("depth_emphasis", ["applied"])
    arc_context = state.get("arc_context", {})

    # Summarize research for the planner (keep it tight)
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

    response = planner.invoke([HumanMessage(content=user)])
    return {**state, "writing_plan": response.content}


def write_draft_node(state: WikiPageState) -> WikiPageState:
    """Write a full wiki page — guided by the writing plan from plan_node."""
    writer = get_llm("writer", temperature=0.3)
    schema = _load_schema()
    persona = state.get("persona", {})
    existing = state.get("existing_stub", "")
    writing_plan = state.get("writing_plan", "")
    page_type = state.get("page_type", "core-concept")
    depth_emphasis = state.get("depth_emphasis", ["applied"])

    # Compile research into a source block for the writer
    def _fmt_papers(results: list[dict], max_items: int = 8) -> str:
        return "\n\n".join(
            f"PAPER: {r['url']}\nTITLE: {r['title']}\n{r.get('text', '')[:600]}"
            for r in results[:max_items]
        )

    def _fmt_sota(results: list[dict], max_items: int = 4) -> str:
        return "\n\n".join(
            f"SOTA: {r['url']}\nTITLE: {r['title']}\nHIGHLIGHTS: {'; '.join(r.get('highlights', [])[:2])}"
            for r in results[:max_items]
        )

    def _fmt_prod(results: list[dict], max_items: int = 3) -> str:
        return "\n\n".join(
            f"PRODUCTION: {r['url']}\nTITLE: {r['title']}\nSUMMARY: {r.get('summary', '')[:300]}"
            for r in results[:max_items]
        )

    research_block = "\n\n".join(filter(None, [
        "=== FOUNDATIONAL PAPERS ===",
        _fmt_papers(state.get("research_results", [])),
        "=== CURRENT SotA (2024+) ===",
        _fmt_sota(state.get("sota_results", [])),
        "=== PRODUCTION DEPLOYMENTS ===",
        _fmt_prod(state.get("production_results", [])),
    ]))

    hf_models = state.get("hf_models", [])
    hf_block = "\n".join(
        f"- {m['model_id']} ({m['downloads']:,} downloads)"
        for m in hf_models[:5]
    ) or "Search huggingface.co/models for this topic"

    # Build system prompt — use .replace() to avoid KeyError from LaTeX {braces}
    system = (
        WRITER_SYSTEM
        .replace("{domain}", persona.get("domain", "AI/ML"))
        .replace("{schema}", schema[:3000])
    )

    improve_block = (
        "IMPROVE THIS EXISTING CONTENT — keep what works, rewrite what doesn't:\n" + existing[:3000]
        if existing.strip() and "🚧" not in existing
        else "Write from scratch — there is no existing content to preserve."
    )

    depth_note = (
        "DEPTH EMPHASIS for this page: " + ", ".join(depth_emphasis) + "\n"
        + ("Lean into the applied/engineering angle — more MVB detail, real code patterns, "
           "production framing.\n" if "applied" in depth_emphasis else "")
        + ("Lean into the theoretical angle — derivation steps, formal definitions, "
           "proof intuitions.\n" if "theoretical" in depth_emphasis else "")
        + ("Lean into the frontier angle — specific benchmark numbers, named 2024-2025 "
           "papers, open problems as precise questions.\n" if "frontier" in depth_emphasis else "")
    )

    user = f"""Write a complete Frontier Wiki page for: **{state['topic']}**
Track: {state['track']} | Page type: {page_type}

{depth_note}

WRITING PLAN (from planning agent — follow this):
{writing_plan}

{improve_block}

RESEARCH SOURCES (verify URLs before citing — use only approved domains):
{research_block}

HuggingFace models available for MVB (use exact IDs):
{hf_block}

Produce the complete page with ALL schema sections. Requirements:
- Every "What it is" / "Why it matters" paragraph must be prose — NO nested lists
- Every LaTeX variable must be annotated on the following line
- "In production": name specific companies + systems + scale numbers + official source links
- MVB: use exact HuggingFace model/dataset IDs from the list above
- End with the GitHub star CTA and "What can you build next?" arc connector
"""

    response = writer.invoke([SystemMessage(content=system), HumanMessage(content=user)])
    return {**state, "draft": response.content}


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
    return {**state, "mvb_section": response.content, "hf_models": hf_models, "hf_datasets": hf_datasets}


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


def review_node(state: WikiPageState) -> WikiPageState:
    """Structured review using Gemini 3.1 Pro Preview via OpenRouter."""
    reviewer = get_llm("reviewer", temperature=0.0)
    draft = state.get("draft", "")

    structured_reviewer = reviewer.with_structured_output(ReviewResult)

    prompt = f"""{REVIEWER_SYSTEM}

---
PAGE TO REVIEW:
{draft[:6000]}
"""

    try:
        result: ReviewResult = structured_reviewer.invoke([HumanMessage(content=prompt)])
        return {
            **state,
            "review_feedback": (
                f"PASS: {result.passed}\nConfidence: {result.confidence}\n"
                f"Issues: {result.issues}\nSuggestions: {result.suggestions}"
            ),
            "review_pass": result.passed,
            "review_confidence": result.confidence,
            "approved": result.passed and result.confidence >= 0.8,
        }
    except Exception:
        fallback = reviewer.invoke([
            HumanMessage(content=prompt + "\n\nRespond: PASS or FAIL, then confidence 0.0-1.0, then issues.")
        ])
        text = fallback.content
        passed = "PASS" in text.upper()
        return {
            **state,
            "review_feedback": text,
            "review_pass": passed,
            "review_confidence": 0.75,
            "approved": passed,
        }


def revise_draft_node(state: WikiPageState) -> WikiPageState:
    """Writer LLM revises the draft based on reviewer feedback."""
    writer = get_llm("writer", temperature=0.2)

    prompt = f"""Revise this Frontier Wiki page based on the reviewer's feedback.

REVIEWER FEEDBACK:
{state.get('review_feedback', '')}

CURRENT DRAFT:
{state.get('draft', '')}

Fix all issues listed. Pay special attention to:
- Convert any nested lists in explanatory sections to flowing prose
- Ensure LaTeX variables are annotated
- Replace vague "large companies" with specific named companies + scale numbers
- Verify all source URLs are from approved domains

Return the complete revised page.
"""

    response = writer.invoke([HumanMessage(content=prompt)])
    count = state.get("revision_count", 0) + 1
    return {**state, "draft": response.content, "revision_count": count}


def write_file_node(state: WikiPageState) -> WikiPageState:
    final = state.get("draft", "")
    write_file(state["output_path"], final)
    return {**state, "final_content": final}


def commit_node(state: WikiPageState) -> WikiPageState:
    topic = state.get("topic", "unknown")
    page_type = state.get("page_type", "core-concept")
    committed = git_commit(
        path=state["output_path"],
        message=f"wiki: add {topic} ({page_type}, agent-generated)",
        docs_dir=DOCS_DIR,
    )
    return {**state, "committed": committed}


def log_run_node(state: WikiPageState) -> WikiPageState:
    """Log this generation run to runs/runs.jsonl and update wiki_status.md."""
    try:
        log_run(state)
    except Exception:
        pass  # logging failure should never break the pipeline
    return state


def flag_human_review_node(state: WikiPageState) -> WikiPageState:
    try:
        from rich.console import Console
        Console().print(
            f"[yellow]⚠ Human review required:[/yellow] {state['output_path']}\n"
            f"Confidence: {state.get('review_confidence', 0):.2f}\n"
            f"{state.get('review_feedback', '')}"
        )
    except ImportError:
        print(f"Human review required: {state['output_path']}")
    return {**state, "approved": False, "committed": False}


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def route_after_review(state: WikiPageState) -> str:
    if state.get("approved"):
        return "write_file"
    if state.get("revision_count", 0) >= 2:
        return "flag_human_review"
    return "revise_draft"
