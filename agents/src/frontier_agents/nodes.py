"""LangGraph node functions — Frontier Wiki editorial agent system.

All LLM calls go through LangChain (ChatOpenAI via OpenRouter or ChatAnthropic direct).
The reviewer uses structured output (Pydantic) for reliable PASS/FAIL + confidence.

Graph flow:
  START → load_persona → read_stub → research
        → [full] write_draft → review
        → [mvb-only] mvb_recipe → merge_mvb → review
        → [approved] write_file → commit → END
        → [rejected, count<2] revise_draft → review
        → [rejected, count≥2] flag_human_review → END
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from .llm import get_llm
from .prompts import MVB_SYSTEM, REVIEWER_SYSTEM, WRITER_SYSTEM
from .state import WikiPageState
from .tools import (
    exa_search,
    git_commit,
    hf_search_datasets,
    hf_search_models,
    load_persona,
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
        # Load by track if topic-level persona not found
        persona = load_persona(state["track"], str(_PERSONAS_DIR))
    output_path = f"{DOCS_DIR}/curriculum/{state['track']}/{state['topic']}.md"
    return {**state, "persona": persona, "output_path": output_path}


def read_stub_node(state: WikiPageState) -> WikiPageState:
    stub = read_stub(state["output_path"])
    return {**state, "existing_stub": stub, "revision_count": 0}


def research_node(state: WikiPageState) -> WikiPageState:
    """Exa search for arXiv, HuggingFace, .edu, and engineering blogs."""
    topic = state["topic"].replace("-", " ")
    persona = state.get("persona", {})

    base_queries = [
        f"{topic} deep learning arxiv paper",
        f"{topic} {persona.get('domain', '')} tutorial survey",
        f"huggingface {topic} implementation model",
    ]
    extra = (persona.get("search_seeds") or [])[:2]

    all_results: list[dict] = []
    for q in base_queries + extra:
        try:
            all_results.extend(exa_search(q, num_results=5))
        except Exception:
            pass

    production_results: list[dict] = []
    try:
        production_results = exa_search(
            f"{topic} production deployment engineering at scale",
            num_results=4,
            section="in_production",
        )
    except Exception:
        pass

    # Dedup by URL
    seen: set[str] = set()
    deduped = []
    for r in all_results + production_results:
        if r["url"] not in seen:
            seen.add(r["url"])
            deduped.append(r)

    return {**state, "research_results": deduped}


def write_draft_node(state: WikiPageState) -> WikiPageState:
    """Write a full wiki page using the writer LLM (claude-opus-4-7 via OpenRouter)."""
    writer = get_llm("writer", temperature=0.3)
    schema = _load_schema()
    persona = state.get("persona", {})
    research = state.get("research_results", [])
    existing = state.get("existing_stub", "")

    research_text = "\n\n".join(
        f"SOURCE: {r['url']}\nTITLE: {r['title']}\n{r['text'][:800]}"
        for r in research[:12]
    )

    # Use manual replacement to avoid KeyError from LaTeX {L} in WRITER_SYSTEM
    system = (
        WRITER_SYSTEM
        .replace("{domain}", persona.get("domain", "AI/ML"))
        .replace("{schema}", schema[:3000])
        + f"\nTrack: {persona.get('domain', '')} | Seminal authors: {', '.join(persona.get('seminal_authors', []))}\nProduction examples to draw from: {', '.join(persona.get('production_examples', [])[:3])}"
    )

    improve_block = (
        "IMPROVE THIS EXISTING CONTENT:\n" + existing[:3000]
        if existing.strip() and "🚧" not in existing
        else "Write from scratch."
    )

    user = f"""Write a complete Frontier Wiki page for: **{state['topic']}**
Track: {state['track']} | Depth: {state.get('depth', 'all')}

{improve_block}

RESEARCH SOURCES (verify URLs before citing — use only approved domains):
{research_text}

Produce the complete page with all schema sections. For the MVB section, use real HuggingFace model/dataset IDs.
For "In production", name specific companies and link to official sources only.
Annotate every LaTeX equation — explain what each variable means.
"""

    response = writer.invoke([SystemMessage(content=system), HumanMessage(content=user)])
    return {**state, "draft": response.content}


def mvb_recipe_node(state: WikiPageState) -> WikiPageState:
    """On-demand: generate only the MVB section using HuggingFace search + LLM."""
    writer = get_llm("mvb", temperature=0.2)
    topic = state["topic"].replace("-", " ")

    hf_models, hf_datasets = [], []
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
        # Wrap MVB in minimal page structure
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
        # Replace existing MVB section
        before, rest = existing.split("## Minimum Valuable Build", 1)
        next_section = rest.find("\n## ", 1)
        after = rest[next_section:] if next_section != -1 else ""
        draft = before + mvb + "\n" + after
    else:
        # Insert before "## Code & implementations" or append
        marker = "## Code & implementations"
        if marker in existing:
            draft = existing.replace(marker, mvb + "\n\n" + marker, 1)
        else:
            draft = existing + "\n\n" + mvb

    return {**state, "draft": draft}


def review_node(state: WikiPageState) -> WikiPageState:
    """Structured review using Gemini 2.5 Pro via OpenRouter."""
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
            "review_feedback": f"PASS: {result.passed}\nConfidence: {result.confidence}\nIssues: {result.issues}\nSuggestions: {result.suggestions}",
            "review_pass": result.passed,
            "review_confidence": result.confidence,
            "approved": result.passed and result.confidence >= 0.8,
        }
    except Exception as e:
        # If structured output fails, fallback to text parsing
        fallback = reviewer.invoke([HumanMessage(content=prompt + "\n\nRespond: PASS or FAIL, then confidence score 0.0-1.0, then issues.")])
        text = fallback.content
        passed = "PASS" in text.upper()
        confidence = 0.75  # conservative default
        return {
            **state,
            "review_feedback": text,
            "review_pass": passed,
            "review_confidence": confidence,
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

Fix all issues listed. Apply suggestions where they improve quality. Return the complete revised page.
"""

    response = writer.invoke([HumanMessage(content=prompt)])
    count = state.get("revision_count", 0) + 1
    return {**state, "draft": response.content, "revision_count": count}


def write_file_node(state: WikiPageState) -> WikiPageState:
    final = state.get("draft", "")
    write_file(state["output_path"], final)
    return {**state, "final_content": final}


def commit_node(state: WikiPageState) -> WikiPageState:
    committed = git_commit(
        path=state["output_path"],
        message=f"wiki: add {state['topic']} page (agent-generated)",
        docs_dir=DOCS_DIR,
    )
    return {**state, "committed": committed}


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
