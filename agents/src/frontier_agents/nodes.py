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
from .prompts import (
    MVB_SYSTEM,
    PLAN_SYSTEM,
    REVIEWER_SYSTEM,
    SCRATCH_SYSTEM,
    WRITE_INSTRUCTIONS,
    WRITER_SYSTEM,
)
from .skills import context_tokens_from_state, format_skills_block, load_skills, select_skills
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

# Loaded once at import time; safe because skills files are read-only at runtime.
_SKILLS = load_skills()


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


def scratch_node(state: WikiPageState) -> WikiPageState:
    """Compile raw research into a structured working-memory fact sheet.

    Uses the fast RESEARCH_MODEL — this is synthesis/extraction, not prose.
    The scratch_pad replaces raw research results in all subsequent writing calls:
    the writer never sees unprocessed search results, only the compiled facts.
    """
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
    response = compiler.invoke([HumanMessage(content=user)])
    return {**state, "scratch_pad": response.content}


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
        .replace("{schema}", schema[:2000])
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

    user = f"""Topic: **{topic}** | Track: {track} | Page type: {page_type}
{depth_note}

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
    draft = response.content.strip()
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
{draft}
"""

    try:
        result: ReviewResult = structured_reviewer.invoke([HumanMessage(content=prompt)])
        return {
            **state,
            "review_feedback": (
                f"PASS: {result.passed}\nConfidence: {result.confidence}\n"
                f"Issues: {result.issues}\nSuggestions: {result.suggestions}"
            ),
            "review_issues": result.issues,
            "review_pass": result.passed,
            "review_confidence": result.confidence,
            "approved": result.passed and result.confidence >= float(
                os.getenv("GIT_COMMIT_THRESHOLD", "0.8")
            ),
        }
    except Exception:
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
        # Parse confidence from text — look for decimal like "0.82" or "0.9"
        conf_match = _re.search(r"\b(0\.\d+|1\.0)\b", text)
        fallback_conf = float(conf_match.group(1)) if conf_match else 0.7
        threshold = float(os.getenv("GIT_COMMIT_THRESHOLD", "0.8"))

        return {
            **state,
            "review_feedback": text,
            "review_pass": passed,
            "review_confidence": fallback_conf,
            "approved": passed and fallback_conf >= threshold,
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
        .replace("{schema}", schema[:2000])
    )

    prompt = f"""You are revising a Frontier Wiki page. The reviewer flagged specific issues. Fix them.

REVIEWER FEEDBACK (fix every issue listed):
{state.get('review_feedback', '')}

WORKING MEMORY (verified facts — use ONLY these citations when adding new content):
{scratch_pad[:3000]}

CURRENT DRAFT TO REVISE:
{state.get('draft', '')}

Return the complete revised page. Maintain all sections. Fix the flagged issues without breaking what's already good.
"""

    response = writer.invoke([SystemMessage(content=system), HumanMessage(content=prompt)])
    count = state.get("revision_count", 0) + 1
    return {**state, "draft": response.content, "revision_count": count}


def write_file_node(state: WikiPageState) -> WikiPageState:
    final = state.get("draft", "")
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
    """Scan curriculum pages and return {track/slug: {title, track, slug, concepts_snippet}}.

    Skips stub pages (no real content to link to).
    Pure filesystem scan — no LLM needed.
    """
    curriculum = docs_dir / "curriculum"
    pages: dict[str, dict] = {}

    if not curriculum.exists():
        return pages

    for page_file in curriculum.rglob("*.md"):
        if page_file.name == "index.md":
            continue

        track = page_file.parent.name
        slug = page_file.stem
        full_slug = f"{track}/{slug}"

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

        concepts_snippet = ""
        core_match = re.search(
            r"## Core concepts?\s*\n+(.*?)(?:\n\n|\n#)", content, re.DOTALL
        )
        if core_match:
            for line in core_match.group(1).split("\n"):
                line = line.strip().lstrip("*-").strip()
                if line and not line.startswith("**") and len(line) > 20:
                    concepts_snippet = line[:200]
                    break

        pages[full_slug] = {
            "title": title,
            "track": track,
            "slug": slug,
            "concepts_snippet": concepts_snippet,
        }

    return pages


def _inject_links(
    draft: str,
    existing_pages: dict[str, dict],
    current_track: str,
    current_topic: str,
    docs_dir: Path,
) -> str:
    """Use RESEARCH_MODEL to find related existing pages and inject real links.

    Returns the draft unchanged on any error — link injection is best-effort.
    """
    if not existing_pages:
        return draft

    linker = get_llm("research", temperature=0.0)

    catalog_lines = [
        f"- `{slug}` | {meta['title']}"
        + (f" — {meta['concepts_snippet'][:80]}" if meta["concepts_snippet"] else "")
        for slug, meta in list(existing_pages.items())[:60]
    ]
    catalog = "\n".join(catalog_lines)

    prompt = f"""You are a knowledge graph linker for a machine learning wiki.

Current page: **{current_topic}** (track: {current_track})

Existing wiki pages (slug | title — snippet):
{catalog}

Task: Identify up to 6 existing pages most directly related to "{current_topic}".
For each, write one sentence explaining the relationship (max 15 words).

Return ONLY a JSON array — no explanation, no markdown fences:
[
  {{"slug": "track/page-slug", "title": "Page Title", "relationship": "one sentence"}},
  ...
]

Only use slugs from the catalog above. Never invent slugs."""

    try:
        response = linker.invoke([HumanMessage(content=prompt)])
        raw = response.content.strip()
        json_match = re.search(r"\[.*?\]", raw, re.DOTALL)
        if not json_match:
            return draft
        links_data = _json.loads(json_match.group(0))
    except Exception:
        return draft

    if not links_data:
        return draft

    link_entries = []
    backlink_targets = []

    for item in links_data:
        slug = item.get("slug", "")
        if slug not in existing_pages:
            continue  # never create links to pages that don't exist

        meta = existing_pages[slug]
        title = meta["title"]
        target_track = meta["track"]
        target_slug = meta["slug"]
        relationship = item.get("relationship", "")

        rel_path = (
            f"./{target_slug}.md"
            if target_track == current_track
            else f"../{target_track}/{target_slug}.md"
        )

        link_entries.append(f"- [{title}]({rel_path}) — {relationship}")
        backlink_targets.append(slug)

    if not link_entries:
        return draft

    links_block = "\n".join(link_entries)

    # Replace content of ## Connected topics section if it exists
    connected_pattern = r"(## Connected topics?\s*\n)(.*?)(?=\n## |\Z)"
    if re.search(connected_pattern, draft, re.DOTALL):
        draft = re.sub(
            connected_pattern,
            lambda m: m.group(1) + links_block + "\n\n",
            draft,
            flags=re.DOTALL,
        )
    elif "## Further reading" in draft:
        draft = draft.replace(
            "## Further reading",
            f"## Connected topics\n\n{links_block}\n\n## Further reading",
            1,
        )
    else:
        draft += f"\n\n## Connected topics\n\n{links_block}\n"

    _update_backlinks(docs_dir, current_track, current_topic, backlink_targets)
    return draft


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
