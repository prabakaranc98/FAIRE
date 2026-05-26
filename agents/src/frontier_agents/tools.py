"""Tools used by the editorial agent system.

All tools are pure functions that take typed inputs and return typed outputs.
They are called from nodes.py, not directly from the LangGraph graph.

Source policy (enforced at search time):
  - Default: arxiv.org, *.edu, huggingface.co, official library docs
  - "In production" section only: official engineering blogs allowed

Exa search API notes (2026-05-25):
  - Use type="auto" (not "neural" — that's legacy terminology)
  - category="research paper" dramatically improves paper discovery
  - contents.highlights=True is the recommended default for agent workflows
  - useAutoprompt is deprecated and removed
  - search_and_contents is deprecated; use search() with ContentsOptions
"""

from __future__ import annotations

import json
import os
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path


APPROVED_RESEARCH_DOMAINS = [
    "arxiv.org",
    "huggingface.co",
    ".edu",
    "pytorch.org",
    "jax.readthedocs.io",
    "openai.com/research",
    "anthropic.com",
    "deepmind.google",
]

APPROVED_ENGINEERING_BLOGS = [
    "engineering.linkedin.com",
    "ai.meta.com/research",
    "developer.nvidia.com/blog",
    "research.google",
    "openai.com/research",
    "blog.google",
    "aws.amazon.com/blogs/machine-learning",
    "techblog.netflix.com",
    "databricks.com/blog",
    "stability.ai/research",
]

# Keep the old name for backwards compatibility in tests
APPROVED_DOMAINS = APPROVED_RESEARCH_DOMAINS


def _get_exa():
    try:
        from exa_py import Exa
    except ImportError:
        raise ImportError("exa-py not installed. Run: uv sync")
    api_key = os.getenv("EXA_API_KEY")
    if not api_key:
        raise ValueError("EXA_API_KEY not set in environment. See .env.example")
    return Exa(api_key=api_key)


def exa_search_papers(
    query: str,
    num_results: int | None = None,
    foundational: bool = True,
) -> list[dict]:
    """Search for academic papers on arXiv and .edu.

    Args:
        query: semantic search query (describe the paper's contribution, not just keywords)
        num_results: defaults to EXA_NUM_RESULTS_PAPERS env var or 8
        foundational: True → all-time papers; False → 2024+ only (for SotA)

    Returns:
        List of {url, title, text, domain, published_date}
    """
    n = num_results or int(os.getenv("EXA_NUM_RESULTS_PAPERS", "8"))
    exa = _get_exa()

    kwargs: dict = dict(
        query=query,
        num_results=n,
        type="auto",
        category="research paper",
        include_domains=["arxiv.org", ".edu"],
        contents={"text": {"max_characters": 4000}},
    )
    if not foundational:
        kwargs["start_published_date"] = "2024-01-01"

    results = exa.search(**kwargs)
    return [
        {
            "url": r.url,
            "title": r.title or "",
            "text": getattr(r, "text", "") or "",
            "domain": _extract_domain(r.url),
            "published_date": getattr(r, "published_date", None),
            "source_type": "paper",
        }
        for r in results.results
    ]


def exa_search_sota(query: str, num_results: int | None = None) -> list[dict]:
    """Search for current SotA — recent papers (2024+) with highlights.

    Uses highlights (not full text) to avoid flooding the LLM context window.
    Best for finding specific benchmark numbers and named systems.

    Returns:
        List of {url, title, highlights, domain, published_date}
    """
    n = num_results or int(os.getenv("EXA_NUM_RESULTS_SOTA", "6"))
    exa = _get_exa()

    results = exa.search(
        query=query,
        num_results=n,
        type="auto",
        category="research paper",
        include_domains=["arxiv.org", ".edu", "huggingface.co"],
        start_published_date="2024-01-01",
        contents={"highlights": True},
    )
    return [
        {
            "url": r.url,
            "title": r.title or "",
            "highlights": getattr(r, "highlights", []) or [],
            "domain": _extract_domain(r.url),
            "published_date": getattr(r, "published_date", None),
            "source_type": "sota",
        }
        for r in results.results
    ]


def exa_search_production(query: str, num_results: int | None = None) -> list[dict]:
    """Search engineering blogs for production deployments at scale.

    Uses LLM-generated summaries targeted to deployment context.
    Only searches approved engineering blog domains.

    Returns:
        List of {url, title, summary, domain}
    """
    n = num_results or int(os.getenv("EXA_NUM_RESULTS_PROD", "5"))
    exa = _get_exa()

    results = exa.search(
        query=query,
        num_results=n,
        type="auto",
        include_domains=APPROVED_ENGINEERING_BLOGS,
        contents={
            "summary": {
                "query": f"production deployment at scale, architecture, real numbers: {query}"
            }
        },
    )
    return [
        {
            "url": r.url,
            "title": r.title or "",
            "summary": getattr(r, "summary", "") or "",
            "domain": _extract_domain(r.url),
            "source_type": "production",
        }
        for r in results.results
    ]


def exa_search(
    query: str,
    num_results: int = 8,
    section: str = "default",
) -> list[dict]:
    """Legacy unified search — kept for backwards compatibility with existing tests.

    Prefer exa_search_papers / exa_search_sota / exa_search_production in new code.
    """
    include_domains = list(APPROVED_RESEARCH_DOMAINS)
    if section == "in_production":
        include_domains.extend(APPROVED_ENGINEERING_BLOGS)

    exa = _get_exa()

    results = exa.search(
        query=query,
        num_results=num_results,
        type="auto",
        include_domains=include_domains,
        contents={"text": {"max_characters": 2000}},
    )
    return [
        {
            "url": r.url,
            "title": r.title or "",
            "text": getattr(r, "text", "") or "",
            "domain": _extract_domain(r.url),
            "published_date": getattr(r, "published_date", None),
        }
        for r in results.results
    ]


def hf_search_models(query: str, limit: int = 5) -> list[dict]:
    """Search HuggingFace Hub for models relevant to a topic.

    Returns: List of {model_id, downloads, likes, url}
    """
    try:
        import httpx
    except ImportError:
        raise ImportError("httpx not installed. Run: uv sync")

    resp = httpx.get(
        "https://huggingface.co/api/models",
        params={"search": query, "limit": limit, "sort": "downloads", "direction": -1},
        timeout=10,
    )
    resp.raise_for_status()

    return [
        {
            "model_id": m.get("modelId", ""),
            "downloads": m.get("downloads", 0),
            "likes": m.get("likes", 0),
            "url": f"https://huggingface.co/{m.get('modelId', '')}",
        }
        for m in resp.json()
    ]


def hf_search_datasets(query: str, limit: int = 5) -> list[dict]:
    """Search HuggingFace Hub for datasets relevant to a topic."""
    try:
        import httpx
    except ImportError:
        raise ImportError("httpx not installed. Run: uv sync")

    resp = httpx.get(
        "https://huggingface.co/api/datasets",
        params={"search": query, "limit": limit, "sort": "downloads", "direction": -1},
        timeout=10,
    )
    resp.raise_for_status()

    return [
        {
            "dataset_id": d.get("id", ""),
            "downloads": d.get("downloads", 0),
            "likes": d.get("likes", 0),
            "url": f"https://huggingface.co/datasets/{d.get('id', '')}",
        }
        for d in resp.json()
    ]


def read_stub(path: str) -> str:
    """Read an existing wiki page stub or return empty string if not found."""
    p = Path(path)
    if p.exists():
        return p.read_text(encoding="utf-8")
    return ""


def write_file(path: str, content: str) -> None:
    """Write content to a wiki page, creating parent directories if needed."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")


def ensure_track_index(track: str, docs_dir: str = "../docs") -> bool:
    """Create a minimal track index.md if one doesn't already exist.

    Returns True if created, False if already existed.
    """
    index_path = Path(docs_dir) / "curriculum" / track / "index.md"
    if index_path.exists():
        return False

    track_title = track.split("-", 1)[-1].replace("-", " ").title()
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(
        f"# {track_title}\n\n"
        f"> This track covers {track_title.lower()}.\n\n"
        f"Pages in this track are generated by the Frontier Wiki editorial agent.\n",
        encoding="utf-8",
    )
    return True


def update_mkdocs_nav(topic: str, track: str, mkdocs_path: str = "../mkdocs.yml") -> bool:
    """Add a newly generated page to mkdocs.yml nav if not already present.

    Returns True if nav was updated, False if entry already existed.
    This is best-effort — if the nav section for the track doesn't exist yet,
    no change is made (the user can add it manually or via scaffold_arc).
    """
    try:
        import yaml
    except ImportError:
        return False

    p = Path(mkdocs_path)
    if not p.exists():
        return False

    content = p.read_text(encoding="utf-8")
    page_ref = f"curriculum/{track}/{topic}.md"

    if page_ref in content:
        return False  # already in nav

    # Best-effort: append to nav if we find the track section
    # Full nav management is handled by scaffold_arc for batch operations
    return False


def update_arc_json(arc_id: str, node_id: str, data: dict, arcs_json_path: str) -> None:
    """Patch docs/arcs.json with updated node data (e.g., status, doc_path)."""
    p = Path(arcs_json_path)
    if not p.exists():
        return
    arcs_data = json.loads(p.read_text(encoding="utf-8"))

    for arc in arcs_data.get("arcs", []):
        if arc["id"] == arc_id:
            for node in arc.get("nodes", []):
                if node["id"] == node_id:
                    node.update(data)
                    break
            break

    p.write_text(json.dumps(arcs_data, indent=2, ensure_ascii=False), encoding="utf-8")


def git_commit(
    path: str,
    message: str,
    docs_dir: str = "../docs",
    confidence: float = 1.0,
    confidence_threshold: float | None = None,
) -> bool:
    """Stage and commit a wiki page. Returns True on success.

    Commit is gated by agent confidence:
      - If confidence >= threshold (default 0.8 or GIT_COMMIT_THRESHOLD env var): commit
      - If GIT_AUTO_PUSH=true and commit succeeded: also push to origin
      - GIT_AUTO_COMMIT=false overrides everything (safety switch for tests)

    This means agents decide whether a page is good enough to commit —
    not a static env flag set before the run.
    """
    if os.getenv("GIT_AUTO_COMMIT", "true").lower() == "false":
        return False

    threshold = confidence_threshold or float(os.getenv("GIT_COMMIT_THRESHOLD", "0.8"))
    if confidence < threshold:
        return False

    try:
        repo_root = Path(docs_dir).parent.resolve()
        subprocess.run(["git", "add", path], cwd=repo_root, check=True, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", message],
            cwd=repo_root,
            check=True,
            capture_output=True,
        )
        if os.getenv("GIT_AUTO_PUSH", "false").lower() == "true":
            subprocess.run(
                ["git", "push", "origin", "main"],
                cwd=repo_root,
                check=True,
                capture_output=True,
            )
        return True
    except subprocess.CalledProcessError:
        return False


def log_run(state: dict, runs_dir: str = "runs") -> None:
    """Append a run record to runs/runs.jsonl and sync live docs/system/ pages.

    Called at the end of every agent run regardless of outcome.
    Tracks: topic, model, confidence, revision count, status, timing.

    Writes to two places:
      agents/runs/runs.jsonl      — append-only machine log
      docs/system/status.md       — live coverage table served by MkDocs
      docs/system/changelog.md    — per-run changelog visible on the site
    """
    runs_path = Path(__file__).parent.parent.parent / runs_dir
    runs_path.mkdir(parents=True, exist_ok=True)

    now = datetime.now(timezone.utc)
    record = {
        "run_id": state.get("run_id", str(uuid.uuid4())),
        "topic": state.get("topic", ""),
        "track": state.get("track", ""),
        "page_type": state.get("page_type", "core-concept"),
        "depth_emphasis": state.get("depth_emphasis", ["applied"]),
        "mode": state.get("mode", "full"),
        "model_writer": os.getenv("WRITER_MODEL", "anthropic/claude-opus-4.7"),
        "model_reviewer": os.getenv("REVIEWER_MODEL", "google/gemini-3.1-pro-preview"),
        "started_at": state.get("started_at", ""),
        "finished_at": now.isoformat(),
        "status": "approved" if state.get("approved") else (
            "flagged" if state.get("review_confidence", 1.0) < float(
                os.getenv("GIT_COMMIT_THRESHOLD", "0.8")
            ) else "error"
        ),
        "confidence": state.get("review_confidence", 0.0),
        "revision_count": state.get("revision_count", 0),
        "output_path": state.get("output_path", ""),
        "has_mvb": state.get("mvb_decision", False),
        "committed": state.get("committed", False),
        "error": state.get("error", ""),
        "review_issues": state.get("review_issues", []),
        "review_feedback": state.get("review_feedback", "")[:500],  # first 500 chars for debug
    }

    # Append to JSONL (machine log, never rewritten)
    jsonl_path = runs_path / "runs.jsonl"
    with jsonl_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    # Sync live docs/system/ pages
    docs_system = Path(__file__).parent.parent.parent.parent / "docs" / "system"
    docs_system.mkdir(parents=True, exist_ok=True)

    _rebuild_status_page(runs_path, docs_system)
    _append_changelog_entry(record, now, docs_system)

    # Close the feedback loop — update metrics.json + observer.md after every run
    try:
        from .observer import observe, write_metrics_json, write_observer_page
        docs_dir = str(Path(__file__).parent.parent.parent.parent / "docs")
        obs = observe(docs_dir=docs_dir, runs_dir=str(runs_path))
        write_metrics_json(obs, runs_path)
        write_observer_page(obs, docs_dir)
    except Exception:
        pass  # observer update must never break the pipeline


def _load_all_runs(runs_path: Path) -> list[dict]:
    jsonl_path = runs_path / "runs.jsonl"
    if not jsonl_path.exists():
        return []
    records = []
    with jsonl_path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def _rebuild_status_page(runs_path: Path, docs_system: Path) -> None:
    """Write docs/system/status.md — live coverage table shown on the wiki."""
    all_runs = _load_all_runs(runs_path)
    if not all_runs:
        return

    # Keep latest run per topic
    latest: dict[str, dict] = {}
    for r in all_runs:
        topic = r.get("topic", "")
        if topic:
            latest[topic] = r

    rows = sorted(latest.values(), key=lambda r: r.get("track", "") + r.get("topic", ""))
    total = len(rows)
    approved = sum(1 for r in rows if r.get("status") == "approved")
    with_mvb = sum(1 for r in rows if r.get("has_mvb"))
    avg_conf = sum(r.get("confidence", 0) for r in rows) / total if total else 0
    total_runs = len(all_runs)

    lines = [
        "---",
        "title: Generation Status",
        "description: Live coverage and quality metrics for every wiki page",
        "---",
        "",
        "# Generation Status",
        "",
        f"> Last updated: **{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}** "
        f"· {total_runs} total runs",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Pages generated | **{total}** |",
        f"| Approved (conf ≥ 0.8) | **{approved}** ({100*approved//total if total else 0}%) |",
        f"| With MVB | **{with_mvb}** |",
        f"| Avg reviewer confidence | **{avg_conf:.2f}** |",
        f"| Committed to git | **{sum(1 for r in rows if r.get('committed'))}** |",
        "",
        "## Pages",
        "",
        "| Track | Topic | Type | Status | Conf | MVB | Rev | Committed |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        status_icon = "✅" if r.get("status") == "approved" else "⚠️" if r.get("status") == "flagged" else "❌"
        mvb_icon = "✓" if r.get("has_mvb") else "—"
        committed_icon = "✓" if r.get("committed") else "—"
        conf = f"{r.get('confidence', 0):.2f}"
        lines.append(
            f"| {r.get('track','')} | {r.get('topic','')} "
            f"| {r.get('page_type','')} | {status_icon} {r.get('status','')} "
            f"| {conf} | {mvb_icon} | {r.get('revision_count',0)} | {committed_icon} |"
        )

    lines += [
        "",
        "---",
        "",
        "*This page is generated automatically after every agent run. "
        "Source: `agents/runs/runs.jsonl`.*",
    ]

    (docs_system / "status.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    # Also keep the legacy runs/ copy
    (runs_path / "wiki_status.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _append_changelog_entry(record: dict, now: datetime, docs_system: Path) -> None:
    """Append one run entry to docs/system/changelog.md."""
    changelog_path = docs_system / "changelog.md"

    topic = record.get("topic", "")
    track = record.get("track", "")
    status = record.get("status", "")
    confidence = record.get("confidence", 0.0)
    has_mvb = record.get("has_mvb", False)
    committed = record.get("committed", False)
    revision_count = record.get("revision_count", 0)
    model_writer = record.get("model_writer", "")
    page_type = record.get("page_type", "")

    status_icon = "✅" if status == "approved" else "⚠️" if status == "flagged" else "❌"
    commit_note = " · git committed" if committed else ""
    mvb_note = " · has MVB" if has_mvb else ""

    entry_lines = [
        f"### {now.strftime('%Y-%m-%d %H:%M UTC')} — {topic}",
        "",
        f"{status_icon} **{status}** · conf={confidence:.2f}{commit_note}{mvb_note}",
        "",
        f"- **Track:** {track}",
        f"- **Page type:** {page_type}",
        f"- **Revisions:** {revision_count}",
        f"- **Writer model:** {model_writer}",
        "",
        "---",
        "",
    ]
    entry = "\n".join(entry_lines)

    if changelog_path.exists():
        existing = changelog_path.read_text(encoding="utf-8")
        # Insert after frontmatter / header, before previous entries
        if "## Entries" in existing:
            insert_at = existing.index("## Entries") + len("## Entries\n\n")
            updated = existing[:insert_at] + entry + existing[insert_at:]
        else:
            updated = existing + "\n" + entry
        changelog_path.write_text(updated, encoding="utf-8")
    else:
        header = "\n".join([
            "---",
            "title: Agent Changelog",
            "description: Every page generated or improved by the editorial agent",
            "---",
            "",
            "# Agent Changelog",
            "",
            "> Every wiki page generated, improved, or reviewed by the agent system is logged here.",
            "> Entries are prepended (newest first). Source: `agents/runs/runs.jsonl`.",
            "",
            "## Entries",
            "",
        ])
        changelog_path.write_text(header + entry, encoding="utf-8")


def load_persona(track: str, personas_dir: str) -> dict:
    """Load per-track editor persona from YAML file."""
    try:
        import yaml
    except ImportError:
        raise ImportError("pyyaml not installed. Run: uv sync")

    persona_path = Path(personas_dir) / f"{track}.yaml"
    if persona_path.exists():
        return yaml.safe_load(persona_path.read_text(encoding="utf-8"))

    # Fallback: generic persona
    return {
        "track": track,
        "domain": track.replace("-", " ").title(),
        "expertise": "machine learning, deep learning",
        "depth_focus": "mathematical foundations + research-level understanding",
        "seminal_authors": [],
        "key_venues": ["NeurIPS", "ICML", "ICLR"],
        "search_seeds": [f"{track} deep learning arxiv"],
    }


def _extract_domain(url: str) -> str:
    from urllib.parse import urlparse
    return urlparse(url).netloc
