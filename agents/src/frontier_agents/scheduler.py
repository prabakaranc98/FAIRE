"""Scheduler jobs for the self-improving wiki harness.

Jobs registered here run on a 48-hour cycle via APScheduler.
Each job is a pure function — no global state, no side effects beyond
writing to the filesystem and the runs/ log.

Cycle:
  1. audit_job    — scan docs/ for quality issues
  2. sprint_job   — parse sprints/current.md, run pipeline per item
  3. changelog_job — write quality delta summary to runs/changelog.md

Run order: audit → sprint → changelog (enforced by the server).
"""

from __future__ import annotations

import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path


AGENTS_DIR = Path(__file__).parent.parent.parent  # agents/
DOCS_DIR = AGENTS_DIR.parent / "docs"
RUNS_DIR = AGENTS_DIR / "runs"
SPRINTS_DIR = AGENTS_DIR / "sprints"
CHANGELOG_PATH = RUNS_DIR / "changelog.md"


# ── Sprint backlog parsing ─────────────────────────────────────────────────────

def parse_sprint_backlog(sprint_path: Path | None = None) -> list[dict]:
    """Parse unchecked items from sprints/current.md.

    Returns list of task dicts with keys:
      type: "generate" | "improve" | "mvb-only"
      topic, track, page_type, depth_emphasis, arc_context (for generate)
    """
    path = sprint_path or SPRINTS_DIR / "current.md"
    if not path.exists():
        return []

    tasks = []
    current_section = "generate"

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()

        if line.startswith("## New pages"):
            current_section = "generate"
        elif line.startswith("## Improvements"):
            current_section = "improve"
        elif line.startswith("## MVB-only"):
            current_section = "mvb-only"
        elif line.startswith("- [ ]"):
            item = line[5:].strip()
            task = _parse_sprint_item(item, current_section)
            if task:
                tasks.append(task)

    return tasks


def _parse_sprint_item(item: str, section: str) -> dict | None:
    """Parse a sprint line into a task dict.

    Format:  topic | track | page_type | depth_emphasis [| arc:id pos:N prev:x next:y]
    Example: score-matching | 02-generative-modeling | core-concept | theoretical | arc:generative-stack pos:4 prev:ddpm next:flow-matching
    """
    parts = [p.strip() for p in item.split("|")]
    if len(parts) < 2:
        return None

    topic = parts[0]
    track = parts[1] if len(parts) > 1 else ""
    page_type = parts[2] if len(parts) > 2 else "core-concept"
    depth_str = parts[3] if len(parts) > 3 else "applied"
    depth_emphasis = [d.strip() for d in depth_str.split(",")]

    arc_context: dict = {}
    if len(parts) > 4:
        arc_str = parts[4]
        arc_match = re.search(r"arc:(\S+)", arc_str)
        pos_match = re.search(r"pos:(\d+)", arc_str)
        prev_match = re.search(r"prev:(\S+)", arc_str)
        next_match = re.search(r"next:(\S+)", arc_str)
        if arc_match:
            arc_context = {
                "arc_id": arc_match.group(1),
                "position": int(pos_match.group(1)) if pos_match else 0,
                "prev": prev_match.group(1) if prev_match else "",
                "next": next_match.group(1) if next_match else "",
            }

    return {
        "type": section,
        "topic": topic,
        "track": track,
        "page_type": page_type,
        "depth_emphasis": depth_emphasis,
        "arc_context": arc_context,
    }


def mark_sprint_item_done(topic: str, sprint_path: Path | None = None) -> None:
    """Mark a sprint item as completed (- [ ] → - [x])."""
    path = sprint_path or SPRINTS_DIR / "current.md"
    if not path.exists():
        return
    content = path.read_text(encoding="utf-8")
    updated = re.sub(
        rf"(- \[ \] {re.escape(topic)}[^\n]*)",
        lambda m: m.group(0).replace("- [ ]", "- [x]"),
        content,
    )
    path.write_text(updated, encoding="utf-8")


def archive_sprint(sprint_path: Path | None = None) -> None:
    """Archive current sprint if all items are checked off."""
    path = sprint_path or SPRINTS_DIR / "current.md"
    if not path.exists():
        return
    content = path.read_text(encoding="utf-8")
    if "- [ ]" in content:
        return  # unchecked items remain

    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    history_dir = SPRINTS_DIR / "history"
    history_dir.mkdir(exist_ok=True)
    shutil.copy(path, history_dir / f"sprint-{date_str}.md")


# ── Changelog ─────────────────────────────────────────────────────────────────

def _read_confidence_before(topics: list[str]) -> dict[str, float]:
    """Read last recorded confidence for each topic from runs.jsonl."""
    jsonl = RUNS_DIR / "runs.jsonl"
    if not jsonl.exists():
        return {}
    latest: dict[str, float] = {}
    with jsonl.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                t = r.get("topic", "")
                if t in topics:
                    latest[t] = r.get("confidence", 0.0)
    return latest


def write_changelog_entry(
    sprint_tasks: list[dict],
    confidence_before: dict[str, float],
    run_results: list[dict],
) -> None:
    """Append a sprint summary to runs/changelog.md."""
    RUNS_DIR.mkdir(exist_ok=True)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines = [
        f"## Sprint — {now}",
        "",
        f"Tasks queued: {len(sprint_tasks)} | Completed: {len(run_results)}",
        "",
        "| Topic | Track | Status | Conf before | Conf after | Delta |",
        "|---|---|---|---|---|---|",
    ]

    for r in run_results:
        topic = r.get("topic", "")
        conf_after = r.get("confidence", 0.0)
        conf_before = confidence_before.get(topic, 0.0)
        delta = conf_after - conf_before
        delta_str = f"+{delta:.2f}" if delta >= 0 else f"{delta:.2f}"
        status = "✅" if r.get("approved") else "⚠️"
        lines.append(
            f"| {topic} | {r.get('track','')} | {status} {r.get('status','')} "
            f"| {conf_before:.2f} | {conf_after:.2f} | {delta_str} |"
        )

    lines += ["", "---", ""]
    entry = "\n".join(lines)

    if CHANGELOG_PATH.exists():
        existing = CHANGELOG_PATH.read_text(encoding="utf-8")
        CHANGELOG_PATH.write_text(entry + existing, encoding="utf-8")
    else:
        header = "# Frontier Wiki — Sprint Changelog\n\n"
        CHANGELOG_PATH.write_text(header + entry, encoding="utf-8")


# ── Jobs (called by APScheduler) ──────────────────────────────────────────────

def audit_job(docs_dir: str = str(DOCS_DIR)) -> dict:
    """Job 1: scan wiki for quality issues. Returns audit summary dict."""
    from .audit import audit_wiki
    report = audit_wiki(docs_dir)
    summary = {
        "generated_at": report.generated_at,
        "pages_scanned": report.pages_scanned,
        "critical": len(report.critical),
        "warnings": len(report.warnings),
        "issues": [
            {"severity": i.severity, "check": i.check, "page": i.page, "message": i.message}
            for i in report.issues
        ],
    }

    # Write audit report to runs/
    RUNS_DIR.mkdir(exist_ok=True)
    audit_path = RUNS_DIR / "last_audit.json"
    import json
    audit_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def sprint_job(dry_run: bool = False) -> list[dict]:
    """Job 2: run unchecked sprint items through the editorial pipeline."""
    from .graph import compile_wiki_graph, compile_mvb_graph

    tasks = parse_sprint_backlog()
    if not tasks:
        return []

    results = []
    wiki_graph = compile_wiki_graph()
    mvb_graph = compile_mvb_graph()

    for task in tasks:
        topic = task["topic"]
        mode = task["type"] if task["type"] == "mvb-only" else "full"

        initial_state = {
            "topic": topic,
            "track": task["track"],
            "depth": "all",
            "mode": mode,
            "page_type": task["page_type"],
            "depth_emphasis": task["depth_emphasis"],
            "arc_context": task["arc_context"],
        }

        try:
            graph = mvb_graph if mode == "mvb-only" else wiki_graph
            result = graph.invoke(initial_state)
            results.append({
                "topic": topic,
                "track": task["track"],
                "approved": result.get("approved", False),
                "status": "approved" if result.get("approved") else "flagged",
                "confidence": result.get("review_confidence", 0.0),
            })
            if not dry_run:
                mark_sprint_item_done(topic)
        except Exception as exc:
            results.append({
                "topic": topic,
                "track": task["track"],
                "approved": False,
                "status": "error",
                "confidence": 0.0,
                "error": str(exc),
            })

    if not dry_run:
        archive_sprint()

    return results


def full_cycle_job(dry_run: bool = False) -> dict:
    """Run the full 48-hour cycle: audit → sprint → changelog."""
    sprint_tasks = parse_sprint_backlog()
    topics = [t["topic"] for t in sprint_tasks]
    confidence_before = _read_confidence_before(topics)

    audit_summary = audit_job()
    run_results = sprint_job(dry_run=dry_run)
    write_changelog_entry(sprint_tasks, confidence_before, run_results)

    return {
        "audit": audit_summary,
        "runs": run_results,
        "changelog_updated": True,
    }
