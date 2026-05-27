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
import random
import re
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

_sprint_lock = threading.Lock()


AGENTS_DIR = Path(__file__).parent.parent.parent  # agents/
DOCS_DIR = AGENTS_DIR.parent / "docs"
RUNS_DIR = AGENTS_DIR / "runs"
SPRINTS_DIR = AGENTS_DIR / "sprints"
CHANGELOG_PATH = RUNS_DIR / "changelog.md"


# ── Sprint backlog parsing ─────────────────────────────────────────────────────

def parse_sprint_backlog(sprint_path: Path | None = None) -> list[dict]:
    """Parse unchecked items from sprints/current.md.

    Returns list of task dicts with keys:
      type: "generate" | "improve" | "mvb-only" | "arc-step" | "arc-index"
      topic, track, page_type, depth_emphasis, arc_context, mode
    """
    path = sprint_path or SPRINTS_DIR / "current.md"
    if not path.exists():
        return []

    tasks = []
    current_section = "generate"

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()

        # Section headers
        if line.startswith("## New pages") or line.startswith("## New Content") or line.startswith("## Curriculum"):
            current_section = "generate"
        elif line.startswith("## MVB-only"):
            current_section = "mvb-only"
        elif line.startswith("## Arc Steps") or line.startswith("## Arc Step"):
            current_section = "arc-step"
        elif line.startswith("## Arc Index") or line.startswith("## Arc Indexes"):
            current_section = "arc-index"
        elif line.startswith("## Critical") or line.startswith("## High") or line.startswith("## Improvements"):
            current_section = "improve"
        elif line.startswith("- [ ]"):
            item = line[5:].strip()
            task = _parse_sprint_item(item, current_section)
            if task:
                tasks.append(task)

    return tasks


def _parse_sprint_item(item: str, section: str) -> dict | None:
    """Parse a sprint line into a task dict.

    Curriculum format:
      topic | track | page_type | depth_emphasis
      Example: diffusion-models | 02-generative-modeling | core-concept | theoretical frontier

    Arc step format:
      topic | track | arc-step | depth_emphasis | arc:id pos:N ch:K ch_title:"Chapter Title" prev:slug next:slug prev_artifact:"artifact desc" artifact:"this artifact desc" total:M
      Example: ddpm | 02-generative-modeling | arc-step | applied | arc:generative-stack pos:4 ch:2 ch_title:"Score-Based Denoising" prev:score-matching next:ddim prev_artifact:"score network on 2D data" artifact:"working DDPM checkpoint" total:8

    Arc index format:
      topic | track | arc-index | frontier | arc:id dest:"destination phrase" total:N
      Example: generative-stack-index | 02-generative-modeling | arc-index | frontier | arc:generative-stack dest:"conditional latent diffusion from scratch" total:8
    """
    item = re.sub(r"<!--.*?-->", "", item).strip()

    parts = [p.strip() for p in item.split("|")]
    if len(parts) < 2:
        return None

    topic = parts[0]
    track = parts[1] if len(parts) > 1 else ""
    page_type = parts[2] if len(parts) > 2 else "core-concept"
    depth_str = parts[3] if len(parts) > 3 else "applied"
    depth_emphasis = [d.strip() for d in re.split(r"[,\s]+", depth_str) if d.strip()]
    if not depth_emphasis:
        depth_emphasis = ["applied"]

    # Determine mode from page_type or section
    if page_type == "arc-step" or section == "arc-step":
        mode = "arc-step"
        page_type = "arc-step"
    elif page_type == "arc-index" or section == "arc-index":
        mode = "arc-index"
        page_type = "arc-index"
    elif section == "mvb-only":
        mode = "mvb-only"
    else:
        mode = "full"

    arc_context: dict = {}
    if len(parts) > 4:
        arc_str = parts[4]

        def _extract(pattern: str, default="") -> str:
            m = re.search(pattern, arc_str)
            return m.group(1) if m else default

        def _extract_quoted(key: str, default="") -> str:
            # (?<!\w) prevents "artifact" from matching inside "prev_artifact"
            m = re.search(rf'(?<!\w){re.escape(key)}:"([^"]*)"', arc_str)
            if m:
                return m.group(1)
            m = re.search(rf'(?<!\w){re.escape(key)}:(\S+)', arc_str)
            return m.group(1) if m else default

        arc_id = _extract(r"arc:(\S+)")
        if arc_id:
            arc_context = {
                "arc_id": arc_id,
                "arc_title": arc_id.replace("-", " ").title(),
                "destination": _extract_quoted("dest"),
                "position": int(_extract(r"pos:(\d+)", "0")),
                "total": int(_extract(r"total:(\d+)", "0")),
                "chapter": int(_extract(r"ch:(\d+)", "0")),
                "chapter_title": _extract_quoted("ch_title"),
                "prev": _extract(r"prev:(\S+)"),
                "next": _extract(r"next:(\S+)"),
                "prev_artifact": _extract_quoted("prev_artifact"),
                "compounding_artifact": _extract_quoted("artifact"),
            }

    return {
        "type": section,
        "mode": mode,
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
    from .tools import display_ts
    now = display_ts()

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
    """Job 2: run unchecked sprint items through the editorial pipeline.

    Checks budget before running. If budget is "paused", skips generation entirely.
    If "reduced", overrides WRITER_MODEL to FALLBACK_MODEL (cheaper).
    """
    from .graph import compile_wiki_graph, compile_mvb_graph
    import os as _os

    tasks = parse_sprint_backlog()
    if not tasks:
        return []

    # Budget gate — check before spending any tokens
    budget_mode = "full"
    try:
        from .observer import check_budget
        b = check_budget()
        budget_mode = b.mode
    except Exception:
        pass  # if budget check fails, proceed in full mode

    if budget_mode == "paused":
        print("[sprint] Budget exhausted — skipping all generation. Replenish OpenRouter credits.")
        return [{"topic": t["topic"], "track": t["track"], "approved": False,
                 "status": "skipped", "confidence": 0.0, "error": "budget_paused"}
                for t in tasks]

    # Reduced mode: override writer model to cheaper fallback
    _original_writer = None
    if budget_mode == "reduced":
        fallback = _os.getenv("FALLBACK_MODEL", "anthropic/claude-sonnet-4.6")
        _original_writer = _os.environ.get("WRITER_MODEL")
        _os.environ["WRITER_MODEL"] = fallback
        print(f"[sprint] Reduced budget mode — using {fallback} for all writes")

    wiki_graph = compile_wiki_graph()
    mvb_graph = compile_mvb_graph()

    def _run_task(task):
        time.sleep(random.uniform(0, 3))  # jitter to spread API calls
        topic = task["topic"]
        mode = task.get("mode", "full")
        arc_ctx = task.get("arc_context", {})
        initial_state = {
            "topic": topic,
            "track": task["track"],
            "depth": "all",
            "mode": mode,
            "page_type": task["page_type"],
            "depth_emphasis": task["depth_emphasis"],
            "arc_context": arc_ctx,
            # chapter / compounding fields (arc-step only)
            "chapter": arc_ctx.get("chapter", 0),
            "chapter_title": arc_ctx.get("chapter_title", ""),
            "compounding_artifact": arc_ctx.get("compounding_artifact", ""),
            "prev_artifact": arc_ctx.get("prev_artifact", ""),
        }
        try:
            graph = mvb_graph if mode == "mvb-only" else wiki_graph
            result = graph.invoke(initial_state)
            r = {
                "topic": topic,
                "track": task["track"],
                "approved": result.get("approved", False),
                "status": "approved" if result.get("approved") else "flagged",
                "confidence": result.get("review_confidence", 0.0),
            }
            if not dry_run:
                with _sprint_lock:
                    mark_sprint_item_done(topic)
            return r
        except Exception as exc:
            return {
                "topic": topic,
                "track": task["track"],
                "approved": False,
                "status": "error",
                "confidence": 0.0,
                "error": str(exc),
            }

    max_workers = int(_os.getenv("SPRINT_WORKERS", "4"))
    results = []
    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="sprint") as pool:
        futures = {pool.submit(_run_task, t): t for t in tasks}
        for fut in as_completed(futures):
            try:
                results.append(fut.result())
            except Exception as exc:
                t = futures[fut]
                results.append({
                    "topic": t["topic"],
                    "track": t["track"],
                    "approved": False,
                    "status": "error",
                    "confidence": 0.0,
                    "error": str(exc),
                })

    if not dry_run:
        with _sprint_lock:
            archive_sprint()

    # Restore original writer model if we overrode it
    if _original_writer is not None:
        _os.environ["WRITER_MODEL"] = _original_writer
    elif budget_mode == "reduced" and "WRITER_MODEL" in _os.environ:
        del _os.environ["WRITER_MODEL"]

    return results


def full_cycle_job(dry_run: bool = False) -> dict:
    """Run the full 48-hour cycle: supervise → audit → sprint → changelog.

    Cycle order:
      1. Supervisor — assess wiki health, update sprint with priorities
      2. Audit      — structural quality check on all pages
      3. Sprint     — execute queued work items
      4. Changelog  — log quality delta per touched page
    """
    # Step 0: Supervisor updates the sprint with latest priorities
    supervisor_summary = {}
    try:
        from .supervisor import run_supervisor
        report = run_supervisor(
            docs_dir=str(DOCS_DIR),
            runs_dir="runs",
            sprints_dir=str(SPRINTS_DIR),
            dry_run=dry_run,
            verbose=False,
        )
        supervisor_summary = {
            "total_pages": report.health.total_pages,
            "approved": report.health.approved_pages,
            "flagged": report.health.flagged_pages,
            "avg_confidence": round(report.health.avg_confidence, 2),
            "actions_queued": len(report.actions),
        }
    except Exception as e:
        supervisor_summary = {"error": str(e)}

    sprint_tasks = parse_sprint_backlog()
    topics = [t["topic"] for t in sprint_tasks]
    confidence_before = _read_confidence_before(topics)

    audit_summary = audit_job()
    run_results = sprint_job(dry_run=dry_run)
    write_changelog_entry(sprint_tasks, confidence_before, run_results)

    # Step 4.5: Rebuild track index pages from filesystem state. Pages get
    # auto-listed under their track's index.md so visitors actually see them.
    # Without this, track indexes stay as stubs and 71+ generated pages look
    # invisible to anyone navigating /curriculum/core/<track>/.
    if not dry_run:
        try:
            import subprocess as _sp
            from pathlib import Path as _P
            repo_root = _P(__file__).resolve().parent.parent.parent.parent
            _sp.run(["python3", "scripts/rebuild_track_indexes.py"],
                    cwd=repo_root, capture_output=True, text=True, timeout=30)
        except Exception:
            pass  # never break the cycle on cosmetic regeneration

    # Step 4.6: Queue primer-quality improve passes. Pages with primer-quality
    # score < 0.85 (or unscored, which means generated before critic-primer-quality
    # existed) get appended to current.md as `## Primer Improvements`. Capped at
    # 10 per cycle to avoid flooding the queue. Over a few cycles the entire
    # corpus is lifted to primer grade.
    if not dry_run:
        try:
            import subprocess as _sp
            from pathlib import Path as _P
            repo_root = _P(__file__).resolve().parent.parent.parent.parent
            _sp.run(["python3", "scripts/queue_primer_improvements.py"],
                    cwd=repo_root, capture_output=True, text=True, timeout=30)
        except Exception:
            pass  # cosmetic; never block the cycle

    # Step 4.7: Refresh the arc catalog so newly-landed arc-index pages
    # flip from yellow ("designed · next") to green ("live") on the
    # public catalog at /curriculum/arcs/. Auto-rebuilt from
    # docs/system/arc-roadmap.md cross-referenced against on-disk presence.
    if not dry_run:
        try:
            import subprocess as _sp
            from pathlib import Path as _P
            repo_root = _P(__file__).resolve().parent.parent.parent.parent
            _sp.run(["python3", "scripts/build_arc_catalog.py"],
                    cwd=repo_root, capture_output=True, text=True, timeout=30)
        except Exception:
            pass  # cosmetic; never block the cycle

    # Step 5: Retrospective — backlog agent closes the loop. Aggregates patterns
    # from this cycle's runs, runs an LLM scrum-style retrospective, auto-applies
    # safe proposals (stub seeds), writes everything to docs/system/backlog.md
    # for the next cycle to consume.
    retro_summary: dict = {}
    if not dry_run:
        try:
            from .retrospective import retrospective_job
            retro_result = retrospective_job()
            retro_summary = {
                "n_proposals": sum(
                    len(retro_result.get("retro", {}).get(k, []))
                    for k in ("went_well", "went_wrong", "needs_depth", "new_to_add", "process_improvements")
                ),
                "n_new_to_add": len(retro_result.get("retro", {}).get("new_to_add", [])),
                "n_auto_applied": sum(
                    1 for a in retro_result.get("applied", [])
                    if isinstance(a, dict) and a.get("status", "").startswith("applied")
                ),
                "backlog_path": retro_result.get("backlog_path", ""),
                "error": retro_result.get("error"),
            }
        except Exception as e:
            retro_summary = {"error": str(e)}

    # Step 6: Batched push — one push per cycle, not per page. GitHub Pages
    # soft-caps Pages builds at 10/hour; pushing per page (20+/cycle) silently
    # blew past it and stopped publishing for ~28 hours. Now: pages still
    # commit individually via GIT_AUTO_COMMIT (preserves git history), but
    # the network push is a single batched call at cycle close. This means
    # gh-pages gets one new HEAD per cycle → exactly one Pages build per
    # cycle → comfortably under the limit at 1-hour intervals.
    push_summary: dict = {}
    if not dry_run and os.getenv("GIT_BATCHED_PUSH", "true").lower() == "true":
        try:
            import subprocess as _sp
            from pathlib import Path as _P
            repo_root = _P(__file__).resolve().parent.parent.parent.parent
            ahead = _sp.run(
                ["git", "rev-list", "--count", "origin/main..HEAD"],
                cwd=repo_root, capture_output=True, text=True, timeout=15,
            )
            n_ahead = int((ahead.stdout or "0").strip() or "0")
            if n_ahead > 0:
                push = _sp.run(
                    ["git", "push", "origin", "main"],
                    cwd=repo_root, capture_output=True, text=True, timeout=60,
                )
                push_summary = {
                    "n_commits_pushed": n_ahead,
                    "ok": push.returncode == 0,
                    "stderr_tail": (push.stderr or "")[-200:] if push.returncode else "",
                }
            else:
                push_summary = {"n_commits_pushed": 0, "ok": True, "note": "no commits ahead of origin"}
        except Exception as e:
            push_summary = {"ok": False, "error": str(e)}

    return {
        "supervisor": supervisor_summary,
        "audit": audit_summary,
        "runs": run_results,
        "changelog_updated": True,
        "retrospective": retro_summary,
        "batched_push": push_summary,
    }
