#!/usr/bin/env python
"""Frontier Wiki — autonomous agent server.

This is the single entry point. Once started, the server manages the wiki
completely — assessing quality, deciding what to generate, executing, committing.

No manual sprint editing required. The supervisor is the intelligence layer
that determines priorities. The 48h cycle is the cadence.

Usage:
  cd agents && uv run python server.py              # start with 48h cycle
  cd agents && uv run python server.py --run-now    # run one cycle immediately, then schedule
  cd agents && uv run python server.py --interval 4 # 4h cycle (for testing)
  cd agents && uv run python server.py --dry-run    # simulate without writing

HTTP interface (http://localhost:8765):
  GET  /           — plain-text health dashboard
  GET  /status     — JSON: health + sprint + last cycle
  GET  /audit      — run audit now, return issues
  POST /trigger    — run full cycle now (blocks until done)
  POST /generate   — generate one specific page now (JSON body: {topic, track, ...})
  GET  /runs       — last N run records
  GET  /changelog  — full changelog
  GET  /sprint     — current sprint queue
  GET  /supervisor — latest supervisor report

Architecture:
  The server is the system. The CLI (generate.py) is a debugging tool.

  48h cycle: supervisor → audit → sprint → changelog
    1. supervisor — assess wiki health, update sprint with ranked priorities
    2. audit      — structural quality scan (nested lists, banned URLs, missing sections)
    3. sprint     — run the top N items from the supervisor's queue
    4. changelog  — log quality delta per touched page

  On startup: supervisor runs immediately to bootstrap the sprint queue.
  The human never needs to edit sprints/current.md — the supervisor does it.
"""

from __future__ import annotations

import json
import os
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import click
import uvicorn
from apscheduler.schedulers.background import BackgroundScheduler
from dotenv import load_dotenv
from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import PlainTextResponse

load_dotenv(Path(__file__).parent / ".env")

from frontier_agents.scheduler import (
    CHANGELOG_PATH,
    DOCS_DIR,
    RUNS_DIR,
    SPRINTS_DIR,
    audit_job,
    full_cycle_job,
    parse_sprint_backlog,
)
from frontier_agents.observer import check_budget

app = FastAPI(
    title="Frontier Wiki Agent Server",
    description="Autonomous wiki improvement system",
    version="1.0.0",
)

_scheduler: BackgroundScheduler | None = None
_dry_run: bool = False
_last_cycle_result: dict = {}
_server_started_at: str = ""
_cycle_count: int = 0


# ── Health dashboard ──────────────────────────────────────────────────────────

@app.get("/", response_class=PlainTextResponse)
def dashboard():
    """Plain-text health dashboard — reads from metrics.json when available."""
    metrics_path = RUNS_DIR / "metrics.json"
    metrics: dict = {}
    if metrics_path.exists():
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    # Fallback to live JSONL scan if metrics.json not yet populated
    if not metrics:
        jsonl = RUNS_DIR / "runs.jsonl"
        runs: list[dict] = []
        if jsonl.exists():
            with jsonl.open(encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            runs.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
        latest: dict[str, dict] = {}
        for r in runs:
            t = r.get("topic", "")
            if t:
                latest[t] = r
        total = len(latest)
        approved = sum(1 for r in latest.values() if r.get("status") == "approved")
        flagged = sum(1 for r in latest.values() if r.get("status") == "flagged")
        avg_conf = sum(r.get("confidence", 0) for r in latest.values()) / total if total else 0
        budget_str = "unknown"
        coverage_str = "unknown"
        observed_at = "not yet observed"
    else:
        total = metrics.get("generated_pages", 0)
        approved = metrics.get("approved_pages", 0)
        flagged = sum(tm.get("flagged", 0) for tm in metrics.get("track_metrics", {}).values())
        avg_conf = metrics.get("avg_confidence", 0.0)
        b = metrics.get("budget", {})
        remaining = b.get("remaining_usd")
        budget_str = (
            f"${remaining:.2f} remaining ({b.get('mode', '?')} mode)"
            if remaining is not None
            else f"unlimited ({b.get('mode', '?')} mode)"
        )
        coverage_str = f"{metrics.get('coverage_pct', 0):.1%} ({metrics.get('stub_pages', 0)} stubs)"
        observed_at = metrics.get("observed_at", "?")

    sprint = parse_sprint_backlog()
    next_run = (
        str(_scheduler.get_jobs()[0].next_run_time)
        if _scheduler and _scheduler.get_jobs()
        else "not scheduled"
    )
    threshold = os.getenv("GIT_COMMIT_THRESHOLD", "0.8")

    lines = [
        "═══════════════════════════════════════════════",
        "  Frontier Wiki — Autonomous Agent Server",
        "═══════════════════════════════════════════════",
        "",
        f"  Started:      {_server_started_at}",
        f"  Cycles run:   {_cycle_count}",
        f"  Next cycle:   {next_run}",
        f"  Dry run:      {_dry_run}",
        f"  Observed:     {observed_at}",
        "",
        "  Wiki state:",
        f"    Coverage:          {coverage_str}",
        f"    Pages generated:   {total}",
        f"    Approved (≥{threshold}):  {approved} ({int(100*approved/total) if total else 0}%)",
        f"    Flagged:           {flagged}",
        f"    Avg confidence:    {avg_conf:.2f}",
        f"    Budget:            {budget_str}",
        "",
        f"  Sprint queue:  {len(sprint)} items pending",
        "",
        "  Endpoints:",
        "    POST /trigger       — run full cycle now",
        "    GET  /metrics       — observer snapshot (JSON)",
        "    GET  /observer      — observer dashboard (markdown)",
        "    GET  /budget        — current credit state",
        "    GET  /sprint        — view queued work",
        "    GET  /runs          — run history",
        "    GET  /supervisor    — latest supervisor report",
        "    GET  /changelog     — quality changelog",
        "",
        "═══════════════════════════════════════════════",
    ]
    return "\n".join(lines) + "\n"


# ── JSON API routes ───────────────────────────────────────────────────────────

@app.get("/status")
def status():
    """JSON: server state, wiki health, sprint queue, last cycle result."""
    jsonl = RUNS_DIR / "runs.jsonl"
    runs = []
    if jsonl.exists():
        with jsonl.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        runs.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

    latest: dict[str, dict] = {}
    for r in runs:
        t = r.get("topic", "")
        if t:
            latest[t] = r

    total = len(latest)
    approved = sum(1 for r in latest.values() if r.get("status") == "approved")
    avg_conf = (
        sum(r.get("confidence", 0) for r in latest.values()) / total if total else 0
    )

    return {
        "server": {
            "started_at": _server_started_at,
            "cycles_run": _cycle_count,
            "dry_run": _dry_run,
            "next_cycle": (
                str(_scheduler.get_jobs()[0].next_run_time)
                if _scheduler and _scheduler.get_jobs()
                else None
            ),
        },
        "wiki": {
            "pages_generated": total,
            "approved": approved,
            "avg_confidence": round(avg_conf, 3),
            "commit_threshold": float(os.getenv("GIT_COMMIT_THRESHOLD", "0.8")),
        },
        "sprint": parse_sprint_backlog(),
        "last_cycle": _last_cycle_result,
    }


@app.get("/audit")
def run_audit_endpoint():
    """Run a structural audit of all wiki pages. Returns issues found."""
    return audit_job()


@app.post("/trigger")
def trigger_cycle():
    """Run a full cycle immediately (supervisor → audit → sprint → changelog). Blocks."""
    global _last_cycle_result, _cycle_count
    _last_cycle_result = full_cycle_job(dry_run=_dry_run)
    _cycle_count += 1
    return _last_cycle_result


@app.post("/trigger/background")
def trigger_cycle_background(background_tasks: BackgroundTasks):
    """Queue a full cycle in the background. Returns immediately."""
    background_tasks.add_task(_run_cycle_background)
    return {"queued": True, "message": "Cycle started in background — check /status for progress"}


@app.post("/generate")
def generate_page(
    topic: str,
    track: str,
    page_type: str = "core-concept",
    depth_emphasis: str = "applied",
    background_tasks: BackgroundTasks = None,
):
    """Generate or improve one specific page. Runs in background."""
    if background_tasks:
        background_tasks.add_task(
            _generate_page_background, topic, track, page_type, depth_emphasis.split(",")
        )
        return {"queued": True, "topic": topic, "track": track}
    return {"error": "No background task manager available"}


@app.post("/supervise")
def supervise_endpoint(background_tasks: BackgroundTasks, dry_run: bool = False):
    """Run the supervisor: assess health, update sprint queue. Background."""
    background_tasks.add_task(_run_supervisor_background, dry_run or _dry_run)
    return {"queued": True, "message": "Supervisor running — check /supervisor for report"}


@app.get("/supervisor", response_class=PlainTextResponse)
def get_supervisor_report():
    """Return the latest supervisor report."""
    report_path = DOCS_DIR / "system" / "supervisor.md"
    if not report_path.exists():
        return "No supervisor report yet. POST /supervise to generate one, or wait for the next cycle."
    return report_path.read_text(encoding="utf-8")


@app.get("/sprint", response_class=PlainTextResponse)
def get_sprint():
    """Return the current sprint backlog (raw markdown)."""
    sprint_path = SPRINTS_DIR / "current.md"
    if not sprint_path.exists():
        return "No sprint file. The supervisor will create one on the next cycle."
    return sprint_path.read_text(encoding="utf-8")


@app.get("/runs")
def get_runs(limit: int = 20):
    """Return the last N run records."""
    jsonl = RUNS_DIR / "runs.jsonl"
    if not jsonl.exists():
        return []
    records = []
    with jsonl.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records[-limit:]


@app.get("/changelog", response_class=PlainTextResponse)
def get_changelog():
    """Return the sprint changelog."""
    if not CHANGELOG_PATH.exists():
        return "No changelog yet — run a cycle first."
    return CHANGELOG_PATH.read_text(encoding="utf-8")


@app.get("/metrics")
def get_metrics():
    """Return the latest WikiObservation snapshot from metrics.json."""
    metrics_path = RUNS_DIR / "metrics.json"
    if not metrics_path.exists():
        return {"error": "No metrics yet — wait for first supervisor run or POST /supervise"}
    try:
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    except Exception as e:
        return {"error": str(e)}


@app.get("/observer", response_class=PlainTextResponse)
def get_observer():
    """Return the observer control dashboard (markdown)."""
    observer_path = DOCS_DIR / "system" / "observer.md"
    if not observer_path.exists():
        return "No observer report yet — wait for first supervisor run or POST /supervise"
    return observer_path.read_text(encoding="utf-8")


@app.get("/budget")
def get_budget():
    """Return current OpenRouter credit state."""
    try:
        b = check_budget()
        return {
            "usage_usd": b.usage_usd,
            "limit_usd": b.limit_usd,
            "remaining_usd": b.remaining_usd,
            "is_free_tier": b.is_free_tier,
            "mode": b.mode,
            "error": b.error or None,
        }
    except Exception as e:
        return {"error": str(e)}


# ── Background workers ────────────────────────────────────────────────────────

def _run_cycle_background():
    global _last_cycle_result, _cycle_count
    _last_cycle_result = full_cycle_job(dry_run=_dry_run)
    _cycle_count += 1


def _run_supervisor_background(dry_run: bool = False):
    try:
        from frontier_agents.supervisor import run_supervisor
        run_supervisor(
            docs_dir=str(DOCS_DIR),
            runs_dir="runs",
            sprints_dir=str(SPRINTS_DIR),
            dry_run=dry_run,
        )
    except Exception as e:
        print(f"[supervisor] Error: {e}")


def _generate_page_background(topic: str, track: str, page_type: str, depth_emphasis: list[str]):
    try:
        from frontier_agents.graph import compile_wiki_graph
        graph = compile_wiki_graph()
        graph.invoke({
            "topic": topic,
            "track": track,
            "depth": "all",
            "mode": "full",
            "page_type": page_type,
            "depth_emphasis": depth_emphasis,
            "arc_context": {},
        })
    except Exception as e:
        print(f"[generate] Error on {topic}: {e}")


# ── CLI startup ───────────────────────────────────────────────────────────────

@click.command()
@click.option("--host", default="127.0.0.1", help="Bind host (use 0.0.0.0 to expose on network)")
@click.option("--port", default=8765, type=int, help="Port")
@click.option("--interval", default=48, type=int, help="Cycle interval in hours")
@click.option("--dry-run", is_flag=True, help="Simulate — read-only; no disk writes, no git commits")
@click.option("--run-now", is_flag=True, help="Run one full cycle immediately on startup, then schedule")
def serve(host, port, interval, dry_run, run_now):
    """Start the Frontier Wiki autonomous agent server.

    The server manages the wiki indefinitely: assessing quality, deciding what
    to generate, executing the pipeline, committing approved pages.

    No manual sprint editing required — the supervisor handles priorities.

    \b
    Typical usage:
      uv run python server.py              # start with 48h auto-cycle
      uv run python server.py --run-now    # run one cycle now, then schedule
    """
    global _scheduler, _dry_run, _server_started_at
    _dry_run = dry_run
    _server_started_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # Step 1: Bootstrap — run supervisor immediately in background to populate sprint
    # This ensures the sprint is always fresh when the server starts, regardless
    # of whether --run-now is set.
    bootstrap_thread = threading.Thread(
        target=_run_supervisor_background,
        args=(dry_run,),
        daemon=True,
        name="supervisor-bootstrap",
    )
    bootstrap_thread.start()
    click.echo(f"[{_server_started_at}] Supervisor bootstrapping sprint queue...")

    # Step 2: Set up the APScheduler for recurring cycles
    scheduler = BackgroundScheduler()
    scheduler.add_job(
        _run_cycle_background,
        trigger="interval",
        hours=interval,
        id="full_cycle",
        max_instances=1,
        coalesce=True,
    )
    scheduler.start()
    _scheduler = scheduler

    # Step 3: Optionally run a full cycle immediately
    if run_now:
        click.echo("Running initial full cycle now (supervisor → audit → sprint → changelog)...")
        full_cycle_job(dry_run=dry_run)
        click.echo("Initial cycle complete.")

    click.echo(
        f"\nFrontier Wiki agent server running — http://{host}:{port}\n"
        f"  Cycle every {interval}h | dry_run={dry_run}\n"
        f"  GET / for health dashboard | POST /trigger to run a cycle\n"
    )

    uvicorn.run(app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    serve()
