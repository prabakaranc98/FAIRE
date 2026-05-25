#!/usr/bin/env python
"""Frontier Wiki — self-improving agent server.

Runs a background scheduler that executes the full wiki improvement cycle
every 48 hours. Also exposes a FastAPI HTTP interface for monitoring,
manual triggers, and status inspection.

Usage:
  cd agents
  uv run python server.py              # start (48h cycle, default)
  uv run python server.py --interval 1 # run every 1 hour (for testing)
  uv run python server.py --dry-run    # run cycle but don't write to disk

API endpoints (http://localhost:8765):
  GET  /status         — current sprint + last audit + changelog summary
  GET  /audit          — run audit now, return issues
  POST /sprint         — run sprint now (background)
  GET  /runs           — last N run records from runs.jsonl
  GET  /changelog      — last N changelog entries
  POST /trigger        — run one full cycle immediately

The server writes nothing to disk until the first scheduled or triggered run.
"""

from __future__ import annotations

import json
import sys
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
    AGENTS_DIR,
    CHANGELOG_PATH,
    RUNS_DIR,
    SPRINTS_DIR,
    audit_job,
    full_cycle_job,
    parse_sprint_backlog,
    sprint_job,
)

app = FastAPI(
    title="Frontier Wiki Agent Server",
    description="Self-improving wiki harness — audit, sprint, changelog",
    version="0.1.0",
)

_scheduler: BackgroundScheduler | None = None
_dry_run: bool = False
_last_cycle_result: dict = {}


# ── HTTP routes ────────────────────────────────────────────────────────────────

@app.get("/status")
def status():
    """Current sprint queue, last audit summary, and server state."""
    tasks = parse_sprint_backlog()
    audit_path = RUNS_DIR / "last_audit.json"
    last_audit = json.loads(audit_path.read_text()) if audit_path.exists() else {}

    changelog_snippet = ""
    if CHANGELOG_PATH.exists():
        lines = CHANGELOG_PATH.read_text(encoding="utf-8").splitlines()
        changelog_snippet = "\n".join(lines[:30])

    return {
        "server": "running",
        "dry_run": _dry_run,
        "sprint_queue": tasks,
        "last_audit": last_audit,
        "last_cycle": _last_cycle_result,
        "changelog_preview": changelog_snippet,
        "scheduler_running": _scheduler.running if _scheduler else False,
        "next_run": (
            str(_scheduler.get_jobs()[0].next_run_time)
            if _scheduler and _scheduler.get_jobs()
            else "not scheduled"
        ),
    }


@app.get("/audit")
def run_audit():
    """Run a wiki quality audit now and return the issues."""
    return audit_job()


@app.post("/sprint")
def run_sprint(background_tasks: BackgroundTasks):
    """Start a sprint run in the background. Returns immediately."""
    background_tasks.add_task(_run_sprint_background)
    tasks = parse_sprint_backlog()
    return {"queued": True, "tasks": len(tasks)}


def _run_sprint_background():
    global _last_cycle_result
    _last_cycle_result = full_cycle_job(dry_run=_dry_run)


@app.post("/trigger")
def trigger_cycle():
    """Run one full cycle (audit → sprint → changelog) synchronously. Blocks until done."""
    global _last_cycle_result
    _last_cycle_result = full_cycle_job(dry_run=_dry_run)
    return _last_cycle_result


@app.get("/runs")
def get_runs(limit: int = 20):
    """Return the last N run records from runs.jsonl."""
    jsonl = RUNS_DIR / "runs.jsonl"
    if not jsonl.exists():
        return []
    records = []
    with jsonl.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records[-limit:]


@app.get("/changelog", response_class=PlainTextResponse)
def get_changelog():
    """Return the full changelog as plain text."""
    if not CHANGELOG_PATH.exists():
        return "No changelog yet — run a cycle first."
    return CHANGELOG_PATH.read_text(encoding="utf-8")


@app.get("/sprint/queue")
def sprint_queue():
    """Return the current sprint backlog (unchecked items only)."""
    return parse_sprint_backlog()


@app.post("/supervise")
def supervise(background_tasks: BackgroundTasks, dry_run: bool = False):
    """Run the supervising agent: assess health, prioritise work, update sprint.

    Returns immediately; runs in background. Check /supervisor for the report.
    """
    background_tasks.add_task(_run_supervisor_background, dry_run)
    return {"queued": True, "message": "Supervisor running in background — check /supervisor when done"}


@app.get("/supervisor", response_class=PlainTextResponse)
def get_supervisor_report():
    """Return the latest supervisor report (docs/system/supervisor.md)."""
    docs_dir = Path(__file__).parent.parent / "docs"
    report_path = docs_dir / "system" / "supervisor.md"
    if not report_path.exists():
        return "No supervisor report yet — POST /supervise to generate one."
    return report_path.read_text(encoding="utf-8")


def _run_supervisor_background(dry_run: bool = False):
    """Run supervisor in background thread."""
    try:
        from frontier_agents.supervisor import run_supervisor
        docs_dir = str(Path(__file__).parent.parent / "docs")
        run_supervisor(docs_dir=docs_dir, dry_run=dry_run)
    except Exception as e:
        print(f"[supervisor] Error: {e}")


# ── CLI ───────────────────────────────────────────────────────────────────────

@click.command()
@click.option("--host", default="127.0.0.1", help="Server host")
@click.option("--port", default=8765, type=int, help="Server port")
@click.option(
    "--interval",
    default=48,
    type=int,
    help="Cycle interval in hours (default: 48)",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Run cycles but don't write pages or commit to git",
)
@click.option(
    "--run-now",
    is_flag=True,
    help="Run one full cycle immediately on startup, then schedule",
)
def serve(host, port, interval, dry_run, run_now):
    """Start the self-improving wiki agent server."""
    global _scheduler, _dry_run
    _dry_run = dry_run

    scheduler = BackgroundScheduler()
    scheduler.add_job(
        lambda: full_cycle_job(dry_run=dry_run),
        trigger="interval",
        hours=interval,
        id="full_cycle",
        max_instances=1,
        coalesce=True,
    )
    scheduler.start()
    _scheduler = scheduler

    if run_now:
        click.echo("Running initial cycle now...")
        full_cycle_job(dry_run=dry_run)

    click.echo(
        f"Frontier Wiki server starting — "
        f"http://{host}:{port} | "
        f"cycle every {interval}h | "
        f"dry_run={dry_run}"
    )
    click.echo(
        f"API: GET /status | GET /audit | POST /trigger | GET /runs | GET /changelog"
    )

    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    serve()
