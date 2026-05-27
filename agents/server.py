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

import atexit
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import click
import uvicorn
from apscheduler.schedulers.background import BackgroundScheduler
from dotenv import load_dotenv
from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import HTMLResponse, PlainTextResponse

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

_POOL = ThreadPoolExecutor(max_workers=12, thread_name_prefix="wiki")
atexit.register(_POOL.shutdown, wait=False)

_active: dict[str, str] = {}
_active_lock = threading.Lock()


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


@app.get("/dashboard", response_class=HTMLResponse)
def get_dashboard():
    """Live HTML metrics dashboard — auto-refreshes every 20s."""
    metrics_path = RUNS_DIR / "metrics.json"
    runs_path = RUNS_DIR / "runs.jsonl"

    m: dict = {}
    if metrics_path.exists():
        try:
            m = json.loads(metrics_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    recent_runs: list[dict] = []
    if runs_path.exists():
        try:
            lines = runs_path.read_text(encoding="utf-8").splitlines()
            for line in reversed(lines[-30:]):
                if line.strip():
                    recent_runs.append(json.loads(line))
            recent_runs = list(reversed(recent_runs))
        except Exception:
            pass

    # ── helpers ────────────────────────────────────────────────────────────────
    def pct_bar(pct: float, width: int = 120) -> str:
        fill = int(pct * width)
        color = "#22c55e" if pct >= 0.8 else "#f59e0b" if pct >= 0.4 else "#ef4444"
        return (
            f'<div style="background:#1e293b;border-radius:4px;height:8px;width:{width}px;display:inline-block;vertical-align:middle">'
            f'<div style="background:{color};border-radius:4px;height:8px;width:{fill}px"></div></div>'
        )

    def conf_color(c: float) -> str:
        return "#22c55e" if c >= 0.8 else "#f59e0b" if c >= 0.6 else "#ef4444"

    def status_badge(status: str) -> str:
        colors = {"approved": "#22c55e", "flagged": "#f59e0b", "error": "#ef4444", "skipped": "#64748b"}
        c = colors.get(status, "#94a3b8")
        return f'<span style="background:{c};color:#fff;padding:1px 7px;border-radius:10px;font-size:11px">{status}</span>'

    # ── track rows ─────────────────────────────────────────────────────────────
    track_rows = ""
    track_metrics = m.get("track_metrics", {})
    for track, t in sorted(track_metrics.items(), key=lambda x: -x[1].get("coverage_pct", 0)):
        cov = t.get("coverage_pct", 0)
        conf = t.get("avg_confidence", 0)
        gen = t.get("generated", 0)
        total = t.get("total_pages", 0)
        appr = t.get("approved", 0)
        flagged = t.get("flagged", 0)
        short = track.split("-", 1)[1].replace("-", " ").title() if "-" in track else track
        track_rows += f"""
        <tr>
          <td style="padding:6px 10px;color:#94a3b8;font-size:12px">{track}</td>
          <td style="padding:6px 10px;color:#e2e8f0">{short}</td>
          <td style="padding:6px 10px;text-align:center;color:{conf_color(cov)};font-weight:600">{cov*100:.0f}%</td>
          <td style="padding:6px 10px">{pct_bar(cov, 100)}</td>
          <td style="padding:6px 10px;text-align:center;color:#e2e8f0">{gen}/{total}</td>
          <td style="padding:6px 10px;text-align:center;color:#22c55e">{appr}</td>
          <td style="padding:6px 10px;text-align:center;color:#f59e0b">{flagged}</td>
          <td style="padding:6px 10px;text-align:center;color:{conf_color(conf)}">{conf:.2f}</td>
        </tr>"""

    # ── recent runs rows ────────────────────────────────────────────────────────
    run_rows = ""
    for r in recent_runs[:20]:
        topic = r.get("topic", "")
        track = r.get("track", "").split("-", 1)[-1].replace("-", " ")
        conf = r.get("confidence", 0)
        status = r.get("status", "")
        revisions = r.get("revision_count", 0)
        committed = "✓" if r.get("committed") else "—"
        ts = r.get("finished_at", "")[:16].replace("T", " ")
        issues = r.get("review_issues") or []
        issues_str = "; ".join(issues[:2]) if issues else "—"
        run_rows += f"""
        <tr>
          <td style="padding:5px 10px;color:#94a3b8;font-size:11px">{ts}</td>
          <td style="padding:5px 10px;color:#e2e8f0;font-weight:500">{topic}</td>
          <td style="padding:5px 10px;color:#94a3b8;font-size:11px">{track}</td>
          <td style="padding:5px 10px">{status_badge(status)}</td>
          <td style="padding:5px 10px;text-align:center;color:{conf_color(conf)};font-weight:600">{conf:.2f}</td>
          <td style="padding:5px 10px;text-align:center;color:#94a3b8">{revisions}</td>
          <td style="padding:5px 10px;text-align:center;color:#22c55e">{committed}</td>
          <td style="padding:5px 10px;color:#64748b;font-size:11px;max-width:220px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="{issues_str}">{issues_str}</td>
        </tr>"""

    # ── budget ─────────────────────────────────────────────────────────────────
    budget = m.get("budget", {})
    usage = budget.get("usage_usd", 0)
    limit = budget.get("limit_usd")
    remaining = budget.get("remaining_usd")
    bmode = budget.get("mode", "full")
    bmode_color = {"full": "#22c55e", "reduced": "#f59e0b", "paused": "#ef4444"}.get(bmode, "#94a3b8")
    budget_bar = ""
    if limit and limit > 0:
        used_pct = min(usage / limit, 1.0)
        budget_bar = f"""
        <div style="margin-top:6px">
          <div style="background:#1e293b;border-radius:4px;height:10px;width:200px;display:inline-block">
            <div style="background:#f59e0b;border-radius:4px;height:10px;width:{int(used_pct*200)}px"></div>
          </div>
          <span style="color:#94a3b8;font-size:11px;margin-left:8px">${usage:.2f} / ${limit:.2f}</span>
        </div>"""

    # ── error signals ──────────────────────────────────────────────────────────
    signals = m.get("error_signals", {})

    def signal_bar(val: float, label: str, max_val: float = 1.0) -> str:
        pct = min(val / max_val, 1.0) if max_val > 0 else 0
        color = "#22c55e" if pct < 0.2 else "#f59e0b" if pct < 0.6 else "#ef4444"
        return f"""
        <div style="display:flex;align-items:center;gap:10px;margin:4px 0">
          <span style="color:#94a3b8;font-size:12px;width:140px">{label}</span>
          <div style="background:#1e293b;border-radius:4px;height:7px;width:150px">
            <div style="background:{color};border-radius:4px;height:7px;width:{int(pct*150)}px"></div>
          </div>
          <span style="color:{color};font-size:12px;font-weight:600">{val:.2f}</span>
        </div>"""

    # ── quality trend ──────────────────────────────────────────────────────────
    qt = m.get("quality_trend", {})

    # ── total stats ────────────────────────────────────────────────────────────
    total = m.get("total_pages", 0)
    generated = m.get("generated_pages", 0)
    approved = m.get("approved_pages", 0)
    stubs = m.get("stub_pages", 0)
    avg_conf = m.get("avg_confidence", 0)
    cov_pct = m.get("coverage_pct", 0)
    obs_at = m.get("observed_at", "—")

    # sprint queue size
    sprint_pending = 0
    sprint_path = Path(__file__).parent / "sprints" / "current.md"
    if sprint_path.exists():
        sprint_pending = sprint_path.read_text().count("- [ ]")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>FAIRE Wiki — Dashboard</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ background: #0f172a; color: #e2e8f0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", monospace; font-size: 13px; }}
    h2 {{ font-size: 11px; text-transform: uppercase; letter-spacing: 0.08em; color: #64748b; margin-bottom: 12px; }}
    .card {{ background: #1e293b; border-radius: 8px; padding: 18px 20px; }}
    table {{ width: 100%; border-collapse: collapse; }}
    tr:hover td {{ background: rgba(255,255,255,0.03); }}
    th {{ color: #475569; font-size: 11px; text-transform: uppercase; letter-spacing: 0.05em; padding: 6px 10px; text-align: left; border-bottom: 1px solid #1e293b; }}
    .stat {{ text-align: center; }}
    .stat .n {{ font-size: 28px; font-weight: 700; color: #f8fafc; }}
    .stat .l {{ font-size: 11px; color: #64748b; margin-top: 2px; text-transform: uppercase; letter-spacing: 0.05em; }}
    .dot {{ width: 8px; height: 8px; border-radius: 50%; display: inline-block; margin-right: 5px; }}
    .refreshing {{ animation: pulse 1.5s infinite; }}
    @keyframes pulse {{ 0%,100% {{ opacity:1 }} 50% {{ opacity:0.4 }} }}
  </style>
</head>
<body>
<div style="max-width:1100px;margin:0 auto;padding:24px 20px">

  <!-- Header -->
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:24px">
    <div>
      <div style="font-size:20px;font-weight:700;color:#f8fafc">FAIRE Wiki</div>
      <div style="color:#475569;font-size:12px;margin-top:2px">Autonomous generation dashboard</div>
    </div>
    <div style="text-align:right">
      <div style="color:#475569;font-size:11px">observed {obs_at}</div>
      <div style="color:#475569;font-size:11px;margin-top:3px" id="countdown">auto-refresh in 8s</div>
    </div>
  </div>

  <!-- Top stats -->
  <div style="display:grid;grid-template-columns:repeat(6,1fr);gap:12px;margin-bottom:20px">
    <div class="card stat">
      <div class="n" style="color:#60a5fa">{cov_pct*100:.1f}%</div>
      <div class="l">Coverage</div>
    </div>
    <div class="card stat">
      <div class="n">{generated}</div>
      <div class="l">Generated</div>
    </div>
    <div class="card stat">
      <div class="n" style="color:#22c55e">{approved}</div>
      <div class="l">Approved</div>
    </div>
    <div class="card stat">
      <div class="n" style="color:#f59e0b">{stubs}</div>
      <div class="l">Stubs left</div>
    </div>
    <div class="card stat">
      <div class="n" style="color:{conf_color(avg_conf)}">{avg_conf:.2f}</div>
      <div class="l">Avg conf</div>
    </div>
    <div class="card stat">
      <div class="n" style="color:{bmode_color}">{sprint_pending}</div>
      <div class="l">In queue</div>
    </div>
  </div>

  <!-- Budget + error signals -->
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:20px">
    <div class="card">
      <h2>Budget</h2>
      <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px">
        <span class="dot" style="background:{bmode_color}"></span>
        <span style="color:{bmode_color};font-weight:600;font-size:14px">{bmode.upper()}</span>
        <span style="color:#475569">mode</span>
      </div>
      <div style="color:#e2e8f0;font-size:14px">${usage:.3f} used
        {"/ $"+str(limit) if limit else " (no limit set)"}
      </div>
      {budget_bar}
      <div style="color:#475569;font-size:11px;margin-top:8px">
        WRITER: ~anthropic/claude-sonnet-latest · REVIEWER: gemini-3.1-pro
      </div>
    </div>
    <div class="card">
      <h2>Error signals</h2>
      {signal_bar(signals.get("coverage_deficit", 0), "Coverage deficit")}
      {signal_bar(signals.get("quality_deficit", 0), "Quality deficit")}
      {signal_bar(min(signals.get("flagged_pages", 0), 10), "Flagged pages", 10)}
      {signal_bar(signals.get("budget_pressure", 0), "Budget pressure")}
    </div>
  </div>

  <!-- Quality trend -->
  <div class="card" style="margin-bottom:20px">
    <h2>Quality trend (last {qt.get("window_size", 10)} runs)</h2>
    <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:20px">
      <div>
        <div style="font-size:22px;font-weight:700;color:{conf_color(qt.get('avg_confidence',0))}">{qt.get('avg_confidence',0):.2f}</div>
        <div style="color:#475569;font-size:11px">avg confidence</div>
      </div>
      <div>
        <div style="font-size:22px;font-weight:700;color:{conf_color(qt.get('approval_rate',0))}">{qt.get('approval_rate',0)*100:.0f}%</div>
        <div style="color:#475569;font-size:11px">approval rate</div>
      </div>
      <div>
        <div style="font-size:22px;font-weight:700;color:#94a3b8">{qt.get('avg_revisions',0):.1f}</div>
        <div style="color:#475569;font-size:11px">avg revisions</div>
      </div>
      <div>
        <div style="font-size:22px;font-weight:700;color:{'#22c55e' if qt.get('confidence_delta',0)>=0 else '#ef4444'}">
          {'↑' if qt.get('confidence_delta',0)>=0 else '↓'}{abs(qt.get('confidence_delta',0)):.2f}
        </div>
        <div style="color:#475569;font-size:11px">conf delta</div>
      </div>
    </div>
  </div>

  <!-- Track coverage table -->
  <div class="card" style="margin-bottom:20px">
    <h2>Per-track coverage</h2>
    <table>
      <thead><tr>
        <th>Track</th><th>Name</th><th>Coverage</th><th></th>
        <th>Gen/Total</th><th>Approved</th><th>Flagged</th><th>Avg Conf</th>
      </tr></thead>
      <tbody>{track_rows}</tbody>
    </table>
  </div>

  <!-- Recent runs table -->
  <div class="card">
    <h2>Recent runs (last 20)</h2>
    <table>
      <thead><tr>
        <th>Time</th><th>Topic</th><th>Track</th><th>Status</th>
        <th>Conf</th><th>Rev</th><th>Committed</th><th>Issues</th>
      </tr></thead>
      <tbody>{run_rows if run_rows else '<tr><td colspan="8" style="text-align:center;color:#475569;padding:20px">No runs yet — cycle is starting</td></tr>'}</tbody>
    </table>
  </div>

</div>

<script>
  // Live countdown + auto-refresh
  let secs = 8;
  const el = document.getElementById('countdown');
  setInterval(() => {{
    secs--;
    if (secs <= 0) {{ location.reload(); }}
    el.textContent = `auto-refresh in ${{secs}}s`;
  }}, 1000);
</script>
</body>
</html>"""
    return HTMLResponse(html)


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


@app.get("/progress")
def get_progress():
    """Live view of pages currently being generated."""
    with _active_lock:
        return {"active_pages": dict(_active), "count": len(_active)}


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
    with _active_lock:
        _active[topic] = "running"
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
    finally:
        with _active_lock:
            _active.pop(topic, None)


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
    supervisor_future = _POOL.submit(_run_supervisor_background, dry_run)
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

    # Step 3: Optionally run a full cycle immediately (in background so HTTP starts first).
    # Chain it to fire ONLY AFTER the bootstrap supervisor finishes — otherwise the
    # cycle's internal sprint_job can fire before the queue is populated and return
    # zero results (the race condition that left the agent idle for 8+ minutes after
    # boot in earlier sessions).
    if run_now:
        click.echo("Scheduling immediate cycle (waits for supervisor bootstrap first)...")

        def _cycle_after_supervisor():
            try:
                # Cap the wait so a hung supervisor doesn't block forever.
                supervisor_future.result(timeout=180)
            except Exception as exc:
                print(f"[boot] Supervisor bootstrap failed or timed out: {exc}")
                # Fall through and run the cycle anyway — full_cycle_job runs its
                # own supervisor pass as step 1, so we'll still get a fresh queue.
            _run_cycle_background()

        _POOL.submit(_cycle_after_supervisor)

    click.echo(
        f"\nFrontier Wiki agent server running — http://{host}:{port}\n"
        f"  Cycle every {interval}h | dry_run={dry_run}\n"
        f"  GET / for health dashboard | POST /trigger to run a cycle\n"
    )

    uvicorn.run(app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    serve()
