"""Wiki Observer — the sensor layer of the closed-loop control system.

The observer aggregates all available quality signals into a structured
WikiObservation snapshot. This is the system's single source of truth
about its own state at any given moment.

Control-system analogy:
  Observer = sensor
  WikiObservation = plant state vector
  error_signals = deviation from set points
  Supervisor = controller (reads error signals, generates actions)
  Pipeline graph = actuator (executes actions)
  log_run → observer → metrics.json = feedback path

Set points (configurable via env vars):
  QUALITY_SETPOINT    = 0.85  (reviewer confidence per page)
  COVERAGE_SETPOINT   = 0.80  (fraction of stubs generated per track)
  STALENESS_THRESHOLD = 180   (days before a page's SotA is stale)
  BUDGET_MINIMUM      = 1.0   (USD remaining before generation pauses)

Usage:
  from frontier_agents.observer import observe, check_budget, write_metrics_json

  obs = observe(docs_dir="../docs", runs_dir="runs")
  write_metrics_json(obs, runs_dir="runs")
  write_observer_page(obs, docs_dir="../docs")
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path


# ── Set points ────────────────────────────────────────────────────────────────

QUALITY_SETPOINT = float(os.getenv("QUALITY_SETPOINT", "0.85"))
COVERAGE_SETPOINT = float(os.getenv("COVERAGE_SETPOINT", "0.80"))
STALENESS_THRESHOLD_DAYS = int(os.getenv("STALENESS_THRESHOLD", "180"))
BUDGET_MINIMUM_USD = float(os.getenv("BUDGET_MINIMUM", "1.0"))
BUDGET_REDUCED_USD = float(os.getenv("BUDGET_REDUCED", "3.0"))

# Estimated cost per action (USD) — used for budget forecasting
COST_ESTIMATES = {
    "generate": 0.20,   # full page: opus writer + gemini reviewer
    "improve":  0.15,   # improve pass
    "mvb-only": 0.07,   # MVB section only
}


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class TrackMetrics:
    track: str
    total_pages: int = 0       # stubs + generated
    generated: int = 0         # pages with real content (not stubs)
    approved: int = 0          # confidence >= threshold
    flagged: int = 0           # generated but below threshold
    coverage_pct: float = 0.0  # generated / total
    avg_confidence: float = 0.0
    pages_with_mvb: int = 0
    stale_pages: int = 0       # updated > STALENESS_THRESHOLD days ago


@dataclass
class QualityTrend:
    """Computed from the last N runs in runs.jsonl."""
    window_size: int = 0
    avg_confidence: float = 0.0
    approval_rate: float = 0.0   # fraction approved on first pass (rev_count=0)
    avg_revisions: float = 0.0
    confidence_delta: float = 0.0  # avg_conf[last5] - avg_conf[prev5] — positive = improving
    top_issues: list[str] = field(default_factory=list)


@dataclass
class BudgetState:
    usage_usd: float = 0.0
    limit_usd: float | None = None
    remaining_usd: float | None = None
    is_free_tier: bool = False
    mode: str = "full"  # "full" | "reduced" | "paused"
    error: str = ""     # set if the API call failed


@dataclass
class WikiObservation:
    observed_at: str

    # ── Global metrics ────────────────────────────────────────────
    total_pages: int = 0
    stub_pages: int = 0
    generated_pages: int = 0
    approved_pages: int = 0
    avg_confidence: float = 0.0
    coverage_pct: float = 0.0  # generated / total (global)
    approved_pct: float = 0.0  # approved / generated
    pages_with_mvb: int = 0
    tracks_covered: int = 0
    tracks_total: int = 10

    # ── Per-track breakdown ───────────────────────────────────────
    track_metrics: dict[str, TrackMetrics] = field(default_factory=dict)

    # ── Quality trend ─────────────────────────────────────────────
    quality_trend: QualityTrend = field(default_factory=QualityTrend)

    # ── Budget ────────────────────────────────────────────────────
    budget: BudgetState = field(default_factory=BudgetState)

    # ── Error signals (deviation from set points, positive = deficit) ─────────
    # e.g. {"coverage": 0.68, "quality": -0.10, "stale_pages": 3}
    error_signals: dict[str, float] = field(default_factory=dict)


# ── Core observation function ─────────────────────────────────────────────────

def observe(
    docs_dir: str | Path = "../docs",
    runs_dir: str | Path = "runs",
) -> WikiObservation:
    """Build a full WikiObservation snapshot from the current filesystem state."""
    docs_path = Path(docs_dir)
    runs_path = Path(runs_dir) if Path(runs_dir).is_absolute() else (
        Path(__file__).parent.parent.parent / runs_dir
    )
    now = datetime.now(timezone.utc)
    from .tools import display_ts
    obs = WikiObservation(observed_at=display_ts())

    # ── 1. Scan curriculum filesystem ─────────────────────────────
    curriculum = docs_path / "curriculum"
    track_page_counts: dict[str, dict] = {}  # track → {total, stub, generated}

    if curriculum.exists():
        for page in curriculum.rglob("*.md"):
            if page.name == "index.md":
                continue
            # v2 layout: docs/curriculum/core/<track>/<page_type>/<slug>.md
            # legacy:    docs/curriculum/<track>/<slug>.md
            parts = page.parent.parts
            if "core" in parts:
                ci = parts.index("core")
                track = parts[ci + 1] if ci + 1 < len(parts) else page.parent.name
            else:
                track = page.parent.name
            if track not in track_page_counts:
                track_page_counts[track] = {"total": 0, "stubs": 0, "stale": 0}

            content = page.read_text(encoding="utf-8", errors="ignore")
            track_page_counts[track]["total"] += 1

            if "🚧" in content or "Agent-generated content pending" in content:
                track_page_counts[track]["stubs"] += 1
            else:
                # Check SotA staleness from frontmatter updated: field
                updated_match = re.search(r"updated:\s*(\d{4}-\d{2}-\d{2})", content)
                if updated_match:
                    try:
                        updated_dt = datetime.strptime(updated_match.group(1), "%Y-%m-%d")
                        age_days = (now.date() - updated_dt.date()).days
                        if age_days > STALENESS_THRESHOLD_DAYS:
                            track_page_counts[track]["stale"] += 1
                    except ValueError:
                        pass

    # ── 2. Read runs.jsonl for quality metrics ─────────────────────
    jsonl_path = runs_path / "runs.jsonl"
    all_runs: list[dict] = []
    latest_per_topic: dict[str, dict] = {}

    if jsonl_path.exists():
        with jsonl_path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        r = json.loads(line)
                        all_runs.append(r)
                        topic = r.get("topic", "")
                        if topic:
                            latest_per_topic[topic] = r
                    except json.JSONDecodeError:
                        pass

    # ── 3. Build per-track metrics ─────────────────────────────────
    track_conf: dict[str, list[float]] = {}
    for run in latest_per_topic.values():
        t = run.get("track", "")
        conf = run.get("confidence", 0.0)
        if t:
            track_conf.setdefault(t, []).append(conf)

    for track, counts in track_page_counts.items():
        total = counts["total"]
        stubs = counts["stubs"]
        generated = total - stubs
        confs = track_conf.get(track, [])
        avg_conf = sum(confs) / len(confs) if confs else 0.0

        track_runs = [r for r in latest_per_topic.values() if r.get("track") == track]
        approved = sum(1 for r in track_runs if r.get("status") == "approved")
        flagged = sum(1 for r in track_runs if r.get("status") == "flagged")
        with_mvb = sum(1 for r in track_runs if r.get("has_mvb"))

        obs.track_metrics[track] = TrackMetrics(
            track=track,
            total_pages=total,
            generated=generated,
            approved=approved,
            flagged=flagged,
            coverage_pct=generated / total if total else 0.0,
            avg_confidence=avg_conf,
            pages_with_mvb=with_mvb,
            stale_pages=counts.get("stale", 0),
        )

    # ── 4. Global rollups ──────────────────────────────────────────
    obs.total_pages = sum(t["total"] for t in track_page_counts.values())
    obs.stub_pages = sum(t["stubs"] for t in track_page_counts.values())
    obs.generated_pages = obs.total_pages - obs.stub_pages
    obs.approved_pages = sum(1 for r in latest_per_topic.values() if r.get("status") == "approved")
    obs.pages_with_mvb = sum(1 for r in latest_per_topic.values() if r.get("has_mvb"))
    obs.tracks_covered = sum(
        1 for tm in obs.track_metrics.values() if tm.generated > 0
    )

    all_confs = [r.get("confidence", 0) for r in latest_per_topic.values()]
    obs.avg_confidence = sum(all_confs) / len(all_confs) if all_confs else 0.0
    obs.coverage_pct = obs.generated_pages / obs.total_pages if obs.total_pages else 0.0
    obs.approved_pct = obs.approved_pages / obs.generated_pages if obs.generated_pages else 0.0

    # ── 5. Quality trend (last 10 runs) ───────────────────────────
    obs.quality_trend = _compute_quality_trend(all_runs)

    # ── 6. Budget ──────────────────────────────────────────────────
    obs.budget = check_budget()

    # ── 7. Error signals ───────────────────────────────────────────
    obs.error_signals = compute_error_signals(obs)

    return obs


def compute_error_signals(obs: WikiObservation) -> dict[str, float]:
    """Compute deviation from set points. Positive = deficit (need more work)."""
    signals: dict[str, float] = {}

    # Coverage deficit: fraction of pages still stubbed
    signals["coverage_deficit"] = max(0.0, COVERAGE_SETPOINT - obs.coverage_pct)

    # Quality deficit: how far avg confidence is from set point
    signals["quality_deficit"] = max(0.0, QUALITY_SETPOINT - obs.avg_confidence)

    # Stale pages count
    signals["stale_pages"] = float(
        sum(tm.stale_pages for tm in obs.track_metrics.values())
    )

    # Flagged pages (generated but below threshold)
    signals["flagged_pages"] = float(
        sum(tm.flagged for tm in obs.track_metrics.values())
    )

    # Budget pressure (0 = no pressure, 1 = at minimum)
    remaining = obs.budget.remaining_usd
    if remaining is not None:
        signals["budget_pressure"] = max(0.0, 1.0 - remaining / max(BUDGET_REDUCED_USD, 1.0))
    else:
        signals["budget_pressure"] = 0.0

    return signals


def _compute_quality_trend(all_runs: list[dict], window: int = 10) -> QualityTrend:
    """Analyse the last N runs for quality trends."""
    if not all_runs:
        return QualityTrend()

    recent = all_runs[-window:]
    confs = [r.get("confidence", 0) for r in recent]
    approved_first_pass = [r for r in recent if r.get("status") == "approved" and r.get("revision_count", 0) == 0]
    revisions = [r.get("revision_count", 0) for r in recent]

    # Confidence delta: compare first half vs second half of window
    mid = len(confs) // 2
    first_half = confs[:mid] if mid else confs
    second_half = confs[mid:] if mid else confs
    avg_first = sum(first_half) / len(first_half) if first_half else 0
    avg_second = sum(second_half) / len(second_half) if second_half else 0
    delta = avg_second - avg_first  # positive = improving

    return QualityTrend(
        window_size=len(recent),
        avg_confidence=sum(confs) / len(confs) if confs else 0.0,
        approval_rate=len(approved_first_pass) / len(recent) if recent else 0.0,
        avg_revisions=sum(revisions) / len(revisions) if revisions else 0.0,
        confidence_delta=round(delta, 3),
        top_issues=[],  # populated by LLM analysis in supervisor
    )


# ── Budget query ──────────────────────────────────────────────────────────────

def check_budget() -> BudgetState:
    """Query OpenRouter for current credit usage and remaining balance.

    Returns a BudgetState with the current financial constraint on the system.
    The budget mode determines which actions the actuator is allowed to take:
      full     → all actions allowed (opus writer, gemini reviewer)
      reduced  → RESEARCH_MODEL for writes, skip MVB passes
      paused   → audit only, no generation
    """
    try:
        import httpx
    except ImportError:
        return BudgetState(error="httpx not installed")

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return BudgetState(error="OPENROUTER_API_KEY not set")

    try:
        resp = httpx.get(
            "https://openrouter.ai/api/v1/auth/key",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=8,
        )
        resp.raise_for_status()
        data = resp.json().get("data", {})
    except Exception as e:
        return BudgetState(error=str(e)[:100])

    usage = float(data.get("usage", 0) or 0)
    limit = data.get("limit")
    remaining = data.get("limit_remaining")
    is_free = bool(data.get("is_free_tier", False))

    # Allow a local soft cap via BUDGET_LIMIT_USD even when OpenRouter has no hard limit.
    # This lets users set a $10 cap without needing an OpenRouter account-level limit.
    local_limit = os.getenv("BUDGET_LIMIT_USD")
    if local_limit and remaining is None:
        try:
            cap = float(local_limit)
            remaining = max(0.0, cap - usage)
            limit = cap
        except ValueError:
            pass

    # Determine operating mode based on remaining budget
    if remaining is None:
        mode = "full"  # no limit set (pay-as-you-go)
    elif remaining < BUDGET_MINIMUM_USD:
        mode = "paused"
    elif remaining < BUDGET_REDUCED_USD:
        mode = "reduced"
    else:
        mode = "full"

    return BudgetState(
        usage_usd=round(usage, 4),
        limit_usd=float(limit) if limit is not None else None,
        remaining_usd=round(float(remaining), 4) if remaining is not None else None,
        is_free_tier=is_free,
        mode=mode,
    )


# ── Persistence ───────────────────────────────────────────────────────────────

def write_metrics_json(obs: WikiObservation, runs_dir: str | Path = "runs") -> None:
    """Persist the full observation to agents/runs/metrics.json."""
    runs_path = Path(runs_dir) if Path(runs_dir).is_absolute() else (
        Path(__file__).parent.parent.parent / runs_dir
    )
    runs_path.mkdir(parents=True, exist_ok=True)

    # Convert dataclasses to dict
    data = _obs_to_dict(obs)
    (runs_path / "metrics.json").write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _obs_to_dict(obs: WikiObservation) -> dict:
    """Serialize WikiObservation to a JSON-safe dict."""
    d = asdict(obs)
    return d


def write_observer_page(obs: WikiObservation, docs_dir: str | Path = "../docs") -> None:
    """Write docs/system/observer.md — the live control dashboard."""
    docs_path = Path(docs_dir)
    system_dir = docs_path / "system"
    system_dir.mkdir(parents=True, exist_ok=True)

    h = obs
    t = obs.quality_trend
    b = obs.budget
    e = obs.error_signals

    def _status_icon(deficit: float, threshold: float = 0.05) -> str:
        if deficit > threshold * 2:
            return "🔴"
        if deficit > threshold:
            return "🟡"
        return "🟢"

    def _budget_icon(mode: str) -> str:
        return {"full": "🟢", "reduced": "🟡", "paused": "🔴"}.get(mode, "⚪")

    def _trend_arrow(delta: float) -> str:
        if delta > 0.02:
            return "↑ improving"
        if delta < -0.02:
            return "↓ declining"
        return "→ stable"

    lines = [
        "---",
        "title: Observer Dashboard",
        "description: Closed-loop control system — real-time wiki quality signals",
        "---",
        "",
        "# Observer Dashboard",
        "",
        f"> Last observed: **{h.observed_at}** · "
        f"[Source: `agents/runs/metrics.json`]",
        "",
        "## Control State",
        "",
        "| Signal | Set Point | Current | Error | Status |",
        "|---|---|---|---|---|",
        f"| Coverage | {COVERAGE_SETPOINT:.0%} | {h.coverage_pct:.1%} "
        f"| {e.get('coverage_deficit', 0):.1%} deficit "
        f"| {_status_icon(e.get('coverage_deficit', 0))} |",
        f"| Quality | {QUALITY_SETPOINT:.2f} | {h.avg_confidence:.2f} "
        f"| {e.get('quality_deficit', 0):.2f} deficit "
        f"| {_status_icon(e.get('quality_deficit', 0))} |",
        f"| Stale pages | 0 | {int(e.get('stale_pages', 0))} "
        f"| {int(e.get('stale_pages', 0))} pages "
        f"| {_status_icon(e.get('stale_pages', 0), threshold=1)} |",
        f"| Flagged pages | 0 | {int(e.get('flagged_pages', 0))} "
        f"| {int(e.get('flagged_pages', 0))} pages "
        f"| {_status_icon(e.get('flagged_pages', 0), threshold=1)} |",
        f"| Budget | >${BUDGET_MINIMUM_USD:.0f} remaining | "
        + (f"${b.remaining_usd:.2f}" if b.remaining_usd is not None else "unlimited")
        + f" | {b.mode} mode | {_budget_icon(b.mode)} |",
        "",
        "## Coverage",
        "",
        f"**{h.generated_pages}** pages generated of **{h.total_pages}** total "
        f"({h.coverage_pct:.1%}) · **{h.stub_pages}** stubs remaining · "
        f"**{h.tracks_covered}** / {h.tracks_total} tracks with content",
        "",
        "## Per-Track Status",
        "",
        "| Track | Total | Generated | Approved | Coverage | Avg Conf | MVB | Stale |",
        "|---|---|---|---|---|---|---|---|",
    ]

    for track in sorted(obs.track_metrics):
        tm = obs.track_metrics[track]
        cov_icon = "🟢" if tm.coverage_pct >= COVERAGE_SETPOINT else ("🟡" if tm.coverage_pct > 0 else "🔴")
        qual_icon = "🟢" if tm.avg_confidence >= QUALITY_SETPOINT else ("🟡" if tm.avg_confidence > 0 else "⚫")
        lines.append(
            f"| {track} | {tm.total_pages} | {tm.generated} "
            f"| {tm.approved} | {cov_icon} {tm.coverage_pct:.0%} "
            f"| {qual_icon} {tm.avg_confidence:.2f} | {tm.pages_with_mvb} | {tm.stale_pages} |"
        )

    lines += [
        "",
        "## Quality Trend",
        "",
        f"Last **{t.window_size}** runs · "
        f"avg confidence **{t.avg_confidence:.2f}** · "
        f"first-pass approval **{t.approval_rate:.0%}** · "
        f"avg revisions **{t.avg_revisions:.1f}** · "
        f"trend **{_trend_arrow(t.confidence_delta)}** ({t.confidence_delta:+.3f})",
        "",
        "## Budget",
        "",
    ]

    if b.error:
        lines.append(f"> Budget query failed: `{b.error}`")
    else:
        usage_str = f"${b.usage_usd:.4f} used"
        limit_str = f" of ${b.limit_usd:.2f} limit" if b.limit_usd else " (no limit)"
        remaining_str = f" · ${b.remaining_usd:.2f} remaining" if b.remaining_usd is not None else ""
        mode_str = f" · **{b.mode} mode**"
        lines.append(f"{usage_str}{limit_str}{remaining_str}{mode_str}")
        lines.append("")
        lines.append("| Action | Est. Cost | Actions in Budget |")
        lines.append("|---|---|---|")
        for action, cost in COST_ESTIMATES.items():
            if b.remaining_usd is not None:
                count = int(b.remaining_usd / cost)
                lines.append(f"| {action} | ${cost:.2f} | ~{count} pages |")
            else:
                lines.append(f"| {action} | ${cost:.2f} | unlimited |")

    lines += [
        "",
        "---",
        "",
        "*This dashboard is updated automatically after every agent run and at the start of each 48h cycle.*",
        "*Source: `agents/runs/metrics.json` · Pipeline: `observer.observe()` → `write_observer_page()`*",
    ]

    (system_dir / "observer.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
