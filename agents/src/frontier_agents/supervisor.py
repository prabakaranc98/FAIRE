"""Supervising agent — the wiki's systems manager.

Responsibilities:
  1. ASSESS   — read current wiki state (pages, runs, confidence scores)
  2. DIAGNOSE — identify quality regressions, content gaps, stale SotA
  3. PRIORITISE — rank work by impact (critical fixes > gaps > improvements)
  4. QUEUE    — update agents/sprints/current.md with prioritised work items
  5. REPORT   — write a health summary to docs/system/supervisor.md

The supervisor uses LLM reasoning (RESEARCH_MODEL — fast + capable) to make
editorial judgements, but only reads from disk and the runs log — it never
directly edits wiki pages. The editorial pipeline does that.

Invocation:
    from frontier_agents.supervisor import run_supervisor
    report = run_supervisor(docs_dir="../docs", sprints_dir="sprints", dry_run=False)

CLI (via generate.py):
    uv run python generate.py supervise
    uv run python generate.py supervise --dry-run       # report only, no sprint write
    uv run python generate.py supervise --verbose
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.messages import HumanMessage

from .audit import AuditReport, audit_wiki
from .llm import get_llm


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class WikiHealth:
    total_pages: int = 0
    stub_pages: int = 0
    generated_pages: int = 0
    approved_pages: int = 0
    flagged_pages: int = 0
    avg_confidence: float = 0.0
    tracks_covered: int = 0
    tracks_total: int = 15
    pages_with_mvb: int = 0
    critical_issues: int = 0
    warnings: int = 0


@dataclass
class SupervisorAction:
    priority: int               # 1=critical, 2=high, 3=medium, 4=low
    action: str                 # "generate" | "improve" | "mvb-only" | "fix"
    topic: str
    track: str
    page_type: str              # "arc-entry" | "core-concept" | "supporting"
    depth_emphasis: list[str]
    reason: str                 # why this was prioritised
    arc: str = ""
    arc_position: int = 0


@dataclass
class SupervisorReport:
    generated_at: str
    health: WikiHealth
    audit: AuditReport
    actions: list[SupervisorAction] = field(default_factory=list)
    llm_analysis: str = ""      # supervisor's editorial judgement

    def to_markdown(self) -> str:
        h = self.health
        lines = [
            "---",
            "title: Supervisor Report",
            "description: Automated wiki health assessment and editorial priorities",
            "---",
            "",
            "# Supervisor Report",
            "",
            f"> Generated: **{self.generated_at}** by the Frontier Wiki supervising agent",
            "",
            "## Wiki Health",
            "",
            "| Metric | Value |",
            "|---|---|",
            f"| Total pages (stubs + generated) | **{h.total_pages}** |",
            f"| Stub pages (not yet generated) | **{h.stub_pages}** |",
            f"| Generated pages | **{h.generated_pages}** |",
            f"| Approved (conf ≥ 0.8) | **{h.approved_pages}** ({_pct(h.approved_pages, h.generated_pages)}%) |",
            f"| Flagged (conf < 0.8) | **{h.flagged_pages}** |",
            f"| Avg reviewer confidence | **{h.avg_confidence:.2f}** |",
            f"| Tracks with content | **{h.tracks_covered}** / {h.tracks_total} |",
            f"| Pages with MVB | **{h.pages_with_mvb}** |",
            f"| Critical audit issues | **{h.critical_issues}** |",
            f"| Warnings | **{h.warnings}** |",
            "",
            "## Editorial Analysis",
            "",
            self.llm_analysis or "_No analysis available._",
            "",
            "## Prioritised Work Queue",
            "",
            "| # | Priority | Action | Topic | Track | Reason |",
            "|---|---|---|---|---|---|",
        ]
        for i, a in enumerate(self.actions[:20], 1):
            pri_label = ["", "🔴 critical", "🟠 high", "🟡 medium", "🟢 low"][a.priority]
            lines.append(
                f"| {i} | {pri_label} | {a.action} | {a.topic} | {a.track} | {a.reason} |"
            )
        lines += [
            "",
            "## Audit Issues",
            "",
            self.audit.to_markdown(),
            "",
            "---",
            "",
            "*This report is regenerated automatically. "
            "The supervisor updates `agents/sprints/current.md` with the top work items.*",
        ]
        return "\n".join(lines)


def _pct(n: int, d: int) -> int:
    return (100 * n // d) if d else 0


# ── Core logic ────────────────────────────────────────────────────────────────

def run_supervisor(
    docs_dir: str = "../docs",
    runs_dir: str = "runs",
    sprints_dir: str = "sprints",
    dry_run: bool = False,
    verbose: bool = False,
) -> SupervisorReport:
    """Full supervisor cycle: observe → assess → diagnose → prioritise → queue → report."""
    now = datetime.now(timezone.utc)
    docs_path = Path(docs_dir)
    runs_path = Path(__file__).parent.parent.parent / runs_dir
    sprints_path = Path(__file__).parent.parent.parent / sprints_dir

    if verbose:
        print(f"[supervisor] Starting assessment — {now.strftime('%Y-%m-%d %H:%M UTC')}")

    # 1. OBSERVE — unified sensor layer (also updates metrics.json + observer.md)
    obs = None
    try:
        from .observer import observe, write_metrics_json, write_observer_page
        obs = observe(docs_dir=str(docs_path), runs_dir=str(runs_path))
        if not dry_run:
            write_metrics_json(obs, runs_path)
            write_observer_page(obs, docs_path)
    except Exception as e:
        if verbose:
            print(f"[supervisor] Observer failed (continuing without it): {e}")

    # 2. ASSESS — derive WikiHealth (from obs if available, else direct scan)
    if obs is not None:
        health = WikiHealth(
            total_pages=obs.total_pages,
            stub_pages=obs.stub_pages,
            generated_pages=obs.generated_pages,
            approved_pages=obs.approved_pages,
            flagged_pages=sum(tm.flagged for tm in obs.track_metrics.values()),
            avg_confidence=obs.avg_confidence,
            tracks_covered=obs.tracks_covered,
            tracks_total=obs.tracks_total,
            pages_with_mvb=obs.pages_with_mvb,
        )
    else:
        health = _assess_wiki(docs_path, runs_path)

    audit = audit_wiki(docs_path)
    health.critical_issues = len(audit.critical)
    health.warnings = len(audit.warnings)

    if verbose:
        budget_str = (
            f", budget={obs.budget.mode} (${obs.budget.remaining_usd:.2f} left)"
            if obs and obs.budget.remaining_usd is not None
            else ""
        )
        print(
            f"[supervisor] Health: {health.generated_pages} generated, "
            f"{health.stub_pages} stubs, avg conf={health.avg_confidence:.2f}{budget_str}"
        )

    # 3. DIAGNOSE + PRIORITISE via LLM reasoning
    actions = _build_action_list(health, audit, runs_path, docs_path, obs=obs)
    llm_analysis = _llm_editorial_analysis(health, audit, actions, obs=obs, verbose=verbose)

    report = SupervisorReport(
        generated_at=now.strftime("%Y-%m-%d %H:%M UTC"),
        health=health,
        audit=audit,
        actions=actions,
        llm_analysis=llm_analysis,
    )

    # 4. QUEUE — write sprint if not dry run
    if not dry_run and actions:
        _update_sprint(actions[:15], sprints_path, now)
        if verbose:
            print(f"[supervisor] Updated sprint with {min(len(actions), 15)} items")

    # 5. REPORT — write to docs/system/supervisor.md
    _write_report(report, docs_path)
    if verbose:
        print(f"[supervisor] Report written to docs/system/supervisor.md")

    return report


def _assess_wiki(docs_path: Path, runs_path: Path) -> WikiHealth:
    """Count pages, read run logs, compute health metrics."""
    h = WikiHealth()

    # Count all curriculum pages (excluding index.md)
    curriculum = docs_path / "curriculum"
    if curriculum.exists():
        all_pages = [p for p in curriculum.rglob("*.md") if p.name != "index.md"]
        h.total_pages = len(all_pages)

        stub_count = 0
        for p in all_pages:
            content = p.read_text(encoding="utf-8", errors="ignore")
            if "🚧" in content or "Agent-generated content pending" in content:
                stub_count += 1
        h.stub_pages = stub_count

        # Count tracks with at least one non-stub page
        tracks_with_content = set()
        for p in all_pages:
            content = p.read_text(encoding="utf-8", errors="ignore")
            if "🚧" not in content and "Agent-generated content pending" not in content:
                track = p.parent.name
                tracks_with_content.add(track)
        h.tracks_covered = len(tracks_with_content)

    # Read run logs
    jsonl_path = runs_path / "runs.jsonl"
    if jsonl_path.exists():
        runs = []
        with jsonl_path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        runs.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

        # Latest run per topic
        latest: dict[str, dict] = {}
        for r in runs:
            topic = r.get("topic", "")
            if topic:
                latest[topic] = r

        records = list(latest.values())
        h.generated_pages = len(records)
        h.approved_pages = sum(1 for r in records if r.get("status") == "approved")
        h.flagged_pages = sum(1 for r in records if r.get("status") == "flagged")
        h.pages_with_mvb = sum(1 for r in records if r.get("has_mvb"))

        if records:
            h.avg_confidence = sum(r.get("confidence", 0) for r in records) / len(records)

    return h


def _build_action_list(
    health: WikiHealth,
    audit: AuditReport,
    runs_path: Path,
    docs_path: Path,
    obs=None,
) -> list[SupervisorAction]:
    """Rule-based prioritisation weighted by observer error signals.

    Budget modes:
      full    → all action types allowed
      reduced → generate actions use cheaper models (flag in reason); no low-priority
      paused  → no generation actions; only improve/fix for already-generated pages
    """
    from .observer import WikiObservation
    budget_mode = obs.budget.mode if obs is not None else "full"

    # Coverage deficit boosts priority of generate actions (0 → no boost, 0.5 → drop 1 level)
    coverage_deficit = obs.error_signals.get("coverage_deficit", 0.5) if obs else 0.5
    quality_deficit = obs.error_signals.get("quality_deficit", 0.1) if obs else 0.1

    actions: list[SupervisorAction] = []

    # Load latest runs for context
    latest_runs: dict[str, dict] = {}
    jsonl_path = runs_path / "runs.jsonl"
    if jsonl_path.exists():
        with jsonl_path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        r = json.loads(line)
                        topic = r.get("topic", "")
                        if topic:
                            latest_runs[topic] = r
                    except json.JSONDecodeError:
                        pass

    # Priority 1: Flagged pages (generated but below confidence threshold)
    for topic, r in latest_runs.items():
        if r.get("status") == "flagged":
            conf = r.get("confidence", 0)
            actions.append(SupervisorAction(
                priority=1,
                action="improve",
                topic=topic,
                track=r.get("track", ""),
                page_type=r.get("page_type", "core-concept"),
                depth_emphasis=r.get("depth_emphasis", ["applied"]),
                reason=f"Reviewer flagged (conf={conf:.2f}) — needs targeted revision",
            ))

    # Priority 2: Pages with critical audit issues
    critical_pages: set[str] = set()
    for issue in audit.critical:
        page_slug = Path(issue.page).stem
        if page_slug not in latest_runs:
            continue
        r = latest_runs[page_slug]
        if page_slug not in critical_pages:
            critical_pages.add(page_slug)
            actions.append(SupervisorAction(
                priority=2,
                action="improve",
                topic=page_slug,
                track=r.get("track", ""),
                page_type=r.get("page_type", "core-concept"),
                depth_emphasis=r.get("depth_emphasis", ["applied"]),
                reason=f"Audit critical: {issue.check} — {issue.message[:80]}",
            ))

    # Priority 3: Stub pages that haven't been attempted (new content gaps)
    # Skipped entirely when budget is "paused"
    if budget_mode != "paused":
        attempted_topics = set(latest_runs.keys())
        curriculum = docs_path / "curriculum"
        if curriculum.exists():
            for page in curriculum.rglob("*.md"):
                if page.name == "index.md":
                    continue
                content = page.read_text(encoding="utf-8", errors="ignore")
                if "🚧" in content or "Agent-generated content pending" in content:
                    topic_slug = page.stem
                    track = page.parent.name
                    if topic_slug not in attempted_topics:
                        reason = "Stub page — not yet generated"
                        if budget_mode == "reduced":
                            reason += " [reduced budget: cheaper model]"
                        actions.append(SupervisorAction(
                            priority=3,
                            action="generate",
                            topic=topic_slug,
                            track=track,
                            page_type="core-concept",
                            depth_emphasis=["applied"],
                            reason=reason,
                        ))

    # Priority 4: Pages approved but old (SotA may be stale)
    # Skip in reduced/paused mode to conserve budget
    if budget_mode == "full":
        for topic, r in latest_runs.items():
            if r.get("status") != "approved":
                continue
            finished_at = r.get("finished_at", "")
            if finished_at:
                try:
                    run_dt = datetime.fromisoformat(finished_at)
                    age_days = (datetime.now(timezone.utc) - run_dt).days
                    if age_days > 90:
                        actions.append(SupervisorAction(
                            priority=4,
                            action="improve",
                            topic=topic,
                            track=r.get("track", ""),
                            page_type=r.get("page_type", "core-concept"),
                            depth_emphasis=r.get("depth_emphasis", ["applied"]),
                            reason=f"Approved {age_days}d ago — SotA may be stale",
                        ))
                except (ValueError, TypeError):
                    pass

    # Sort by priority, then alphabetically within same priority
    actions.sort(key=lambda a: (a.priority, a.topic))
    return actions


def _llm_editorial_analysis(
    health: WikiHealth,
    audit: AuditReport,
    actions: list[SupervisorAction],
    obs=None,
    verbose: bool = False,
) -> str:
    """Use RESEARCH_MODEL to write a 200-word editorial analysis of wiki state."""
    try:
        llm = get_llm("research", temperature=0.1)

        top_actions = "\n".join(
            f"  {i+1}. [{a.priority}] {a.action} {a.topic} ({a.track}) — {a.reason}"
            for i, a in enumerate(actions[:10])
        )

        # Add observer error signals if available
        observer_block = ""
        if obs is not None:
            e = obs.error_signals
            b = obs.budget
            remaining = f"${b.remaining_usd:.2f}" if b.remaining_usd is not None else "unlimited"
            observer_block = f"""
Control system signals:
- Coverage deficit: {e.get('coverage_deficit', 0):.1%} ({obs.coverage_pct:.1%} generated)
- Quality deficit: {e.get('quality_deficit', 0):.2f} (avg confidence {obs.avg_confidence:.2f})
- Flagged pages: {e.get('flagged_pages', 0):.0f}
- Budget: {remaining} remaining, mode={b.mode}
- Quality trend (last 10): avg conf {obs.quality_trend.avg_confidence:.2f}, "
  first-pass approval {obs.quality_trend.approval_rate:.0%}, "
  delta {obs.quality_trend.confidence_delta:+.3f}"""

        prompt = f"""You are the editorial supervisor of the Frontier Wiki — an AI/ML knowledge base.

Current wiki state:
- Total pages: {health.total_pages} ({health.stub_pages} stubs, {health.generated_pages} generated)
- Approved (conf ≥ threshold): {health.approved_pages}/{health.generated_pages}
- Flagged: {health.flagged_pages}
- Avg reviewer confidence: {health.avg_confidence:.2f}
- Tracks with content: {health.tracks_covered}/{health.tracks_total}
- Pages with MVB: {health.pages_with_mvb}
- Critical audit issues: {len(audit.critical)}
- Warnings: {len(audit.warnings)}{observer_block}

Top prioritised actions:
{top_actions or "  No actions needed."}

Write a concise editorial assessment (150-200 words) as the supervising editor:
- What is the current state of the wiki?
- What are the most important gaps or error signals?
- What should the team focus on in the next sprint?
- Are there any systemic issues (e.g., reviewer too strict, prose quality drift, budget constraint)?

Write as an editor, not as a system report. One flowing paragraph, no headers.
"""
        response = llm.invoke([HumanMessage(content=prompt)])
        return response.content.strip()
    except Exception as e:
        if verbose:
            print(f"[supervisor] LLM analysis failed: {e}")
        return (
            f"Wiki has {health.total_pages} pages ({health.stub_pages} stubs). "
            f"{health.flagged_pages} pages need revision. "
            f"Focus: generate stubs and improve flagged pages."
        )


def _update_sprint(actions: list[SupervisorAction], sprints_path: Path, now: datetime) -> None:
    """Rewrite agents/sprints/current.md with prioritised work items."""
    sprints_path.mkdir(parents=True, exist_ok=True)
    sprint_path = sprints_path / "current.md"

    lines = [
        f"# Sprint: {now.strftime('%Y-%m-%d')} (supervisor-generated)",
        "",
        "> Auto-generated by the supervising agent. Edit to adjust priorities.",
        "> Format: topic | track | page-type | depth [| arc:id pos:N prev:X next:Y]",
        "> Checked items are archived automatically on the next scheduler run.",
        "",
    ]

    # Group by priority
    for priority, label in [(1, "Critical Fixes"), (2, "High Priority"), (3, "New Content"), (4, "Improvements")]:
        group = [a for a in actions if a.priority == priority]
        if not group:
            continue
        lines.append(f"## {label}")
        for a in group:
            depth = " ".join(a.depth_emphasis)
            arc_part = f" | arc:{a.arc} pos:{a.arc_position}" if a.arc else ""
            lines.append(
                f"- [ ] {a.topic} | {a.track} | {a.page_type} | {depth}{arc_part}"
                f"  <!-- {a.reason} -->"
            )
        lines.append("")

    sprint_path.write_text("\n".join(lines), encoding="utf-8")


def _write_report(report: SupervisorReport, docs_path: Path) -> None:
    """Write the supervisor report to docs/system/supervisor.md."""
    system_dir = docs_path / "system"
    system_dir.mkdir(parents=True, exist_ok=True)
    (system_dir / "supervisor.md").write_text(report.to_markdown(), encoding="utf-8")
