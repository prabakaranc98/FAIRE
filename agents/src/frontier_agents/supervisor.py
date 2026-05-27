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
from .tools import display_ts


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
    tracks_total: int = 10
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
        generated_at=display_ts(),
        health=health,
        audit=audit,
        actions=actions,
        llm_analysis=llm_analysis,
    )

    # 4. QUEUE — write sprint if not dry run
    # Cap: take all priority-1/2 (critical), then fill up to 50 total with priority-3+
    # This ensures all stubs get queued in a full-regeneration pass while keeping
    # critical fixes always at the top.
    sprint_cap = int(os.environ.get("SPRINT_CAP", "50"))
    if not dry_run and actions:
        _update_sprint(actions[:sprint_cap], sprints_path, now)
        if verbose:
            print(f"[supervisor] Updated sprint with {min(len(actions), sprint_cap)} items")

    # 4.7. ARC AUTO-SPIN — runs AFTER the queue rewrite so the arc lines
    # survive. For each qualifying track, append a canonical arc-candidate.
    # The helper itself enforces guardrails (>=3 substantive concepts, >=4
    # named steps on disk, <2 active arcs, not already queued).
    budget_mode = obs.budget.mode if obs is not None else "full"
    if not dry_run and budget_mode != "paused":
        try:
            from .retrospective import _apply_arc_proposal
            core_dir = docs_path / "curriculum" / "core"
            if core_dir.exists():
                arc_candidates = _suggest_track_arcs(core_dir)
                spun = 0
                for arc_params in arc_candidates:
                    result = _apply_arc_proposal({}, arc_params, docs_path)
                    if result.startswith("applied"):
                        spun += 1
                if verbose and spun:
                    print(f"[supervisor] Arc auto-spin queued {spun} arcs")
        except Exception as e:
            if verbose:
                print(f"[supervisor] Arc auto-spin failed: {e}")

    # 5. REPORT — write to docs/system/supervisor.md
    _write_report(report, docs_path)
    if verbose:
        print(f"[supervisor] Report written to docs/system/supervisor.md")

    # 6. ARC PROPOSAL phase — run only if curriculum is ready, budget allows,
    #    and fewer than 2 arcs are already active (see arc-selection skill).
    if not dry_run:
        try:
            arc_summary = maybe_propose_arcs(obs, audit, docs_path, verbose=verbose)
            if verbose and arc_summary.get("ran"):
                print(
                    f"[supervisor] arc-proposals → {arc_summary.get('wrote_to')} "
                    f"({arc_summary.get('slots_open')} slot(s) open)"
                )
            elif verbose:
                print(f"[supervisor] arc-proposal skipped — {arc_summary.get('reason')}")
        except Exception as e:
            if verbose:
                print(f"[supervisor] arc-proposal phase failed (non-fatal): {e}")

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
                # v2 path-aware track derivation
                parts = p.parent.parts
                if "core" in parts:
                    ci = parts.index("core")
                    track = parts[ci + 1] if ci + 1 < len(parts) else p.parent.name
                else:
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

    # Priority 1: Flagged AND errored pages (generated but didn't land cleanly).
    # Errored pages were previously orphaned — they now get retried with the
    # latest prompts and code.
    for topic, r in latest_runs.items():
        s = r.get("status")
        if s in ("flagged", "error"):
            conf = r.get("confidence", 0)
            reason = (
                f"Reviewer flagged (conf={conf:.2f}) — needs targeted revision"
                if s == "flagged"
                else f"Previous attempt errored (conf={conf:.2f}) — retry with current prompts"
            )
            actions.append(SupervisorAction(
                priority=1,
                action="improve",
                topic=topic,
                track=r.get("track", ""),
                page_type=r.get("page_type", "core-concept"),
                depth_emphasis=r.get("depth_emphasis", ["applied"]),
                reason=reason,
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

    # (Arc auto-spin moved to run_supervisor() — it runs AFTER _update_sprint
    # writes the queue, otherwise the appended arc lines get wiped on rewrite.)

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
                    # v2 layout: docs/curriculum/core/<track>/concepts/<slug>.md
                    # The parent is "concepts" (or authors/arcs/builds); the
                    # grandparent is the track. Walk up until we find a track-like name.
                    parts = page.parent.parts
                    if "core" in parts:
                        ci = parts.index("core")
                        track = parts[ci + 1] if ci + 1 < len(parts) else page.parent.name
                    else:
                        # Legacy layout fallback: track is parent
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


# ── Arc proposal phase (Unit C) ───────────────────────────────────────────────

ARC_PROPOSE_COVERAGE_THRESHOLD = float(os.environ.get("ARC_PROPOSE_COVERAGE_THRESHOLD", "0.60"))


def _parse_arc_roadmap(docs_path: Path) -> list[dict]:
    """Parse docs/system/arc-roadmap.md and return only `ready` arcs.

    The roadmap is the human-curated, frontier-grounded design doc that
    replaces hardcoded canonical arcs. Each arc has a named frontier
    destination and a diagonal 5-step spine. Only `ready` arcs are returned
    (status `needs-seeds` arcs wait for their missing concepts to land).
    """
    import re as _re
    roadmap = docs_path / "system" / "arc-roadmap.md"
    if not roadmap.exists():
        return []
    text = roadmap.read_text(encoding="utf-8")
    out: list[dict] = []
    # Split by track sections (## NN-track-...)
    track_pattern = _re.compile(r"^## (\d{2}-[a-z\-]+)", _re.MULTILINE)
    matches = list(track_pattern.finditer(text))
    for i, m in enumerate(matches):
        track = m.group(1).split(" ")[0]
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end]
        arc_pattern = _re.compile(
            r"^### \d+\.\s+`([a-z0-9\-]+)`\s+—\s+(ready|needs-seeds)\b(.*?)(?=^### |\Z)",
            _re.MULTILINE | _re.DOTALL,
        )
        for am in arc_pattern.finditer(body):
            if am.group(2) != "ready":
                continue
            arc_id = am.group(1)
            arc_body = am.group(3)
            dest_match = _re.search(
                r"\*\*Destination:\*\*\s*\*?(.+?)\*?\s*$", arc_body, _re.MULTILINE
            )
            destination = dest_match.group(1).strip().rstrip(".") if dest_match else ""
            spine_match = _re.search(
                r"\*\*Diagonal spine:\*\*\s*(.+?)$", arc_body, _re.MULTILINE
            )
            spine_raw = spine_match.group(1).strip() if spine_match else ""
            steps_raw = _re.findall(r"`([a-z0-9\-]+)`", spine_raw)
            if len(steps_raw) < 4:
                continue
            out.append({
                "arc_id": arc_id,
                "track": track,
                "dest": destination,
                "steps": steps_raw[:5],
            })
    return out


def _suggest_track_arcs(core_dir: Path) -> list[dict]:
    """Read `ready` arcs from the roadmap. Falls back to a tiny hardcoded set
    if the roadmap is missing.

    Replaces the earlier `<track>-foundations` placeholder approach (which
    produced vertical reading lists, not arcs). See docs/system/arc-roadmap.md
    for the design and frontier evidence behind each ready arc.
    """
    docs_path = core_dir.parent.parent
    roadmap_arcs = _parse_arc_roadmap(docs_path)
    if roadmap_arcs:
        return roadmap_arcs
    # Fallback if roadmap missing — minimal safe set
    return [
        {"arc_id": "generative-stack",
         "track": "02-generative-modeling",
         "dest": "Five trained generative models compared head-to-head",
         "steps": ["diffusion-models", "score-matching", "latent-diffusion-models",
                   "flow-matching", "consistency-models"]},
    ]


def _suggest_track_arcs_LEGACY(core_dir: Path) -> list[dict]:
    """LEGACY — replaced by roadmap parser above. Kept for reference.

    For each track, build at most one arc-proposal candidate from existing concepts.

    Returns the action_params payloads expected by `retrospective._apply_arc_proposal`.
    The helper there enforces guardrails (>=3 substantive concepts, >=4 of the named
    steps on disk, <2 active arcs, not already queued). So we can speculatively
    propose; rejections are silent.

    Naming heuristic: pick the first 5 substantive concept slugs from the track,
    sorted to give a stable arc_id. The arc title is "<track>-foundations" if no
    canonical arc exists. The retro will propose better-named arcs over time;
    this is a cheap fallback to seed each track with something.
    """
    from .retrospective import _is_substantive_concept

    out: list[dict] = []
    # Canonical arc seeds — preferred over the fallback naming heuristic
    # because they line up with concept slugs that already exist in the corpus.
    canonical = {
        "02-generative-modeling": ("generative-stack",
            ["diffusion-models", "score-matching", "latent-diffusion-models",
             "flow-matching", "consistency-models"],
            "5 trained generative models with comparable FID, ending in a distilled consistency model"),
        "04-neural-networks-deep-learning": ("training-fundamentals",
            ["backpropagation", "gradient-descent", "adaptive-optimizers",
             "regularization", "batch-normalization"],
            "a trained-from-scratch convolutional network with documented loss curves, normalization, and a learned schedule"),
        "09-algorithms-systems-for-ai": ("serve-an-llm-efficiently",
            ["flash-attention", "kv-cache", "kv-cache-management",
             "quantization", "llm-inference"],
            "a quantized 7B model served behind an endpoint with measured p95 latency under 100ms"),
    }

    for track_dir in sorted(core_dir.iterdir()):
        if not track_dir.is_dir():
            continue
        track_name = track_dir.name
        if track_name in canonical:
            arc_id, steps, dest = canonical[track_name]
            out.append({"arc_id": arc_id, "track": track_name, "dest": dest, "steps": steps})
            continue
        # Generic fallback for non-canonical tracks
        concepts = track_dir / "concepts"
        if not concepts.exists():
            continue
        slugs = sorted(p.stem for p in concepts.glob("*.md")
                       if p.name != "index.md" and _is_substantive_concept(p))
        if len(slugs) < 5:
            continue
        out.append({
            "arc_id": f"{track_name}-foundations",
            "track": track_name,
            "dest": f"a working build that ties together the foundational concepts of {track_name}",
            "steps": slugs[:5],
        })
    return out


def _count_active_arcs(docs_path: Path) -> int:
    """An arc is 'active' if its arc-index exists AND at least one step is missing.

    Counts arcs under docs/arcs/{arc}/ where index.md exists but the total step
    files declared in arcs.json (or detected via step-NN-*.md pattern) are not
    all present yet.
    """
    arcs_dir = docs_path / "arcs"
    if not arcs_dir.exists():
        return 0
    active = 0
    for arc_dir in arcs_dir.iterdir():
        if not arc_dir.is_dir():
            continue
        index = arc_dir / "index.md"
        if not index.exists():
            continue
        # Heuristic: scan the arc index for "Step N — ..." references and
        # see if all corresponding step files exist.
        text = index.read_text(encoding="utf-8", errors="ignore")
        import re as _re
        referenced = set(_re.findall(r"step-(\d{2})-[a-z0-9-]+\.md", text))
        present = {f.name.split("-")[1] for f in arc_dir.glob("step-*.md") if f.is_file()}
        if referenced and present < referenced:
            active += 1
    return active


def _arc_propose_preconditions_met(
    obs, audit, active_arcs: int, verbose: bool = False
) -> tuple[bool, str]:
    """Check the three preconditions for arc proposal (see arc-selection skill)."""
    if obs is None:
        return False, "no observer snapshot — cannot evaluate readiness"

    # 1. Curriculum coverage on canonical tracks ≥ threshold
    canonical_tracks = [f"{i:02d}-" for i in range(1, 11)]
    # Find the actual track keys that start with these prefixes
    canonical_coverage = []
    for tm in obs.track_metrics.values():
        if any(tm.track.startswith(p) for p in canonical_tracks):
            canonical_coverage.append(tm.coverage_pct)
    if not canonical_coverage:
        return False, "no canonical-track metrics available"
    avg_canonical = sum(canonical_coverage) / len(canonical_coverage)
    if avg_canonical < ARC_PROPOSE_COVERAGE_THRESHOLD:
        return False, (
            f"canonical-track coverage {avg_canonical:.1%} < "
            f"threshold {ARC_PROPOSE_COVERAGE_THRESHOLD:.0%}"
        )

    # 2. Budget mode must be 'full'
    if obs.budget.mode != "full":
        return False, f"budget mode = {obs.budget.mode} (need 'full' to propose arcs)"

    # 3. Active arcs must be < 2 (the canonical 2-arcs-at-a-time rule)
    if active_arcs >= 2:
        return False, f"{active_arcs} arcs already active (cap = 2)"

    return True, (
        f"all 3 preconditions met: canonical-coverage {avg_canonical:.1%}, "
        f"budget=full, active_arcs={active_arcs}"
    )


def maybe_propose_arcs(
    obs,
    audit,
    docs_path: Path,
    verbose: bool = False,
    forced_seed: str = "",
    forced_track: str = "",
) -> dict:
    """Run the arc-proposal phase if preconditions are met.

    Two invocation paths:
      - Autonomous (default): all three preconditions checked (canonical-track
        coverage ≥ threshold, budget mode = full, active_arcs < 2).
      - Human-in-the-loop (forced_seed != ""): bypasses the coverage check,
        because the user has explicitly named a seed they want explored.
        Still respects budget mode (refuses if 'paused') and 2-active-arcs cap.

    Reads `arc-exploration.md` + `arc-selection.md` + `arc-anatomy.md` +
    `critic-editor.md` skills to drive the proposal; writes candidate arcs
    to docs/system/arc-proposals.md (does NOT queue — human picks).

    Returns a summary dict.
    """
    active_arcs = _count_active_arcs(docs_path)

    if forced_seed:
        # CLI explore path — bypass coverage gate, keep budget + active_arcs gates.
        if obs is not None and obs.budget.mode == "paused":
            ok, reason = False, "budget mode = paused (refusing to explore)"
        elif active_arcs >= 2:
            ok, reason = False, f"{active_arcs} arcs already active (cap = 2)"
        else:
            ok, reason = True, f"CLI explore mode for seed '{forced_seed}'"
    else:
        ok, reason = _arc_propose_preconditions_met(obs, audit, active_arcs, verbose)

    summary = {
        "ran": False,
        "preconditions_met": ok,
        "reason": reason,
        "active_arcs": active_arcs,
        "slots_open": max(0, 2 - active_arcs),
    }

    if not ok:
        if verbose:
            print(f"[supervisor] arc-proposal skipped — {reason}")
        # Write a small marker so docs/system/arc-proposals.md reflects current state
        try:
            (docs_path / "system" / "arc-proposals.md").write_text(
                "---\n"
                "title: Arc proposals\n"
                "description: When the supervisor judges the wiki ready for new arcs, candidates appear here.\n"
                "---\n\n"
                "# Arc proposals\n\n"
                f"> Current cycle: **no proposals**. Reason: {reason}\n\n"
                "Arcs are derived from the curriculum, bounded by budget, and capped at 2 active "
                "at a time. The supervisor proposes new arcs only when all three preconditions hold "
                "(see `agents/skills/arc-selection.md` and `agents/skills/critic-editor.md`).\n",
                encoding="utf-8",
            )
        except Exception:
            pass
        return summary

    # Preconditions met — invoke the LLM with arc-selection skill to draft proposals.
    try:
        from .skills import load_skills
        skills = load_skills()
        sel_skill = next((s for s in skills if s.name == "arc-selection"), None)
        anatomy = next((s for s in skills if s.name == "arc-anatomy"), None)
        editor = next((s for s in skills if s.name == "critic-editor"), None)
        exploration = next((s for s in skills if s.name == "arc-exploration"), None)
        if not (sel_skill and anatomy and editor):
            summary["reason"] = "missing one of arc-selection / arc-anatomy / critic-editor skill"
            return summary

        # Build the budget context
        remaining = obs.budget.remaining_usd or 0.0
        arc_budget = remaining * 0.5

        # Build curriculum context: list of approved curriculum slugs by track
        curriculum_context = []
        for track, tm in sorted(obs.track_metrics.items()):
            if tm.generated > 0:
                curriculum_context.append(
                    f"  {track}: {tm.generated} pages generated, avg_conf={tm.avg_confidence:.2f}"
                )

        seed_block = ""
        if forced_seed:
            seed_block = f"""
EXPLICIT SEED (CLI explore mode)
The user named the seed: "{forced_seed}"{" (track: " + forced_track + ")" if forced_track else ""}
Apply the arc-exploration playbook to THIS seed specifically. Survey via your
knowledge of the field's branching directions (Gaussian-Splatting vs JEPA vs
Diffusion-based vs PAN-based, for the world-models example). Map against the
diagonal pattern. Pick + outline the top 1–2 candidate arcs that branch
from this seed. Do NOT propose arcs from unrelated tracks — stay on-seed.

The arc-exploration skill body (read this carefully):
---
{exploration.body if exploration else "(arc-exploration skill missing — improvise from arc-anatomy)"}
---
"""
        prompt = f"""You are the Frontier Wiki supervisor running the ARC PROPOSAL phase.

Reading these skills as context:

---
{sel_skill.body}
---
{anatomy.body}
---
{editor.body}
---
{seed_block}
CURRENT STATE
- Mode: {"CLI explore (seed-forced)" if forced_seed else "autonomous (coverage-gated)"}
- Remaining budget: ${remaining:.2f} (arc budget = ${arc_budget:.2f})
- Slots open for new active arcs: {summary['slots_open']}
- Curriculum coverage by track:
{chr(10).join(curriculum_context) if curriculum_context else "  (no tracks have generated content yet)"}

YOUR JOB
{"1. Apply the arc-exploration playbook to the seed above. Survey ≥ 3 branches before scoring." if forced_seed else "1. Scan the curriculum-track metrics above. Identify 3–6 candidate arcs that would compound the existing curriculum into a frontier capability."}
2. Score each by EV/$ per the arc-selection skill.
3. Apply the critic-editor judgment: veto / reshape / approve each one.
4. Propose only as many arcs as fit in ${arc_budget:.2f} AND at most
   {summary['slots_open']} new arcs.

Return a markdown document with this structure:

---
title: Arc proposals
generated_at: [today]
remaining_budget: ${remaining:.2f}
slots_open: {summary['slots_open']}
---

# Arc proposals

[One-paragraph summary of which arcs you propose and why.]

## 1. [Arc Title] — EV/$ = X.X — verdict: approve | reshape | veto
**Destination:** ...
**Steps:** N · **Cost:** $X.XX · **Impact:** X/10
**Prereqs in curriculum:** [list with status]
**Persona span:** N personas (list them)
**Seminal anchors:** [papers]
**Outline:** [step list with mvb_persona for each step]
**Editor verdict:** approve | reshape | veto
**If reshape/veto:** [reasons + specific reshape suggestions]
[If approve:] **Approval note:** [one paragraph on why this arc is worth materializing]

## 2. ...

## Deferred branches (not proposed this cycle)

[Optional. If the explorer surveyed branches that did NOT make the cut,
list them here as bullets — NOT numbered as proposals. One line per
branch with the reason it was dropped (insufficient prereqs / vertical
shape / persona span too narrow). This is documentation, not a proposal.]

OUTPUT RULES
- Output raw markdown starting with `---` frontmatter. Do NOT wrap the response
  in a code fence (no ```markdown, no ```yaml, no opening ``` of any kind).
- Numbered `## 1.`, `## 2.` sections are ONLY for actual arc proposals.
- Use `## Deferred branches` (no number) for branches you considered but did
  not propose.

SELF-CRITIQUE BEFORE EMITTING (see reasoning-scaffolding.md)
After you draft each candidate, invoke the critic-editor lens against your
own proposal. Specifically, for each arc you're about to label `approve`:
  - Would the critic-editor veto it for seed-readiness? (Are ≥ 50% of prereqs
    solid curriculum pages, not stubs?)
  - Does the compounding chain hold? (Step N's prev_artifact = step N-1's
    artifact, verbatim, for every step?)
  - Is the persona span ≥ 3? (Otherwise the arc is too narrow.)
  - Does the diagonal shape hold? (Col-1 ≠ col-4 of the diagonal pattern.)

If any answer is "no," downgrade the verdict from `approve` to `reshape` (or
`veto`) yourself and put the reason in the section. Don't emit `approve` for
arcs you can't defend against these objections — surface the issue so the
human sees it.
"""
        llm = get_llm("research", temperature=0.2)
        try:
            response = llm.invoke([HumanMessage(content=prompt)])
            proposal_md = response.content
        except Exception as exc:
            summary["reason"] = f"LLM call failed: {exc}"
            return summary

        # Strip leading/trailing code fences if the model wraps its output anyway.
        from .nodes import _sanitize_draft
        proposal_md = _sanitize_draft(proposal_md)

        (docs_path / "system" / "arc-proposals.md").write_text(
            proposal_md, encoding="utf-8"
        )
        summary["ran"] = True
        summary["wrote_to"] = "docs/system/arc-proposals.md"
        if verbose:
            print(f"[supervisor] wrote arc proposals to {summary['wrote_to']}")
    except Exception as exc:
        summary["reason"] = f"proposal failed: {exc}"

    return summary


# ── Persona-update loop (closes the long-horizon voice loop) ──────────────────

def analyze_critic_patterns(
    runs_path: Path,
    last_n: int = 20,
) -> dict[str, dict]:
    """Aggregate critic_panel scores per track over the last N runs.

    For each track that has any critic_panel data, computes:
      - per-critic average score
      - the critic dimension that scored lowest
      - the most common issue strings across runs (top 3)

    Returns: {track: {dimension_means: {...}, weakest: str, top_issues: [str]}}
    """
    if not runs_path.exists():
        return {}

    runs: list[dict] = []
    with runs_path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    runs.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    runs = runs[-last_n * 5:]  # take a wider window then filter

    by_track: dict[str, list[dict]] = {}
    for r in runs:
        if not r.get("critic_panel"):
            continue
        t = r.get("track", "")
        if t:
            by_track.setdefault(t, []).append(r)

    results: dict[str, dict] = {}
    for track, rs in by_track.items():
        rs = rs[-last_n:]
        # Aggregate per-critic mean score across runs
        dim_scores: dict[str, list[float]] = {}
        dim_issues: dict[str, list[str]] = {}
        for r in rs:
            panel = r.get("critic_panel", {})
            for name, entry in panel.items():
                if not isinstance(entry, dict):
                    continue
                score = entry.get("score", 0.7)
                dim_scores.setdefault(name, []).append(float(score))
                for iss in entry.get("issues", []):
                    dim_issues.setdefault(name, []).append(str(iss))

        if not dim_scores:
            continue

        means = {n: sum(v) / len(v) for n, v in dim_scores.items()}
        weakest = min(means.items(), key=lambda kv: kv[1])
        from collections import Counter
        top = Counter(dim_issues.get(weakest[0], []))
        top_issues = [s for s, _ in top.most_common(3)]

        results[track] = {
            "n_runs": len(rs),
            "dimension_means": {n: round(m, 3) for n, m in means.items()},
            "weakest": weakest[0],
            "weakest_score": round(weakest[1], 3),
            "top_issues": top_issues,
        }
    return results


def propose_persona_updates(
    runs_path: Path,
    personas_dir: Path,
    docs_path: Path,
    weakness_threshold: float = 0.70,
    last_n: int = 20,
    verbose: bool = False,
) -> dict:
    """Generate persona-update proposals for tracks whose critic patterns fall below threshold.

    Writes one consolidated report to docs/system/persona-proposals.md. Does NOT
    auto-edit any persona YAML — the human reviews and applies.
    """
    patterns = analyze_critic_patterns(runs_path, last_n=last_n)
    if not patterns:
        return {"flagged_tracks": [], "reason": "no critic_panel data yet"}

    flagged = [(t, p) for t, p in patterns.items() if p["weakest_score"] < weakness_threshold]
    if not flagged:
        return {"flagged_tracks": [], "reason": "all tracks above weakness threshold"}

    proposals: list[str] = [
        "---",
        "title: Persona update proposals",
        f"description: Critic patterns suggesting persona tweaks (last {last_n} runs per track). Human reviews; no auto-apply.",
        "---",
        "",
        "# Persona update proposals",
        "",
        f"Each section below is one track whose worst-scoring critic dimension fell below "
        f"the threshold ({weakness_threshold:.2f}). The proposed YAML diff is a suggestion — "
        "review against `agents/src/frontier_agents/personas/{track}.yaml` and apply only "
        "if it matches your editorial intent.",
        "",
    ]

    llm = get_llm("research", temperature=0.1)
    for track, p in flagged:
        persona_path = personas_dir / f"{track}.yaml"
        current_yaml = persona_path.read_text(encoding="utf-8") if persona_path.exists() else "(no persona file)"

        prompt = f"""You are reviewing failure patterns in the wiki's writer agent for one track and proposing a YAML update to its persona file.

TRACK: {track}
RUNS ANALYZED: {p['n_runs']}
WEAKEST CRITIC DIMENSION: {p['weakest']} (avg score {p['weakest_score']})
TOP ISSUES ON THAT DIMENSION:
{chr(10).join(f'  - {iss}' for iss in p['top_issues']) if p['top_issues'] else '  (no issue strings recorded)'}

CURRENT PERSONA YAML:
```yaml
{current_yaml}
```

YOUR JOB
Propose ONE concrete persona YAML diff that addresses the recurring weakness.
Keep the diff small — 1-3 added or modified keys. Concrete additions only;
don't reword existing fields unless they're actively misleading.

Examples of good diffs (illustrative — don't copy these literally):
- If critic-wiki-voice keeps flagging marketing language: add
  `voice_guard: "avoid marketing adjectives — strip 'powerful', 'revolutionary', 'state-of-the-art'"`
- If critic-coverage keeps flagging missing mathematical foundations: bump
  `depth_focus` to mention "include derivations for every named theorem"
- If critic-build-nudge keeps flagging phantom HF IDs: tighten `mvb_focus`
  with 3-5 specific verified model IDs the writer should reach for first

Output format: just a markdown section starting with `## {track}` then
showing the proposed diff in a yaml code block + one paragraph explanation
of what pattern this addresses.
"""
        try:
            response = llm.invoke([HumanMessage(content=prompt)])
            proposals.append(response.content.strip())
            proposals.append("")
            proposals.append("---")
            proposals.append("")
        except Exception as exc:
            if verbose:
                print(f"[supervisor] persona-update LLM failed for {track}: {exc}")
            proposals.append(f"## {track}\n\n_LLM call failed: {exc}_\n\n---\n")

    out_path = docs_path / "system" / "persona-proposals.md"
    out_path.write_text("\n".join(proposals), encoding="utf-8")

    return {
        "flagged_tracks": [t for t, _ in flagged],
        "wrote_to": str(out_path),
        "patterns": {t: p for t, p in flagged},
    }
