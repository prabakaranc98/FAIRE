"""Between-cycle backlog agent — closes the FAIRE autonomy loop.

After each sprint cycle, this runs once and:
  - Phase A (deterministic): aggregates patterns from the cycle's runs.jsonl tail
    and the curriculum on disk — per-critic weakness, heading drift, citation
    patterns, unresolved [[wikilinks]] (future stub seeds), under-threshold
    landings.
  - Phase B (LLM, one call): turns the aggregated signals into 3-6 concrete,
    ranked proposals with rationale + evidence + risk class.
  - Phase C (apply + log): auto-applies SAFE proposals (stub seeds, bounded
    sprint-queue bumps) and writes everything timestamped to
    docs/system/backlog.md for human review.

Called from scheduler.full_cycle_job after write_changelog_entry. Cost per
cycle: ~$0.01 (one RESEARCH_MODEL call). Reuses analyze_critic_patterns from
supervisor.py.
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from .llm import get_llm
from .nodes import _coerce_text


# ── Structured output schema (forces clean JSON from the LLM) ──────────────

class _NewToAddItem(BaseModel):
    action: str = Field(description="One-sentence description of the proposed action")
    evidence: str = Field(default="", description="Which signal motivates this")
    evidence_refs: list[str] = Field(default_factory=list, description="Topic/critic refs")
    risk_class: str = Field(default="moderate", description="safe | moderate | risky")
    auto_apply: bool = Field(default=False)
    action_type: str = Field(default="other", description="stub-seed | author-page-seed | arc-proposal | queue-priority-bump | trim-knob-adjust | other")
    action_params: dict = Field(default_factory=dict, description="Action-type-specific params (e.g. {slug, track} for stub-seed)")


class _RetroOutput(BaseModel):
    went_well: list[str] = Field(default_factory=list, description="What to keep doing")
    went_wrong: list[str] = Field(default_factory=list, description="What failed or regressed")
    needs_depth: list[str] = Field(default_factory=list, description="Where coverage is too shallow")
    new_to_add: list[_NewToAddItem] = Field(default_factory=list, description="Concrete additions for next cycle")
    process_improvements: list[str] = Field(default_factory=list, description="Pipeline / prompt / critic tweaks")

# v1 forbidden headings — same blacklist the writer prompt rejects
_OLD_HEADINGS = [
    "## What it is", "## Why it matters", "## Core concepts",
    "## Mathematical foundations", "## Key algorithms", "## Essential reading",
    "## Seminal papers", "## Current SotA", "## What's happening now",
    "## In production", "## Connected topics", "## Further reading",
    "## What comes next", "## Open questions",
]

_WIKILINK_RE = re.compile(r"\[\[([a-z0-9][a-z0-9\-]*)\]\]")
_ARXIV_RE = re.compile(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5}|[a-z\-]+/\d{7})", re.IGNORECASE)


# ── Phase A: deterministic aggregators ──────────────────────────────────────

@dataclass
class CycleSignals:
    """Structured signals snapshot — what Phase B's proposer reads."""
    cycle_window: dict = field(default_factory=dict)            # n_runs, approved, errored, avg_conf
    per_track_health: dict = field(default_factory=dict)        # {track: {n, avg_conf, weakest_critic}}
    heading_drift: dict = field(default_factory=dict)           # {old_heading: occurrence_count}
    under_threshold_landings: list = field(default_factory=list) # pages with conf<0.7 committed via never-throw-away
    unresolved_wikilinks: list = field(default_factory=list)    # candidates for stub seeding
    orphaned_run_records: list = field(default_factory=list)    # runs.jsonl says approved but file missing
    citation_health: dict = field(default_factory=dict)         # {n_unique_urls, n_arxiv, n_per_page_avg}
    recurring_critic_issues: list = field(default_factory=list) # top 5 most common critic issue strings


def aggregate_cycle_signals(
    runs_path: Path,
    docs_path: Path,
    last_n: int = 30,
) -> CycleSignals:
    """Phase A — pure-Python pattern detection. No LLM calls."""
    sig = CycleSignals()

    # Read the runs.jsonl tail
    runs: list[dict] = []
    if runs_path.exists():
        with runs_path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    runs.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    recent = runs[-last_n:] if len(runs) > last_n else runs
    if not recent:
        return sig

    # 1. Cycle window summary
    approved = sum(1 for r in recent if r.get("status") == "approved")
    errored = sum(1 for r in recent if r.get("status") == "error")
    confs = [r.get("confidence", 0.0) for r in recent if r.get("confidence", 0.0) > 0]
    sig.cycle_window = {
        "n_runs": len(recent),
        "approved": approved,
        "errored": errored,
        "avg_confidence": round(sum(confs) / len(confs), 3) if confs else 0.0,
        "first_try_approval_rate": round(
            sum(1 for r in recent if r.get("status") == "approved" and r.get("revision_count", 0) == 0)
            / max(len(recent), 1),
            2,
        ),
    }

    # 2. Per-track health (latest run per topic per track)
    per_track: dict[str, list[dict]] = {}
    latest_by_topic: dict[str, dict] = {}
    for r in recent:
        t = r.get("topic", "")
        if t:
            latest_by_topic[t] = r
    for r in latest_by_topic.values():
        tr = r.get("track", "?")
        per_track.setdefault(tr, []).append(r)

    for tr, rs in per_track.items():
        cs = [r.get("confidence", 0.0) for r in rs if r.get("confidence", 0.0) > 0]
        # Critic panel weakest dim per track (average over track's runs)
        critic_sums: dict[str, list[float]] = {}
        for r in rs:
            panel = r.get("critic_panel") or {}
            for name, payload in panel.items():
                if isinstance(payload, dict) and "score" in payload:
                    critic_sums.setdefault(name, []).append(payload["score"])
        weakest = min(
            ((n, sum(v) / len(v)) for n, v in critic_sums.items() if v),
            key=lambda x: x[1],
            default=("none", 1.0),
        )
        sig.per_track_health[tr] = {
            "n_pages": len(rs),
            "avg_confidence": round(sum(cs) / len(cs), 3) if cs else 0.0,
            "weakest_critic": weakest[0],
            "weakest_critic_avg": round(weakest[1], 3),
        }

    # 3. Recurring critic issues (top 5 issue fingerprints)
    issue_counter: Counter[str] = Counter()
    for r in recent:
        panel = r.get("critic_panel") or {}
        for name, payload in panel.items():
            if isinstance(payload, dict):
                for issue in (payload.get("issues") or [])[:3]:
                    # 80-char fingerprint
                    key = f"{name}: {issue[:80]}"
                    issue_counter[key] += 1
    sig.recurring_critic_issues = [
        {"issue": k, "count": v}
        for k, v in issue_counter.most_common(5)
        if v >= 2
    ]

    # 4. Heading drift detection — scan committed drafts
    drift: Counter[str] = Counter()
    for r in recent:
        if not r.get("committed"):
            continue
        path = r.get("output_path", "")
        if not path:
            continue
        full = _resolve_doc_path(path, docs_path)
        if not full or not full.exists():
            continue
        try:
            content = full.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for h in _OLD_HEADINGS:
            if h in content:
                drift[h] += 1
    sig.heading_drift = dict(drift.most_common())

    # 5. Under-threshold landings (conf < 0.7 but committed via never-throw-away)
    for r in recent:
        if r.get("committed") and r.get("confidence", 0.0) < 0.7:
            sig.under_threshold_landings.append({
                "topic": r.get("topic"),
                "track": r.get("track"),
                "confidence": r.get("confidence"),
                "revision_count": r.get("revision_count"),
            })

    # 6. Unresolved wikilinks — scan ALL concept pages, find [[slugs]] without backing files
    existing_slugs: set[str] = set()
    curriculum = docs_path / "curriculum" / "core"
    if curriculum.exists():
        for p in curriculum.rglob("*.md"):
            if p.name == "index.md":
                continue
            existing_slugs.add(p.stem)

        unresolved: Counter[str] = Counter()
        unresolved_sources: dict[str, list[str]] = {}
        for p in curriculum.rglob("*.md"):
            if p.name == "index.md":
                continue
            try:
                content = p.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            parts = p.parent.parts
            try:
                track = parts[parts.index("core") + 1]
            except (ValueError, IndexError):
                track = "?"
            for m in _WIKILINK_RE.finditer(content):
                target = m.group(1)
                if target not in existing_slugs and target != p.stem:
                    unresolved[target] += 1
                    unresolved_sources.setdefault(target, []).append(f"{track}/{p.stem}")
        # Candidates referenced >=2 times across the wiki
        sig.unresolved_wikilinks = [
            {
                "slug": slug,
                "reference_count": count,
                "referenced_by": unresolved_sources.get(slug, [])[:5],
            }
            for slug, count in unresolved.most_common(15)
            if count >= 2
        ]

    # 7. Orphaned run records (runs.jsonl says approved but file is missing)
    for r in recent:
        if r.get("status") != "approved":
            continue
        path = r.get("output_path", "")
        full = _resolve_doc_path(path, docs_path) if path else None
        if full and not full.exists():
            sig.orphaned_run_records.append({
                "topic": r.get("topic"),
                "expected_at": str(full),
            })

    # 8. Citation health summary
    arxiv_urls: Counter[str] = Counter()
    pages_with_citations = 0
    for r in recent:
        if not r.get("committed"):
            continue
        path = r.get("output_path", "")
        full = _resolve_doc_path(path, docs_path) if path else None
        if not full or not full.exists():
            continue
        try:
            content = full.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        ids = _ARXIV_RE.findall(content)
        if ids:
            pages_with_citations += 1
            for aid in ids:
                arxiv_urls[aid] += 1
    sig.citation_health = {
        "unique_arxiv_ids": len(arxiv_urls),
        "pages_with_arxiv": pages_with_citations,
        "most_cited_top3": [
            {"arxiv_id": k, "count": v}
            for k, v in arxiv_urls.most_common(3)
        ],
    }

    return sig


def _resolve_doc_path(path: str, docs_path: Path) -> Path | None:
    """Run records store output_path as `../docs/curriculum/...` (relative to
    agents/). Resolve to absolute path under docs_path."""
    if not path:
        return None
    # Strip leading `../docs/` if present
    if path.startswith("../docs/"):
        return docs_path / path[len("../docs/"):]
    if path.startswith("docs/"):
        return docs_path.parent / path
    return Path(path)


# ── Phase B: LLM proposer ───────────────────────────────────────────────────

_PROPOSER_PROMPT = """You are the FAIRE backlog agent running a SCRUM-STYLE RETROSPECTIVE
after each sprint cycle. Read the aggregated signals from the cycle's runs and
produce a structured retrospective for the next cycle.

INPUT SIGNALS (JSON):
{signals_json}

YOUR OUTPUT has five buckets, like a scrum retro:

  1. WENT_WELL — what to keep doing: approval rates, first-try wins, tracks
     producing high-confidence pages, critics that performed their job.
  2. WENT_WRONG — what failed or regressed: errored pages, recurring critic
     issues, heading drift, orphaned run records, sub-threshold landings.
  3. NEEDS_DEPTH — where current coverage is too shallow: tracks with few
     pages, topics that needed multiple revisions, weak critic dimensions
     per track, citation health gaps.
  4. NEW_TO_ADD — concrete additions for the next cycle: stub seeds derived
     from unresolved wikilinks, author pages anchoring popular citations,
     arc proposals when 4+ related concept pages exist in a track.
  5. PROCESS_IMPROVEMENTS — pipeline / prompt / critic tweaks: trim-knob
     adjustments, persona refinements, schema rule additions. These are
     reviewer-flagged proposals; they go in the backlog for human approval.

Each item in NEW_TO_ADD is an ACTION with this shape:
{{
  "action": "one-sentence description",
  "evidence": "which signal motivates this",
  "evidence_refs": ["topic-or-critic-name"],
  "risk_class": "safe" | "moderate" | "risky",
  "auto_apply": true|false,
  "action_type": "stub-seed" | "author-page-seed" | "arc-proposal" | "queue-priority-bump" | "trim-knob-adjust" | "other",
  "action_params": {{...}}
    // for stub-seed:        {{slug, track}}
    // for queue:             {{topic, priority}}
    // for arc-proposal:      {{arc_id, track, dest, steps: [slug1, slug2, ...]}}
    //   - arc_id: kebab-case (e.g. "generative-stack")
    //   - track: NN-track-slug (e.g. "02-generative-modeling")
    //   - dest: one-line capability_at_end (e.g. "5 trained generative models with comparable FID")
    //   - steps: 4-5 concept slugs in narrative order; each MUST already exist on disk
}}

Items in WENT_WELL / WENT_WRONG / NEEDS_DEPTH / PROCESS_IMPROVEMENTS are just strings
(observations or recommendations); they don't auto-apply.

auto_apply=true ONLY when:
  - risk_class == "safe"
  - action_type ∈ {{stub-seed, queue-priority-bump, arc-proposal}}
  - action_params are concrete and complete
  - for arc-proposal: track has >=3 substantive concept pages AND >=4 of the 5
    named step slugs already exist on disk AND track has <2 active arcs
    (these guardrails are also enforced at apply-time; mark safe only when
    you have evidence all three hold from the per-track signals above)

OUTPUT JSON ONLY (no preamble, no markdown fences):
{{
  "went_well":            ["string", ...],
  "went_wrong":           ["string", ...],
  "needs_depth":          ["string", ...],
  "new_to_add":           [ {{action...}}, ... ],
  "process_improvements": ["string", ...]
}}
"""


def propose_backlog_actions(signals: CycleSignals) -> dict:
    """Phase B — LLM call with structured output. Returns scrum-style
    retrospective dict with five buckets: went_well, went_wrong, needs_depth,
    new_to_add, process_improvements.

    Uses the reviewer LLM (8192 max_tokens, reasoning-tuned) rather than the
    research LLM (4096 cap) because structured retrospectives over 50+ runs
    can exceed the smaller model's output budget.
    """
    llm = get_llm("reviewer", temperature=0.0)
    structured = llm.with_structured_output(_RetroOutput)

    signals_json = json.dumps(
        {
            "cycle_window": signals.cycle_window,
            "per_track_health": signals.per_track_health,
            "heading_drift": signals.heading_drift,
            "under_threshold_landings": signals.under_threshold_landings[:10],
            "unresolved_wikilinks": signals.unresolved_wikilinks[:10],
            "orphaned_run_records": signals.orphaned_run_records[:10],
            "citation_health": signals.citation_health,
            "recurring_critic_issues": signals.recurring_critic_issues,
        },
        indent=2,
    )
    prompt = _PROPOSER_PROMPT.format(signals_json=signals_json)

    empty = {
        "went_well": [],
        "went_wrong": [],
        "needs_depth": [],
        "new_to_add": [],
        "process_improvements": [],
    }
    try:
        result: _RetroOutput = structured.invoke([HumanMessage(content=prompt)])
        return result.model_dump()
    except Exception as exc:
        # Fall back to plain-text parse on the original llm if structured fails
        try:
            response = llm.invoke([HumanMessage(content=prompt + "\n\nReturn ONLY the JSON object, no markdown.")])
            raw = _coerce_text(response.content)
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            if m:
                data = json.loads(m.group(0))
                for k in empty:
                    data.setdefault(k, [])
                return data
        except Exception:
            pass
        empty["process_improvements"] = [
            f"(LLM proposer failed: {type(exc).__name__}: {str(exc)[:160]})"
        ]
        return empty


# ── Phase C: apply + log ────────────────────────────────────────────────────

_STUB_TEMPLATE = """---
title: {title}
slug: {slug}
layer: core
subject: {subject}
page_type: concept
state: stub
authors_anchored: []
feeds_de_pillar: []
mvb_personas: [cs-student, applied-engineer, applied-researcher, frontier-researcher]
prereqs: []
tags: []
updated: {today}
has_mvb: true
---

# {title}

🚧 Agent-generated content pending. This concept sits in [{subject_h}](../index.md) and will be developed into a full v2 narrative walk-through (~2500 words) following [the schema](../../../../system/structure-v2.md).

*Seeded by the backlog agent — referenced by other wiki pages but not yet authored.*
"""


def apply_safe_proposals(
    retro: dict,
    docs_path: Path,
) -> list[dict]:
    """Phase C — execute safe items from the retro's new_to_add bucket.
    Returns audit log."""
    log: list[dict] = []
    today = datetime.now(timezone.utc).date().isoformat()

    for p in retro.get("new_to_add", []) or []:
        if not (p.get("auto_apply") and p.get("risk_class") == "safe"):
            log.append({"action": p.get("action", "?"), "status": "skipped (not safe-auto)"})
            continue

        action = p.get("action_type", "")
        params = p.get("action_params", {}) or {}

        if action == "stub-seed":
            slug = (params.get("slug") or "").strip()
            track = (params.get("track") or "").strip()
            if not (slug and track):
                log.append({"action": p.get("action", "?"), "status": "skipped (missing slug/track)"})
                continue
            # Find a track directory matching the requested track
            target_dir = docs_path / "curriculum" / "core" / track / "concepts"
            if not target_dir.parent.exists():
                # Try to match by prefix (e.g. "07" → "07-attention-...")
                core_dir = docs_path / "curriculum" / "core"
                if core_dir.exists():
                    matches = [d for d in core_dir.iterdir() if d.is_dir() and d.name.startswith(track)]
                    if matches:
                        target_dir = matches[0] / "concepts"
            if not target_dir.parent.exists():
                log.append({"action": p.get("action", "?"), "status": f"skipped (track {track} not found)"})
                continue
            target_dir.mkdir(parents=True, exist_ok=True)
            file_path = target_dir / f"{slug}.md"
            if file_path.exists():
                log.append({"action": p.get("action", "?"), "status": "skipped (file exists)"})
                continue
            num, *rest = target_dir.parent.name.split("-", 1)
            subject_h = f"{num} · {' '.join(w.capitalize() for w in rest[0].split('-'))}" if rest else target_dir.parent.name
            file_path.write_text(
                _STUB_TEMPLATE.format(
                    title=slug.replace("-", " ").title(),
                    slug=slug,
                    subject=target_dir.parent.name,
                    today=today,
                    subject_h=subject_h,
                ),
                encoding="utf-8",
            )
            log.append({"action": p.get("action", "?"), "status": f"applied → {file_path.relative_to(docs_path)}"})

        elif action == "arc-proposal":
            result = _apply_arc_proposal(p, params, docs_path)
            log.append({"action": p.get("action", "?"), "status": result})

        else:
            log.append({"action": p.get("action", "?"), "status": f"skipped (unsupported action_type={action!r})"})

    return log


# ── Move 1: arc autonomy ────────────────────────────────────────────────────
# Apply a safe arc-proposal by appending arc-index + arc-step lines to the
# next sprint queue. Guardrails ensure the arc has real material to anchor to.

_MIN_CONCEPTS_FOR_ARC = 3
_MIN_NAMED_STEPS_ON_DISK = 4
_MAX_ACTIVE_ARCS_PER_TRACK = 2


def _is_substantive_concept(path: Path) -> bool:
    try:
        body = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return False
    if len(body) < 1500:
        return False
    return "🚧" not in body and "Agent-generated content pending" not in body


def _resolve_track_dir(docs_path: Path, track: str) -> Path | None:
    """Resolve a track key (NN or NN-full-name) to the actual track directory."""
    core_dir = docs_path / "curriculum" / "core"
    if not core_dir.exists():
        return None
    direct = core_dir / track
    if direct.is_dir():
        return direct
    matches = [d for d in core_dir.iterdir() if d.is_dir() and d.name.startswith(track)]
    return matches[0] if matches else None


def _apply_arc_proposal(proposal: dict, params: dict, docs_path: Path) -> str:
    """Append arc-index + arc-step lines to current.md after guardrails pass.

    Returns a status string for the apply_log.
    """
    arc_id = (params.get("arc_id") or "").strip()
    track = (params.get("track") or "").strip()
    dest = (params.get("dest") or "").strip()
    steps = params.get("steps") or []
    if not (arc_id and track and steps and isinstance(steps, list)):
        return "skipped (arc-proposal missing arc_id/track/steps)"

    track_dir = _resolve_track_dir(docs_path, track)
    if not track_dir:
        return f"skipped (track {track!r} not found)"
    track_name = track_dir.name  # canonical NN-full form

    # Guardrail 1: track has >= MIN_CONCEPTS_FOR_ARC substantive concepts
    concepts_dir = track_dir / "concepts"
    if not concepts_dir.exists():
        return "skipped (no concepts/ in track)"
    substantive = [p for p in concepts_dir.glob("*.md")
                   if p.name != "index.md" and _is_substantive_concept(p)]
    if len(substantive) < _MIN_CONCEPTS_FOR_ARC:
        return f"skipped (track has only {len(substantive)} substantive concepts; need {_MIN_CONCEPTS_FOR_ARC})"

    # Guardrail 2: >= MIN_NAMED_STEPS_ON_DISK of the named steps exist as substantive pages
    existing_step_slugs = {p.stem for p in substantive}
    landed_steps = [s for s in steps if s in existing_step_slugs]
    if len(landed_steps) < _MIN_NAMED_STEPS_ON_DISK:
        return (f"skipped (only {len(landed_steps)}/{len(steps)} named steps exist as substantive "
                f"concepts; need >={_MIN_NAMED_STEPS_ON_DISK})")

    # Guardrail 3: track has < MAX_ACTIVE_ARCS_PER_TRACK active arcs
    arcs_dir = track_dir / "arcs"
    existing_arcs = [p for p in arcs_dir.glob("*.md")] if arcs_dir.exists() else []
    existing_arc_files = [p for p in existing_arcs if p.name != "index.md"]
    if len(existing_arc_files) >= _MAX_ACTIVE_ARCS_PER_TRACK:
        return (f"skipped (track already has {len(existing_arc_files)} active arcs; "
                f"cap is {_MAX_ACTIVE_ARCS_PER_TRACK})")

    # Already spun? Either as an on-disk arc-index file, or already queued in
    # current.md as `arc:<arc_id>` (prevents the retro re-firing the same arc
    # before the cycle has had a chance to consume the queued items).
    if arcs_dir.exists() and (arcs_dir / f"{arc_id}.md").exists():
        return f"skipped (arc {arc_id} already exists)"
    sprints_dir = Path(__file__).resolve().parent.parent.parent / "sprints"
    sprint_path = sprints_dir / "current.md"
    if sprint_path.exists():
        try:
            queued = sprint_path.read_text(encoding="utf-8", errors="ignore")
            if f"arc:{arc_id}" in queued:
                return f"skipped (arc {arc_id} already queued in current.md)"
        except Exception:
            pass

    # Build the sprint queue lines.
    # Format documented in scheduler.py::_parse_sprint_item
    # arc-index:
    #   topic | track | arc-index | frontier | arc:id dest:"..." total:N
    # arc-step:
    #   topic | track | arc-step | applied | arc:id pos:N ch:K ch_title:"..."
    #     prev:slug next:slug prev_artifact:"..." artifact:"..." total:M
    total = len(landed_steps)
    lines: list[str] = []
    lines.append("\n## Arc Index")
    lines.append(
        f'- [ ] {arc_id}-index | {track_name} | arc-index | frontier | '
        f'arc:{arc_id} dest:"{dest or arc_id.replace("-", " ")}" total:{total}'
    )
    lines.append("\n## Arc Steps")
    for i, slug in enumerate(landed_steps, 1):
        prev_slug = landed_steps[i - 2] if i > 1 else ""
        next_slug = landed_steps[i] if i < total else ""
        prev_art = "(prior step artifact)" if prev_slug else ""
        artifact = f"{slug} build at step {i}"
        chapter = (i + 1) // 2  # group steps into 2-3 chapters
        lines.append(
            f'- [ ] {slug} | {track_name} | arc-step | applied | '
            f'arc:{arc_id} pos:{i} ch:{chapter} ch_title:"step {i}" '
            f'prev:{prev_slug} next:{next_slug} '
            f'prev_artifact:"{prev_art}" artifact:"{artifact}" total:{total}'
        )

    # Append to current.md
    sprints_dir = Path(__file__).resolve().parent.parent.parent / "sprints"
    sprint_path = sprints_dir / "current.md"
    if not sprint_path.exists():
        return f"skipped (sprint queue {sprint_path} not found)"

    with sprint_path.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    return f"applied → queued arc-index + {total} arc-step items for {arc_id} in {track_name}"


def write_backlog_md(
    signals: CycleSignals,
    retro: dict,
    apply_log: list[dict],
    backlog_path: Path,
) -> None:
    """Phase C — write/prepend a timestamped scrum-retrospective entry to
    docs/system/backlog.md."""
    from .tools import display_ts
    now = display_ts()

    def _bullets(items: list, prefix: str = "- ") -> list[str]:
        return [f"{prefix}{s}" for s in (items or []) if s]

    new_section: list[str] = [
        f"## Cycle — {now} — Retrospective",
        "",
        f"> Runs analyzed: **{signals.cycle_window.get('n_runs', 0)}** · "
        f"Approved: **{signals.cycle_window.get('approved', 0)}** · "
        f"Errored: **{signals.cycle_window.get('errored', 0)}** · "
        f"Avg conf: **{signals.cycle_window.get('avg_confidence', 0.0):.2f}** · "
        f"First-try: **{signals.cycle_window.get('first_try_approval_rate', 0.0):.0%}**",
        "",
        "### 🟢 What went well",
        "",
        *(_bullets(retro.get("went_well")) or ["_(no observations this cycle)_"]),
        "",
        "### 🔴 What went wrong",
        "",
        *(_bullets(retro.get("went_wrong")) or ["_(none flagged)_"]),
        "",
        "### 🟡 What needs depth",
        "",
        *(_bullets(retro.get("needs_depth")) or ["_(coverage feels adequate)_"]),
        "",
        "### ➕ What to add (auto-applies marked ✓, others queued for review)",
        "",
    ]
    new_to_add = retro.get("new_to_add", []) or []
    if not new_to_add:
        new_section.append("_(no additions proposed)_")
        new_section.append("")
    else:
        applied_actions = {a.get("action") for a in apply_log if a["status"].startswith("applied")}
        for item in new_to_add:
            action_str = item.get("action", "(unnamed)")
            risk = item.get("risk_class", "?")
            auto = item.get("auto_apply", False)
            marker = "✓" if action_str in applied_actions else ("⏳" if auto else "○")
            new_section.append(f"- {marker} {action_str}")
            ev = item.get("evidence", "")
            if ev:
                new_section.append(f"  - _evidence:_ {ev}")
            refs = item.get("evidence_refs") or []
            if refs:
                new_section.append(f"  - _refs:_ `{'`, `'.join(refs)}`")
            new_section.append(f"  - risk: `{risk}` · auto_apply: `{auto}` · type: `{item.get('action_type', '?')}`")
        new_section.append("")

    new_section.extend([
        "### ⚙️ Process improvements (human review)",
        "",
        *(_bullets(retro.get("process_improvements")) or ["_(no process changes proposed)_"]),
        "",
    ])

    # Data appendix — the raw signals Phase A computed
    new_section.extend([
        "<details><summary><strong>Data appendix — Phase A signals</strong></summary>",
        "",
    ])

    if signals.per_track_health:
        new_section.extend([
            "**Per-track health**",
            "",
            "| Track | Pages | Avg conf | Weakest critic |",
            "|---|---|---|---|",
        ])
        for tr in sorted(signals.per_track_health):
            h = signals.per_track_health[tr]
            new_section.append(
                f"| `{tr}` | {h['n_pages']} | {h['avg_confidence']:.2f} | `{h['weakest_critic']}` ({h['weakest_critic_avg']:.2f}) |"
            )
        new_section.append("")

    if signals.heading_drift:
        new_section.append("**Heading drift (v1 forbidden headings)**")
        new_section.append("")
        for h, n in signals.heading_drift.items():
            new_section.append(f"- `{h}` × {n}")
        new_section.append("")

    if signals.under_threshold_landings:
        new_section.append(f"**Under-threshold landings ({len(signals.under_threshold_landings)})**")
        new_section.append("")
        for u in signals.under_threshold_landings[:10]:
            new_section.append(f"- `{u['topic']}` (conf {u['confidence']:.2f}, rev {u['revision_count']})")
        new_section.append("")

    if signals.unresolved_wikilinks:
        new_section.append("**Unresolved wikilinks (stub-seed candidates)**")
        new_section.append("")
        for u in signals.unresolved_wikilinks[:10]:
            srcs = ", ".join(u["referenced_by"][:3])
            new_section.append(f"- `[[{u['slug']}]]` × {u['reference_count']} refs (from {srcs})")
        new_section.append("")

    if signals.orphaned_run_records:
        new_section.append(f"**Orphaned run records ({len(signals.orphaned_run_records)})**")
        new_section.append("")
        for o in signals.orphaned_run_records[:5]:
            new_section.append(f"- `{o['topic']}` — runs.jsonl says approved, file missing")
        new_section.append("")

    if signals.recurring_critic_issues:
        new_section.append("**Recurring critic issues (top 5)**")
        new_section.append("")
        for r in signals.recurring_critic_issues:
            new_section.append(f"- × {r['count']} — {r['issue']}")
        new_section.append("")

    new_section.extend([
        "</details>",
        "",
        "---",
        "",
    ])

    # If file doesn't exist, write header + first entry
    if not backlog_path.exists():
        lines = [
            "---",
            "title: Backlog — Sprint Retrospectives",
            "description: Auto-generated scrum-style retrospective after each sprint cycle. What went well, what went wrong, what needs depth, what to add next, and process improvements.",
            "---",
            "",
            "# Backlog — Sprint Retrospectives",
            "",
            "Generated by the FAIRE backlog agent (`agents/src/frontier_agents/retrospective.py`). "
            "Each section is one cycle's scrum-style retro: ✓ = auto-applied, ⏳ = queued for next cycle, ○ = deferred to human review. "
            "Newest first. Distinct from the per-page [Agent Changelog](changelog.md) and the human-written [Learnings Log](learnings-log.md).",
            "",
        ]
        lines.extend(new_section)
        backlog_path.parent.mkdir(parents=True, exist_ok=True)
        backlog_path.write_text("\n".join(lines), encoding="utf-8")
        return

    # Else prepend the new section after the header
    existing = backlog_path.read_text(encoding="utf-8")
    # Find the first existing "## Cycle —" line; insert before it
    cycle_match = re.search(r"\n## Cycle —", existing)
    if cycle_match:
        header_part = existing[: cycle_match.start()].rstrip() + "\n\n"
        rest = existing[cycle_match.start():].lstrip("\n")
        backlog_path.write_text(header_part + "\n".join(new_section) + "\n" + rest, encoding="utf-8")
    else:
        # No prior cycle section; append after the doc header
        backlog_path.write_text(existing.rstrip() + "\n\n" + "\n".join(new_section), encoding="utf-8")


# ── Main entrypoint ─────────────────────────────────────────────────────────

def retrospective_job(
    runs_path: Path | None = None,
    docs_path: Path | None = None,
    enable_llm_proposer: bool = True,
) -> dict:
    """Called once per sprint cycle, after write_changelog_entry.

    Returns: {signals: dict, proposals: list, applied: list}
    """
    here = Path(__file__).resolve()
    agents_dir = here.parent.parent.parent  # .../agents/
    runs_path = runs_path or agents_dir / "runs" / "runs.jsonl"
    docs_path = docs_path or agents_dir.parent / "docs"

    try:
        signals = aggregate_cycle_signals(runs_path, docs_path)
    except Exception as exc:
        return {"error": f"aggregate_cycle_signals failed: {exc!r}"}

    retro: dict = {
        "went_well": [],
        "went_wrong": [],
        "needs_depth": [],
        "new_to_add": [],
        "process_improvements": [],
    }
    if enable_llm_proposer:
        try:
            retro = propose_backlog_actions(signals)
        except Exception as exc:
            retro["process_improvements"] = [f"(Phase B failed: {exc!r}) — check OPENAI_API_BASE + key"]

    try:
        apply_log = apply_safe_proposals(retro, docs_path)
    except Exception as exc:
        apply_log = [{"action": None, "status": f"apply failed: {exc!r}"}]

    backlog_path = docs_path / "system" / "backlog.md"
    try:
        write_backlog_md(signals, retro, apply_log, backlog_path)
    except Exception as exc:
        return {
            "error": f"write_backlog_md failed: {exc!r}",
            "signals": signals.__dict__,
            "retro": retro,
            "apply_log": apply_log,
        }

    return {
        "signals": signals.__dict__,
        "retro": retro,
        "applied": apply_log,
        "backlog_path": str(backlog_path),
    }


__all__ = [
    "CycleSignals",
    "aggregate_cycle_signals",
    "propose_backlog_actions",
    "apply_safe_proposals",
    "write_backlog_md",
    "retrospective_job",
]
