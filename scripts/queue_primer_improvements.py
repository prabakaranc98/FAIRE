"""Queue primer-quality improve passes for pages that score below the bar.

Move 3 of the post-/plan plan. The new critic-primer-quality (commit 5ed181d)
now scores every page. Pages from before the critic existed have no score yet;
pages with score < 0.85 are borderline-readable but not yet primer-grade.

This script:
  1. Reads agents/runs/runs.jsonl and finds the LATEST primer score per topic.
  2. Picks pages where primer < 0.85 (or score is missing entirely AND the page
     exists on disk as substantive content).
  3. Appends them to a `## Primer Improvements` section in
     agents/sprints/current.md so the scheduler picks them up as `improve`
     items next sprint.
  4. Idempotent — skips topics already queued in current.md.

Wired into the scheduler as Step 4.6 (after track-index rebuild, before
retrospective). Runs once per cycle, automatically.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS_JSONL = ROOT / "agents" / "runs" / "runs.jsonl"
SPRINT_MD = ROOT / "agents" / "sprints" / "current.md"
CONCEPTS_GLOB = ROOT / "docs" / "curriculum" / "core"

PRIMER_BAR = 0.85          # below this, queue for improvement
PRIMER_KEY = "critic-primer-quality"
MAX_QUEUE_PER_CYCLE = 10   # don't flood the sprint queue


def _is_substantive(path: Path) -> bool:
    try:
        body = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return False
    if len(body) < 1500:
        return False
    return "🚧" not in body and "Agent-generated content pending" not in body


def _scan_pages() -> dict[str, dict]:
    """{topic: {track, path}} for every substantive concept page on disk."""
    out: dict[str, dict] = {}
    if not CONCEPTS_GLOB.exists():
        return out
    for track_dir in sorted(CONCEPTS_GLOB.iterdir()):
        if not track_dir.is_dir():
            continue
        concepts = track_dir / "concepts"
        if not concepts.exists():
            continue
        for p in concepts.glob("*.md"):
            if p.name == "index.md":
                continue
            if _is_substantive(p):
                out[p.stem] = {"track": track_dir.name, "path": p}
    return out


def _latest_primer_scores() -> dict[str, float | None]:
    """Walk runs.jsonl forward; keep the latest primer score per topic."""
    scores: dict[str, float | None] = {}
    if not RUNS_JSONL.exists():
        return scores
    with RUNS_JSONL.open(encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            topic = r.get("topic", "")
            if not topic:
                continue
            rubric = r.get("review_rubric") or {}
            scores[topic] = rubric.get(PRIMER_KEY)  # None if critic wasn't run for this row
    return scores


def _already_queued(topic: str, queue_text: str) -> bool:
    """True if `current.md` already mentions this topic in any sprint line."""
    pattern = re.compile(rf"^- \[[ x]\] {re.escape(topic)} \|", re.MULTILINE)
    return bool(pattern.search(queue_text))


def main() -> int:
    if not SPRINT_MD.exists():
        print(f"  no sprint file at {SPRINT_MD}; nothing to do")
        return 0

    pages = _scan_pages()
    scores = _latest_primer_scores()
    queue_text = SPRINT_MD.read_text(encoding="utf-8")

    # candidates: substantive pages whose latest primer score is < bar or missing
    candidates: list[tuple[str, dict, float | None]] = []
    for topic, meta in pages.items():
        s = scores.get(topic)
        if s is None or s < PRIMER_BAR:
            candidates.append((topic, meta, s))

    # filter out anything already queued
    fresh = [c for c in candidates if not _already_queued(c[0], queue_text)]

    # sort: lowest primer score first (missing scores treated as 0.5 — mid prio)
    fresh.sort(key=lambda c: (c[2] if c[2] is not None else 0.5, c[0]))

    pick = fresh[:MAX_QUEUE_PER_CYCLE]
    if not pick:
        print(f"  nothing to queue (candidates={len(candidates)} fresh={len(fresh)})")
        return 0

    # build the new sprint section
    lines = ["", "## Primer Improvements"]
    for topic, meta, s in pick:
        score_note = f"primer={s:.2f}" if s is not None else "primer=unscored"
        lines.append(
            f"- [ ] {topic} | {meta['track']} | core-concept | applied  "
            f"<!-- Primer-quality improve pass ({score_note}) -->"
        )

    with SPRINT_MD.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"  queued {len(pick)} primer-improvement items "
          f"(from {len(fresh)} candidates, cap {MAX_QUEUE_PER_CYCLE})")
    for topic, meta, s in pick:
        marker = f"{s:.2f}" if s is not None else "—"
        print(f"    {marker}  {meta['track']}/{topic}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
