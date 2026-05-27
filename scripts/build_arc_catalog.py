"""Build a top-level arc catalog page from arc-roadmap.md.

Reads docs/system/arc-roadmap.md (the human-curated frontier scan), parses
every arc proposal, and writes docs/curriculum/arcs.md — a single card-based
catalog the reader can scan to find an arc that matches their interest.

Each arc renders as a card with:
  - Title (clickable to the arc-index page when it exists, otherwise to the
    arc's section on this catalog page)
  - One-line destination
  - Status badge: "ready", "needs-seeds", "in-progress" (queued), "live"
    (arc-index file exists)
  - 5-step preview chain
  - Home track tag

Grouped by track. Each track section has its track-name as a chip.
"""
from __future__ import annotations
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ROADMAP = ROOT / "docs" / "system" / "arc-roadmap.md"
OUT = ROOT / "docs" / "curriculum" / "arcs.md"
CORE_DIR = ROOT / "docs" / "curriculum" / "core"

TRACK_TITLES = {
    "01-ai": "AI",
    "02-generative-modeling": "Generative Modeling",
    "03-representation-learning": "Representation Learning",
    "04-neural-networks-deep-learning": "Neural Networks & Deep Learning",
    "05-statistical-probabilistic-ml": "Statistical & Probabilistic ML",
    "06-reinforcement-learning": "Reinforcement Learning",
    "07-attention-memory-reasoning-continual": "Attention, Memory, Reasoning, Continual",
    "08-causal-statistical-inference": "Causal & Statistical Inference",
    "09-algorithms-systems-for-ai": "Algorithms & Systems for AI",
    "10-complexity-cognition-natural-intelligence": "Complexity, Cognition & Natural Intelligence",
}


def parse_roadmap() -> list[dict]:
    """Parse the markdown roadmap into a structured list of arcs."""
    if not ROADMAP.exists():
        return []
    text = ROADMAP.read_text(encoding="utf-8")

    arcs: list[dict] = []
    # Split by track sections (## NN-track-...)
    track_pattern = re.compile(r"^## (\d{2}-[a-z\-]+)", re.MULTILINE)
    matches = list(track_pattern.finditer(text))
    for i, m in enumerate(matches):
        track = m.group(1).strip()
        # Some sections have a title after the dash, normalise
        track = track.split(" ")[0]
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        track_body = text[start:end]

        # Find each arc within: ### N. `arc_id` — status
        arc_pattern = re.compile(
            r"^### \d+\.\s+`([a-z0-9\-]+)`\s+—\s+(ready|needs-seeds)\b(.*?)(?=^### |\Z)",
            re.MULTILINE | re.DOTALL,
        )
        for am in arc_pattern.finditer(track_body):
            arc_id = am.group(1)
            status = am.group(2)
            body = am.group(3)

            # Destination — line starting "**Destination:**"
            dest_match = re.search(
                r"\*\*Destination:\*\*\s*\*?(.+?)\*?\s*$", body, re.MULTILINE
            )
            destination = dest_match.group(1).strip().rstrip(".") if dest_match else ""

            # Spine — line starting "**Diagonal spine:**"
            spine_match = re.search(r"\*\*Diagonal spine:\*\*\s*(.+?)$", body, re.MULTILINE)
            spine_raw = spine_match.group(1).strip() if spine_match else ""
            # Extract slugs (backtick'd or italics)
            spine_steps = re.findall(r"`([a-z0-9\-]+)`|\*([a-z0-9\-]+)\*", spine_raw)
            steps = [s[0] or s[1] for s in spine_steps]

            arcs.append({
                "arc_id": arc_id,
                "track": track,
                "status": status,
                "destination": destination,
                "steps": steps,
            })
    return arcs


def runtime_status(arc: dict) -> str:
    """Cross-check roadmap status vs on-disk presence."""
    track_dir = CORE_DIR / arc["track"] / "arcs"
    if not track_dir.exists():
        return arc["status"]
    if (track_dir / f"{arc['arc_id']}.md").exists() or \
       (track_dir / arc["arc_id"]).exists():
        return "live"
    return arc["status"]


STATUS_BADGE = {
    "live": "🟢 live — read now",
    "ready": "🟡 designed · next to be written",
    "needs-seeds": "🟠 designed · waiting on missing concept pages",
    "in-progress": "🔵 queued — writer is on it",
}


def render_arc_card(arc: dict) -> str:
    """One arc as a markdown 'card'."""
    track_pretty = TRACK_TITLES.get(arc["track"], arc["track"])
    status = runtime_status(arc)
    badge = STATUS_BADGE.get(status, status)

    steps_preview = " → ".join(f"`{s}`" for s in arc["steps"][:5])

    # Try to link to the arc-index page if it exists, otherwise to the anchor
    arc_url_disk = CORE_DIR / arc["track"] / "arcs" / f"{arc['arc_id']}.md"
    arc_dir_disk = CORE_DIR / arc["track"] / "arcs" / arc["arc_id"]
    if arc_url_disk.exists():
        href = f"core/{arc['track']}/arcs/{arc['arc_id']}/"
        title_link = f"[{arc['arc_id']}]({href})"
    elif arc_dir_disk.exists():
        href = f"core/{arc['track']}/arcs/{arc['arc_id']}/"
        title_link = f"[{arc['arc_id']}]({href})"
    else:
        title_link = f"**`{arc['arc_id']}`**"

    return (
        f"### {title_link}\n\n"
        f"{badge} · **track:** [{track_pretty}](core/{arc['track']}/index.md)\n\n"
        f"**Destination —** {arc['destination']}\n\n"
        f"{steps_preview}\n"
    )


def main() -> int:
    arcs = parse_roadmap()
    if not arcs:
        print("  no arcs parsed from roadmap; nothing to write")
        return 0

    # Group by track
    by_track: dict[str, list[dict]] = {}
    for a in arcs:
        by_track.setdefault(a["track"], []).append(a)

    # Counts for the header
    n_total = len(arcs)
    n_live = sum(1 for a in arcs if runtime_status(a) == "live")
    n_ready = sum(1 for a in arcs if runtime_status(a) == "ready")
    n_seeds = sum(1 for a in arcs if runtime_status(a) == "needs-seeds")

    lines = [
        "---",
        "title: Arcs — the wiki's USP",
        "description: Every arc is a diagonal path from a tool you already touch to a named frontier capability you can build. Curated against the 2026 frontier; backed by the arc-roadmap design doc.",
        "---",
        "",
        "# Arcs",
        "",
        "> An arc is a **diagonal** learning path: from a tool you already touch, through a broader frame, to a synthesised capability, landing at the intersection of two active research areas. Each arc names a **specific frontier destination** you build toward. The MVB at each step is the recipe; the arc is the journey.",
        "",
        f"**{n_live} live · {n_ready} designed and next-up · {n_seeds} waiting on missing concept pages · {n_total} total**",
        "",
        "*Status meanings:* 🟢 **live** = readable on the site now. 🟡 **designed · next** = the arc is designed in the [roadmap](../system/arc-roadmap.md) and all 5 concept pages it needs exist; the autonomous loop will write it next time it runs. 🟠 **waiting on missing concept pages** = the arc is designed but one or more of its concept pages need to be written first; those concept pages get auto-seeded by the retrospective, then the arc unlocks.",
        "",
        "---",
        "",
    ]

    for track in sorted(by_track):
        track_pretty = TRACK_TITLES.get(track, track)
        lines.append(f"## {track_pretty}")
        lines.append("")
        for arc in by_track[track]:
            lines.append(render_arc_card(arc))
            lines.append("")
        lines.append("---")
        lines.append("")

    lines.append(
        "*This page is auto-rebuilt from `docs/system/arc-roadmap.md` by "
        "`scripts/build_arc_catalog.py`. Refreshed each cycle.*"
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  wrote {OUT} ({n_total} arcs across {len(by_track)} tracks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
