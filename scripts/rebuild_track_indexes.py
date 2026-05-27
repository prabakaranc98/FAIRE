"""Rebuild each track's index.md to list the real concept/author/arc/build pages.

Run after generating new concept pages so the track overview surfaces them.
Wired into the scheduler after sprint_job (one-shot per cycle).
"""
from __future__ import annotations
import re
from pathlib import Path

CORE_DIR = Path(__file__).resolve().parent.parent / "docs" / "curriculum" / "core"

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

TRACK_BLURBS = {
    "01-ai": "What AI is, what it isn't, what it can do — agentic systems, RLHF, mechanistic interpretability, alignment.",
    "02-generative-modeling": "How to draw a sample from a distribution you don't know — autoregressive, VAEs, GANs, diffusion, score matching, flow matching, optimal transport, energy-based models.",
    "03-representation-learning": "How to learn embeddings that carry useful structure — contrastive, self-supervised, masked, JEPA, multimodal.",
    "04-neural-networks-deep-learning": "The architectural primitives every modern model is built from — MLPs, CNNs, RNNs, residuals, normalization, optimization, scaling laws.",
    "05-statistical-probabilistic-ml": "Reasoning under uncertainty — Bayesian inference, variational methods, MCMC, Gaussian processes, EM, uncertainty quantification.",
    "06-reinforcement-learning": "Learning from delayed reward — MDPs, policy gradients, actor-critic, model-based RL, world models, RLHF.",
    "07-attention-memory-reasoning-continual": "How models attend, remember, and reason — attention, retrieval, long context, in-context learning, multi-step reasoning, continual learning.",
    "08-causal-statistical-inference": "Beyond correlation — structural causal models, do-calculus, counterfactuals, instrumental variables, mediation, causal representation learning.",
    "09-algorithms-systems-for-ai": "Making models fast, cheap, and deployable — distributed training, parallelism, kv-cache, FlashAttention, quantization, inference optimization.",
    "10-complexity-cognition-natural-intelligence": "What intelligence looks like beyond gradient descent — generalization, double descent, scaling collapse, emergence, compositionality.",
}


def _title_from_md(path: Path) -> str:
    """Read the title from frontmatter or the first H1; fall back to slug-ified filename."""
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return path.stem.replace("-", " ").title()
    m = re.search(r"^title:\s*(.+?)$", text, re.MULTILINE)
    if m:
        return m.group(1).strip().strip("\"'")
    m = re.search(r"^#\s+(.+?)$", text, re.MULTILINE)
    if m:
        return m.group(1).strip()
    return path.stem.replace("-", " ").title()


def _is_stub(path: Path) -> bool:
    """Return True if the page is still an unfilled stub."""
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return True
    if len(text) < 1500:
        return True
    return "🚧" in text or "Agent-generated content pending" in text


def _list_section(track_dir: Path, subfolder: str, label: str, empty_msg: str) -> str:
    """Render a markdown section that links every non-stub item in <track>/<subfolder>/.

    Two layouts supported:
      - Flat .md files (concepts, authors, builds): each file becomes one entry.
      - Subdirectories with an index.md inside (arcs): each subdir is one arc;
        the arc-index page is its index.md.
    """
    folder = track_dir / subfolder
    if not folder.exists():
        return f"## {label}\n\n*{empty_msg}*\n"

    # Flat-file entries
    pages = sorted(p for p in folder.glob("*.md") if p.name != "index.md")
    # Subdirectory entries (arcs/<arc-id>/index.md or any other arcs/<arc-id>/*.md)
    subdirs = sorted(d for d in folder.iterdir() if d.is_dir())

    real_pages = [p for p in pages if not _is_stub(p)]
    stub_pages = [p for p in pages if _is_stub(p)]
    real_subdirs = []
    for d in subdirs:
        idx = d / "index.md"
        if idx.exists() and not _is_stub(idx):
            real_subdirs.append((d, idx))

    lines = [f"## {label}", ""]
    if not real_pages and not stub_pages and not real_subdirs:
        lines.append(f"*{empty_msg}*")
        lines.append("")
        return "\n".join(lines)

    # Subdirectory items first (arcs are the headline for the section)
    for d, idx in real_subdirs:
        title = _title_from_md(idx)
        # Try to pull a one-line destination for arc cards. Look only at
        # explicit destination/capability fields, never blockquotes (those
        # contain the auto-injected breadcrumb that points back at this
        # very section).
        dest = ""
        try:
            body = idx.read_text(encoding="utf-8", errors="ignore")
            m = re.search(r"^\*\*Capability at end[^:]*:\*\*\s*(.+?)$", body, re.MULTILINE) or \
                re.search(r"^\*\*Destination[^:]*:\*\*\s*(.+?)$", body, re.MULTILINE) or \
                re.search(r"^\*\*dest[^:]*:\*\*\s*(.+?)$", body, re.MULTILINE | re.IGNORECASE)
            if m:
                dest = " — " + m.group(1).strip().strip("*").strip()[:160]
        except Exception:
            pass
        lines.append(f"- **[{title}]({subfolder}/{d.name}/)**{dest}")

    for p in real_pages:
        title = _title_from_md(p)
        lines.append(f"- [{title}]({subfolder}/{p.stem}/)")

    if stub_pages:
        lines.append("")
        lines.append(f"*Auto-seeded stubs awaiting next cycle: " + ", ".join(f"`{p.stem}`" for p in stub_pages) + "*")
    lines.append("")
    return "\n".join(lines)


def rebuild_track_index(track_dir: Path) -> str:
    """Compose the full index.md content for one track."""
    track_id = track_dir.name
    title = TRACK_TITLES.get(track_id, track_id.replace("-", " ").title())
    blurb = TRACK_BLURBS.get(track_id, "")

    concepts_section = _list_section(track_dir, "concepts", "Concepts",
        "No concept pages yet — the supervisor will queue them.")
    arcs_section = _list_section(track_dir, "arcs", "Arcs through this subject",
        "No arcs yet — the retrospective proposes these once concept coverage hits ≥4 pages per track.")
    authors_section = _list_section(track_dir, "authors", "Key thinkers",
        "Author pages pending.")
    builds_section = _list_section(track_dir, "builds", "Builds tied to this subject",
        "MVB recipes pending — currently they live inside concept pages' Build it sections.")

    # Count for the header
    n_real = sum(1 for p in (track_dir / "concepts").glob("*.md")
                 if p.name != "index.md" and not _is_stub(p)) if (track_dir / "concepts").exists() else 0
    n_stubs = sum(1 for p in (track_dir / "concepts").glob("*.md")
                  if p.name != "index.md" and _is_stub(p)) if (track_dir / "concepts").exists() else 0

    frontmatter = f"""---
title: {title}
slug: {track_id}
layer: core
subject: {track_id}
page_type: subject-overview
state: drafted
authors_anchored: []
feeds_de_pillar: []
tags: []
updated: {Path(__file__).stat().st_mtime}
---
"""
    # use today's date for updated
    from datetime import datetime
    frontmatter = re.sub(r"updated: .+", f"updated: {datetime.utcnow().date().isoformat()}", frontmatter)

    body = f"""
# {title}

> **What this subject is for:** {blurb}

**Track status:** {n_real} substantive concept page{'' if n_real == 1 else 's'}"""
    if n_stubs:
        body += f" · {n_stubs} stub{'' if n_stubs == 1 else 's'} awaiting next cycle"
    body += f". See the live [generation status](../../../system/status.md) and the [latest retrospective](../../../system/backlog.md).\n\n"

    body += concepts_section + "\n"
    body += arcs_section + "\n"
    body += authors_section + "\n"
    body += builds_section + "\n"

    body += "---\n\n"
    body += f"*Auto-rebuilt from filesystem state by `scripts/rebuild_track_indexes.py` — see [system architecture](../../../system/architecture.md).*\n"

    return frontmatter + body


def main():
    if not CORE_DIR.exists():
        print(f"FATAL: {CORE_DIR} does not exist")
        return 1
    for track_dir in sorted(CORE_DIR.iterdir()):
        if not track_dir.is_dir():
            continue
        index_path = track_dir / "index.md"
        new_content = rebuild_track_index(track_dir)
        old = index_path.read_text(encoding="utf-8") if index_path.exists() else ""
        if old != new_content:
            index_path.write_text(new_content, encoding="utf-8")
            print(f"  rebuilt: {index_path.relative_to(CORE_DIR.parent.parent.parent)}")
        else:
            print(f"  unchanged: {index_path.relative_to(CORE_DIR.parent.parent.parent)}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
