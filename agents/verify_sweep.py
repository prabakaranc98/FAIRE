#!/usr/bin/env python
"""verify_sweep.py — Post-sweep MVB feasibility + URL trust verification.

Walks every generated curriculum + arc page, extracts MVB stacks and external
URLs, runs `verify_mvb_stack` + `verify_source_trust` per item, aggregates
findings into `docs/system/mvb-verification.md`. Pure tool-calls — no LLM
spend.

Usage:
    cd agents && uv run python verify_sweep.py
    cd agents && uv run python verify_sweep.py --tracks 01,02,06   # filter
    cd agents && uv run python verify_sweep.py --skip-stubs        # default
"""

from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

from frontier_agents.tools import verify_mvb_stack, verify_source_trust


DOCS = Path(__file__).parent.parent / "docs"
REPORT = DOCS / "system" / "mvb-verification.md"

STUB_MARKERS = ("🚧", "Agent-generated content pending")


# ── Page discovery ─────────────────────────────────────────────────────────────

def is_stub(content: str) -> bool:
    return any(marker in content for marker in STUB_MARKERS)


def list_pages(tracks_filter: list[str] | None = None) -> list[tuple[Path, str]]:
    """Return [(path, kind)] for non-stub generated pages.

    kind: 'curriculum' | 'arc-step' | 'arc-index'
    """
    pages: list[tuple[Path, str]] = []

    curriculum = DOCS / "curriculum"
    if curriculum.exists():
        for p in curriculum.rglob("*.md"):
            if p.name == "index.md":
                continue
            track = p.parent.name
            if tracks_filter and not any(track.startswith(t) for t in tracks_filter):
                continue
            content = p.read_text(encoding="utf-8", errors="ignore")
            if is_stub(content):
                continue
            pages.append((p, "curriculum"))

    arcs = DOCS / "arcs"
    if arcs.exists():
        for p in arcs.glob("*/index.md"):
            content = p.read_text(encoding="utf-8", errors="ignore")
            if not is_stub(content):
                pages.append((p, "arc-index"))
        for p in arcs.glob("*/step-*.md"):
            content = p.read_text(encoding="utf-8", errors="ignore")
            if not is_stub(content):
                pages.append((p, "arc-step"))

    return pages


# ── MVB extraction ─────────────────────────────────────────────────────────────

# Curriculum pages: 6 sub-sections under "## Minimum Valuable Builds — by persona"
# Each sub-section is "### N. For the <persona> ..." with Model/Dataset/Stack lines.
PERSONA_HEADER_RE = re.compile(
    r"^###\s*\d+\.\s*For the\s*([a-zA-Z /\-]+?)\s*\(", re.MULTILINE
)
MODEL_LINE_RE = re.compile(r"^\s*[-*]?\s*\*\*Model:\*\*\s*[`']?([^\n`'<*]+)", re.MULTILINE)
DATASET_LINE_RE = re.compile(r"^\s*[-*]?\s*\*\*Dataset:\*\*\s*[`']?([^\n`'<*]+)", re.MULTILINE)
COMPUTE_LINE_RE = re.compile(r"^\s*[-*]?\s*\*\*Compute:\*\*\s*([^\n*<]+)", re.MULTILINE)
STACK_BLOCK_RE = re.compile(r"\*\*Stack:\*\*\s*([^\n]+)", re.MULTILINE)


def _strip_md(s: str) -> str:
    """Strip trailing markdown emphasis / inline-code wrappers."""
    return s.strip().rstrip("`*_ ").lstrip("`*_ ")


def _hf_id_from(s: str) -> str:
    """Pick out a `org/name` HuggingFace ID from a stack line."""
    m = re.search(r"([A-Za-z0-9_\-]+/[A-Za-z0-9_\-.]+)", s)
    return m.group(1) if m else ""


def extract_mvb_variants(content: str, kind: str) -> list[dict]:
    """Return list of {persona, model_id, dataset_id, compute, training, raw}."""
    variants: list[dict] = []

    if kind == "curriculum":
        # Find MVB block, then iterate ### sub-sections
        mvb_start = content.find("## Minimum Valuable Builds")
        if mvb_start == -1:
            return []
        block = content[mvb_start:]
        # Split into sub-section blocks by "### " markers
        chunks = re.split(r"\n(?=###\s)", block)[1:]
        for chunk in chunks:
            persona_m = re.search(r"^###\s*\d+\.\s*For the\s*([a-zA-Z /\-]+?)\s*\(", chunk)
            persona = persona_m.group(1).strip().lower() if persona_m else "(unknown)"
            model_m = MODEL_LINE_RE.search(chunk)
            dataset_m = DATASET_LINE_RE.search(chunk)
            compute_m = COMPUTE_LINE_RE.search(chunk)
            stack_m = STACK_BLOCK_RE.search(chunk)
            model_id = _strip_md(model_m.group(1)) if model_m else (_hf_id_from(stack_m.group(1)) if stack_m else "")
            dataset_id = _strip_md(dataset_m.group(1)) if dataset_m else ""
            compute = _strip_md(compute_m.group(1)) if compute_m else ""
            training = "train" in chunk.lower() and "training" in chunk.lower()
            if model_id or dataset_id:
                variants.append({
                    "persona": persona,
                    "model_id": model_id,
                    "dataset_id": dataset_id,
                    "compute": compute,
                    "training": training,
                })
    elif kind == "arc-step":
        # Arc-step has one MVB block under "## Minimum Valuable Build"
        model_m = MODEL_LINE_RE.search(content)
        dataset_m = DATASET_LINE_RE.search(content)
        compute_m = COMPUTE_LINE_RE.search(content)
        if model_m or dataset_m:
            # Try to pull mvb_persona from frontmatter
            persona_m = re.search(r"^mvb_persona:\s*([a-z\-]+)", content, re.MULTILINE)
            persona = persona_m.group(1).strip() if persona_m else "(arc-step)"
            variants.append({
                "persona": persona,
                "model_id": _strip_md(model_m.group(1)) if model_m else "",
                "dataset_id": _strip_md(dataset_m.group(1)) if dataset_m else "",
                "compute": _strip_md(compute_m.group(1)) if compute_m else "",
                "training": "train" in content.lower(),
            })

    return variants


# ── URL extraction ─────────────────────────────────────────────────────────────

# Match [anchor](url) markdown links, plus bare URLs.
LINK_RE = re.compile(r"\[([^\]]+)\]\((https?://[^\)\s]+)\)")
BARE_URL_RE = re.compile(r"(?<!\()https?://[\w\-./?%&=#]+")


def extract_urls(content: str) -> list[tuple[str, str]]:
    """Return [(url, section_hint)] tuples for external URLs in the page.

    section_hint is one of: in_production / code_implementations /
    further_reading / default — based on which heading the URL falls under.
    """
    urls: list[tuple[str, str]] = []

    # Build (line_index, section_hint) map by scanning headings
    sections: list[tuple[int, str]] = [(0, "default")]
    for i, line in enumerate(content.splitlines()):
        if line.startswith("##"):
            heading = line.lower()
            if "in production" in heading:
                sections.append((i, "in_production"))
            elif "code & implementations" in heading or "code and implementations" in heading:
                sections.append((i, "code_implementations"))
            elif "further reading" in heading or "essential reading" in heading:
                sections.append((i, "further_reading"))
            else:
                sections.append((i, "default"))

    def section_for(line_idx: int) -> str:
        active = "default"
        for start, sec in sections:
            if line_idx >= start:
                active = sec
            else:
                break
        return active

    for i, line in enumerate(content.splitlines()):
        for _anchor, url in LINK_RE.findall(line):
            # Skip relative links and intra-page anchors
            if url.startswith("#") or not url.startswith("http"):
                continue
            urls.append((url, section_for(i)))

    return urls


# ── Main sweep ────────────────────────────────────────────────────────────────

def run_sweep(tracks_filter: list[str] | None = None) -> dict:
    pages = list_pages(tracks_filter)

    page_results = []
    mvb_total = 0
    mvb_passed = 0
    mvb_issues: list[str] = []

    url_total = 0
    url_trusted = 0
    url_failures: list[str] = []

    # Per-page rollup
    for path, kind in pages:
        content = path.read_text(encoding="utf-8", errors="ignore")
        rel = str(path.relative_to(DOCS.parent))

        # MVB checks
        variants = extract_mvb_variants(content, kind)
        page_mvb_results = []
        for v in variants:
            mvb_total += 1
            result = verify_mvb_stack(
                model_id=v["model_id"],
                dataset_id=v["dataset_id"],
                compute=v["compute"],
                training=v["training"],
            )
            if result["feasible"]:
                mvb_passed += 1
            else:
                for issue in result["issues"]:
                    mvb_issues.append(f"{rel} :: {v['persona']} → {issue}")
            page_mvb_results.append((v, result))

        # URL trust
        urls = extract_urls(content)
        page_url_results = []
        for url, section in urls:
            url_total += 1
            result = verify_source_trust(url, section)
            if result["trusted"]:
                url_trusted += 1
            else:
                # Only flag if it has score 0 (negative-list) or 0 < score < 0.5
                if result["score"] == 0.0 and "negative-list" in result["reason"]:
                    url_failures.append(f"{rel} → NEGATIVE: {url}  ({result['reason']})")
                elif result["score"] < 0.5:
                    url_failures.append(f"{rel} → low-trust ({result['score']:.2f}): {url}  ({result['reason']})")
            page_url_results.append((url, section, result))

        page_results.append({
            "path": rel,
            "kind": kind,
            "mvb": page_mvb_results,
            "urls": page_url_results,
        })

    return {
        "pages": page_results,
        "mvb_total": mvb_total,
        "mvb_passed": mvb_passed,
        "mvb_issues": mvb_issues,
        "url_total": url_total,
        "url_trusted": url_trusted,
        "url_failures": url_failures,
    }


def write_report(results: dict) -> Path:
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    pages = results["pages"]

    lines = [
        "---",
        "title: MVB feasibility + URL trust verification",
        "description: Post-sweep audit. MVB stacks checked against HuggingFace reality + GPU-VRAM math; external URLs scored against the multi-signal trust function.",
        "---",
        "",
        "# MVB feasibility + URL trust verification",
        "",
        f"**Pages scanned:** {len(pages)}",
        f"**MVB stacks checked:** {results['mvb_total']} (passed: **{results['mvb_passed']}**, failed: **{results['mvb_total'] - results['mvb_passed']}**)",
        f"**External URLs checked:** {results['url_total']} (trusted: **{results['url_trusted']}**, flagged: **{results['url_total'] - results['url_trusted']}**)",
        "",
    ]

    # MVB failures
    lines.append("## MVB feasibility failures")
    lines.append("")
    if not results["mvb_issues"]:
        lines.append("_No MVB feasibility failures across the scanned pages._")
    else:
        lines.append("Each line: `{page} :: {persona} → {issue}`. Most common failure modes are phantom HF IDs (model not on the Hub) and incoherent compute-vs-model-size triples.")
        lines.append("")
        for issue in results["mvb_issues"]:
            lines.append(f"- {issue}")
    lines.append("")

    # URL failures
    lines.append("## URL trust failures")
    lines.append("")
    if not results["url_failures"]:
        lines.append("_No URL trust failures across the scanned pages._")
    else:
        lines.append("Each line: `{page} → {verdict}: {url} ({reason})`. NEGATIVE = on the terminal block-list (Medium/Substack/Wikipedia/social media). low-trust = scored below 0.5 on the multi-signal verifier.")
        lines.append("")
        for fail in results["url_failures"]:
            lines.append(f"- {fail}")
    lines.append("")

    # Per-page detail (compact summary table)
    lines.append("## Per-page summary")
    lines.append("")
    lines.append("| Page | Kind | MVB variants | MVB pass | URLs | URL trusted |")
    lines.append("|---|---|---|---|---|---|")
    for p in pages:
        n_mvb = len(p["mvb"])
        n_mvb_ok = sum(1 for _, r in p["mvb"] if r["feasible"])
        n_url = len(p["urls"])
        n_url_ok = sum(1 for _, _, r in p["urls"] if r["trusted"])
        lines.append(
            f"| `{p['path']}` | {p['kind']} | {n_mvb} | {n_mvb_ok} | {n_url} | {n_url_ok} |"
        )
    lines.append("")

    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return REPORT


# ── CLI entry ─────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracks", default=None, help="comma-separated track prefixes (e.g. 01,02,06)")
    args = parser.parse_args()

    tracks_filter = [t.strip() for t in args.tracks.split(",")] if args.tracks else None

    print("Running verification sweep...")
    results = run_sweep(tracks_filter)
    out = write_report(results)

    print(f"\nPages scanned:    {len(results['pages'])}")
    print(f"MVB stacks:       {results['mvb_passed']}/{results['mvb_total']} feasible")
    print(f"URL trust:        {results['url_trusted']}/{results['url_total']} trusted")
    print(f"Report written:   {out}")


if __name__ == "__main__":
    main()
