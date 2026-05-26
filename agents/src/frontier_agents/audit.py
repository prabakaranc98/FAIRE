"""Wiki quality audit — scans docs/curriculum (v2 tree) for regressions and staleness.

Called by the scheduler before each sprint execution. Returns an AuditReport
that lists issues per page with severity (critical, warning, info).

No LLM is used here — purely structural analysis. LLM-based review is done
by the reviewer_node inside the main pipeline after a page is rewritten.

v2 layout: docs/curriculum/core/<track>/{concepts,authors,arcs,builds}/<slug>.md
See docs/system/structure-v2.md for the full spec.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path


# Required sections — v2 narrative template (see agents/SCHEMA.md).
# These are the section headings the writer must produce. The audit emits a
# warning per missing section. The LLM reviewer enforces the deeper checks
# (narrative form, embedded math, hook quality).
REQUIRED_SECTIONS = [
    "## The territory",
    "## How it works",
    "## Where the field is now",
    "## What's still open",
    "## Where to read next",
]

BANNED_DOMAINS = [
    "medium.com",
    "towardsdatascience.com",
    "substack.com",
    "youtube.com",
    "wikipedia.org",
]

SOTA_MAX_AGE_DAYS = 180  # flag SotA older than 6 months


@dataclass
class PageIssue:
    severity: str          # "critical" | "warning" | "info"
    check: str             # machine name for the check
    message: str
    page: str              # relative path from docs/


@dataclass
class AuditReport:
    generated_at: str
    docs_dir: str
    pages_scanned: int = 0
    issues: list[PageIssue] = field(default_factory=list)

    @property
    def critical(self) -> list[PageIssue]:
        return [i for i in self.issues if i.severity == "critical"]

    @property
    def warnings(self) -> list[PageIssue]:
        return [i for i in self.issues if i.severity == "warning"]

    def summary(self) -> str:
        return (
            f"Audit {self.generated_at}: "
            f"{self.pages_scanned} pages scanned, "
            f"{len(self.critical)} critical, {len(self.warnings)} warnings"
        )

    def to_markdown(self) -> str:
        lines = [
            f"## Audit — {self.generated_at}",
            f"> {self.pages_scanned} pages scanned · "
            f"{len(self.critical)} critical · {len(self.warnings)} warnings",
            "",
        ]
        if not self.issues:
            lines.append("No issues found.")
            return "\n".join(lines)

        for severity in ("critical", "warning", "info"):
            section_issues = [i for i in self.issues if i.severity == severity]
            if not section_issues:
                continue
            lines.append(f"### {severity.capitalize()}")
            for issue in section_issues:
                lines.append(f"- **{issue.page}** [{issue.check}]: {issue.message}")
            lines.append("")
        return "\n".join(lines)


def audit_wiki(docs_dir: str | Path = "../docs") -> AuditReport:
    """Scan all curriculum pages and arc step pages and return an AuditReport."""
    docs_path = Path(docs_dir)
    curriculum_dir = docs_path / "curriculum"
    arcs_dir = docs_path / "arcs"
    now = datetime.now(timezone.utc)

    report = AuditReport(
        generated_at=now.strftime("%Y-%m-%d %H:%M UTC"),
        docs_dir=str(docs_path),
    )

    if curriculum_dir.exists():
        pages = [p for p in curriculum_dir.rglob("*.md") if p.name != "index.md"]
        report.pages_scanned += len(pages)
        for page in pages:
            rel = str(page.relative_to(docs_path))
            content = page.read_text(encoding="utf-8")
            _check_page(content, rel, report, now)

    if arcs_dir.exists():
        _check_compounding_chain(arcs_dir, docs_path, report)

    return report


def _check_compounding_chain(arcs_dir: Path, docs_path: Path, report: AuditReport) -> None:
    """For each arc, verify step N's prev_artifact matches step N-1's artifact.

    A broken compounding chain is a warning, not a critical, because the page
    might still be readable — but it violates the arc-anatomy contract.

    Also emits info-level entries when an arc step page is missing the
    mvb_persona declaration in frontmatter.
    """
    import re as _re
    import yaml as _yaml

    for arc_dir in arcs_dir.iterdir():
        if not arc_dir.is_dir():
            continue
        arc_id = arc_dir.name
        steps = sorted(arc_dir.glob("step-*.md"))
        report.pages_scanned += len(steps)
        if not steps:
            continue

        parsed = []  # list of (pos, slug, frontmatter dict, rel-path)
        for step_path in steps:
            rel = str(step_path.relative_to(docs_path))
            text = step_path.read_text(encoding="utf-8", errors="ignore")
            # Extract YAML frontmatter
            m = _re.match(r"^---\n(.+?)\n---", text, _re.DOTALL)
            fm: dict = {}
            if m:
                try:
                    fm = _yaml.safe_load(m.group(1)) or {}
                except _yaml.YAMLError:
                    report.issues.append(PageIssue(
                        severity="critical",
                        check="invalid-frontmatter",
                        page=rel,
                        message=f"YAML frontmatter could not be parsed",
                    ))
                    fm = {}
            else:
                report.issues.append(PageIssue(
                    severity="critical",
                    check="missing-frontmatter",
                    page=rel,
                    message=f"Page does not start with `---` frontmatter",
                ))
            # Extract position from filename like step-04-ddpm.md
            file_match = _re.match(r"step-(\d{2})-(.+)\.md", step_path.name)
            pos = int(file_match.group(1)) if file_match else 0
            slug = file_match.group(2) if file_match else step_path.stem
            parsed.append((pos, slug, fm, rel))

            # Per-step checks
            if "mvb_persona" not in fm and (fm.get("has_mvb") is True or fm.get("has_mvb") is None):
                report.issues.append(PageIssue(
                    severity="info",
                    check="missing-mvb-persona",
                    page=rel,
                    message=(
                        f"Arc step lacks `mvb_persona:` in frontmatter — readers can't see "
                        f"which lane this step walks. Add one of: curious-learner, ml-tinkerer, "
                        f"applied-engineer, applied-researcher, theory-student, frontier-researcher."
                    ),
                ))

        # Compounding chain check — step N's prev_artifact must match step N-1's artifact
        parsed.sort(key=lambda x: x[0])
        for i in range(1, len(parsed)):
            prev_pos, prev_slug, prev_fm, prev_rel = parsed[i - 1]
            curr_pos, curr_slug, curr_fm, curr_rel = parsed[i]
            if curr_pos != prev_pos + 1:
                report.issues.append(PageIssue(
                    severity="warning",
                    check="arc-chain-gap",
                    page=curr_rel,
                    message=(
                        f"Step ordering gap: step {prev_pos} → step {curr_pos} "
                        f"in arc {arc_id} (expected consecutive)"
                    ),
                ))
                continue
            prev_artifact_declared = (
                prev_fm.get("compounding_artifact")
                or prev_fm.get("artifact")
                or ""
            ).strip()
            curr_prev_artifact = (curr_fm.get("prev_artifact") or "").strip()
            if not prev_artifact_declared or not curr_prev_artifact:
                continue  # one side undeclared — already flagged elsewhere
            # Normalize whitespace for comparison
            norm_a = " ".join(prev_artifact_declared.split()).lower()
            norm_b = " ".join(curr_prev_artifact.split()).lower()
            if norm_a != norm_b:
                report.issues.append(PageIssue(
                    severity="warning",
                    check="compounding-chain-broken",
                    page=curr_rel,
                    message=(
                        f"Step {curr_pos}'s prev_artifact does not match step {prev_pos}'s "
                        f"artifact verbatim. Step {prev_pos} produces: {prev_artifact_declared[:80]!r}. "
                        f"Step {curr_pos} expects: {curr_prev_artifact[:80]!r}. "
                        f"Compounding chain is broken — reader cannot follow."
                    ),
                ))


def _check_page(content: str, rel: str, report: AuditReport, now: datetime) -> None:
    # Skip stubs entirely — audit_node flags them separately
    if "🚧" in content or "Agent-generated content pending" in content:
        report.issues.append(PageIssue(
            severity="info",
            check="stub",
            message="Page is still a stub — not yet agent-generated",
            page=rel,
        ))
        return

    # Required sections
    for section in REQUIRED_SECTIONS:
        if section not in content:
            report.issues.append(PageIssue(
                severity="warning",
                check="missing_section",
                message=f"Missing section '{section}'",
                page=rel,
            ))

    # Nested bullet lists
    lines = content.split("\n")
    for i, line in enumerate(lines[:-1]):
        stripped = line.lstrip()
        if stripped.startswith("- ") or stripped.startswith("* "):
            next_stripped = lines[i + 1].lstrip()
            indent_next = len(lines[i + 1]) - len(lines[i + 1].lstrip())
            indent_curr = len(line) - len(line.lstrip())
            if (
                (next_stripped.startswith("- ") or next_stripped.startswith("* "))
                and indent_next > indent_curr
            ):
                report.issues.append(PageIssue(
                    severity="warning",
                    check="nested_list",
                    message=f"Nested bullet list found near line {i + 1}",
                    page=rel,
                ))
                break  # one report per page

    # Banned domains
    links = re.findall(r'\(https?://[^)]+\)', content)
    for link in links:
        for banned in BANNED_DOMAINS:
            if banned in link:
                report.issues.append(PageIssue(
                    severity="critical",
                    check="banned_domain",
                    message=f"Banned domain '{banned}' in link: {link[:80]}",
                    page=rel,
                ))

    # No arXiv sources
    if "arxiv.org" not in content:
        report.issues.append(PageIssue(
            severity="warning",
            check="no_arxiv",
            message="No arXiv sources found — page needs academic citations",
            page=rel,
        ))

    # Stale SotA (check frontmatter updated: field)
    updated_match = re.search(r"updated:\s*(\d{4}-\d{2}-\d{2})", content)
    if updated_match:
        updated_str = updated_match.group(1)
        try:
            updated_dt = datetime.strptime(updated_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            age_days = (now - updated_dt).days
            if age_days > SOTA_MAX_AGE_DAYS:
                report.issues.append(PageIssue(
                    severity="info",
                    check="stale_sota",
                    message=f"Page updated {age_days} days ago — SotA may be outdated",
                    page=rel,
                ))
        except ValueError:
            pass
