"""Wiki quality audit — scans docs/curriculum pages for regressions and staleness.

Called by the scheduler before each sprint execution. Returns an AuditReport
that lists issues per page with severity (critical, warning, info).

No LLM is used here — purely structural analysis. LLM-based review is done
by the reviewer_node inside the main pipeline after a page is rewritten.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path


REQUIRED_SECTIONS = [
    "## What it is",
    "## Why it matters",
    "## Essential reading",
    "## Current SotA",
    "## In production",
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
    """Scan all curriculum pages and return an AuditReport."""
    docs_path = Path(docs_dir)
    curriculum_dir = docs_path / "curriculum"
    now = datetime.now(timezone.utc)

    report = AuditReport(
        generated_at=now.strftime("%Y-%m-%d %H:%M UTC"),
        docs_dir=str(docs_path),
    )

    if not curriculum_dir.exists():
        return report

    pages = [p for p in curriculum_dir.rglob("*.md") if p.name != "index.md"]
    report.pages_scanned = len(pages)

    for page in pages:
        rel = str(page.relative_to(docs_path))
        content = page.read_text(encoding="utf-8")
        _check_page(content, rel, report, now)

    return report


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
