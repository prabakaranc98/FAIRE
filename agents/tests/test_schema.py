"""Validate existing wiki pages against the canonical schema.

These tests run against the actual docs/ content to catch regressions
when pages are edited manually or agent-generated.
"""

import re
from pathlib import Path

import pytest

DOCS_DIR = Path(__file__).parent.parent.parent / "docs"
CURRICULUM_DIR = DOCS_DIR / "curriculum"

# Pages expected to be "full" (not stubs) — these get strict schema validation
FULL_PAGES = [
    CURRICULUM_DIR / "02-generative-modeling" / "diffusion-models.md",
    CURRICULUM_DIR / "02-generative-modeling" / "flow-matching.md",
    CURRICULUM_DIR / "07-attention-memory-reasoning" / "transformer.md",
    CURRICULUM_DIR / "06-reinforcement-learning" / "rlhf.md",
]

# Required sections in every full wiki page
REQUIRED_SECTIONS = [
    "## What it is",
    "## Why it matters at the frontier",
    "## Core concepts",
    "## Essential reading",
    "## Current SotA",
    "## In production",
    "## Connected topics",
]

# Domains never allowed anywhere on a page
BANNED_DOMAINS = [
    "medium.com",
    "towardsdatascience.com",
    "substack.com",
    "youtube.com",
    "wikipedia.org",
]

# Required frontmatter keys
REQUIRED_FRONTMATTER = ["title:", "track:", "updated:"]


def _read_page(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _extract_frontmatter(content: str) -> str:
    """Return frontmatter block between --- delimiters."""
    if content.startswith("---"):
        end = content.find("---", 3)
        return content[:end] if end != -1 else ""
    return ""


def _extract_links(content: str) -> list[str]:
    """Extract all URLs from markdown links."""
    return re.findall(r'\(https?://[^)]+\)', content)


@pytest.mark.parametrize("page_path", FULL_PAGES, ids=[p.name for p in FULL_PAGES])
class TestFullPageSchema:
    def test_page_exists(self, page_path):
        assert page_path.exists(), f"Expected full page not found: {page_path}"

    def test_has_required_frontmatter(self, page_path):
        content = _read_page(page_path)
        fm = _extract_frontmatter(content)
        for key in REQUIRED_FRONTMATTER:
            assert key in fm, f"{page_path.name}: Missing frontmatter key '{key}'"

    def test_has_required_sections(self, page_path):
        content = _read_page(page_path)
        for section in REQUIRED_SECTIONS:
            assert section in content, (
                f"{page_path.name}: Missing required section '{section}'"
            )

    def test_has_tldr_line(self, page_path):
        content = _read_page(page_path)
        assert "TL;DR" in content or "**TL;DR**" in content, (
            f"{page_path.name}: Missing TL;DR opener line"
        )

    def test_no_banned_domains(self, page_path):
        content = _read_page(page_path)
        links = _extract_links(content)
        for link in links:
            for banned in BANNED_DOMAINS:
                assert banned not in link, (
                    f"{page_path.name}: Banned domain '{banned}' found in link: {link}"
                )

    def test_has_latex_equations(self, page_path):
        content = _read_page(page_path)
        has_latex = "$$" in content or "\\(" in content or "$" in content
        assert has_latex, f"{page_path.name}: No LaTeX equations found (expected for full pages)"

    def test_not_a_stub(self, page_path):
        content = _read_page(page_path)
        assert "🚧" not in content, (
            f"{page_path.name}: Page still marked as stub (contains 🚧)"
        )

    def test_has_arxiv_sources(self, page_path):
        content = _read_page(page_path)
        assert "arxiv.org" in content, (
            f"{page_path.name}: No arXiv sources found — pages need verified academic sources"
        )


class TestHasMvbPages:
    @pytest.mark.parametrize("page_path", [
        CURRICULUM_DIR / "02-generative-modeling" / "diffusion-models.md",
        CURRICULUM_DIR / "06-reinforcement-learning" / "rlhf.md",
        CURRICULUM_DIR / "07-attention-memory-reasoning" / "transformer.md",
    ], ids=["diffusion-models", "rlhf", "transformer"])
    def test_has_mvb_section(self, page_path):
        content = _read_page(page_path)
        assert "## Minimum Valuable Build" in content, (
            f"{page_path.name}: Missing '## Minimum Valuable Build' section"
        )

    @pytest.mark.parametrize("page_path", [
        CURRICULUM_DIR / "02-generative-modeling" / "diffusion-models.md",
        CURRICULUM_DIR / "06-reinforcement-learning" / "rlhf.md",
    ], ids=["diffusion-models", "rlhf"])
    def test_mvb_has_huggingface_link(self, page_path):
        content = _read_page(page_path)
        mvb_start = content.find("## Minimum Valuable Build")
        next_section = content.find("\n## ", mvb_start + 1)
        mvb_block = content[mvb_start:next_section] if next_section != -1 else content[mvb_start:]
        assert "huggingface.co" in mvb_block, (
            f"{page_path.name}: MVB section has no HuggingFace model/dataset link"
        )


class TestProseQuality:
    """Enforce narrative prose rules — no nested lists, explanatory sections must flow."""

    def _get_section(self, content: str, heading: str) -> str:
        """Extract content between `heading` and the next ## section."""
        start = content.find(heading)
        if start == -1:
            return ""
        next_section = content.find("\n## ", start + 1)
        return content[start:next_section] if next_section != -1 else content[start:]

    @pytest.mark.parametrize("page_path", FULL_PAGES, ids=[p.name for p in FULL_PAGES])
    def test_no_nested_lists(self, page_path):
        """No bullet point should be immediately followed by an indented bullet."""
        content = _read_page(page_path)
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
                    pytest.fail(
                        f"{page_path.name}: Nested bullet list found at line {i + 1}:\n"
                        f"  {line!r}\n  {lines[i + 1]!r}"
                    )

    @pytest.mark.parametrize("page_path", FULL_PAGES, ids=[p.name for p in FULL_PAGES])
    def test_what_it_is_is_prose(self, page_path):
        """The 'What it is' section must contain at least 2 prose paragraphs."""
        content = _read_page(page_path)
        section = self._get_section(content, "## What it is")
        if not section:
            pytest.skip(f"{page_path.name}: No 'What it is' section found")

        # Count non-empty, non-heading, non-bullet lines as prose
        prose_paragraphs = [
            p for p in section.split("\n\n")
            if p.strip()
            and not p.strip().startswith("#")
            and not p.strip().startswith("- ")
            and not p.strip().startswith("* ")
            and not p.strip().startswith("|")
            and len(p.strip()) > 60
        ]
        assert len(prose_paragraphs) >= 2, (
            f"{page_path.name}: 'What it is' section has fewer than 2 prose paragraphs "
            f"(found {len(prose_paragraphs)}). Must be narrative, not a bullet dump."
        )

    @pytest.mark.parametrize("page_path", FULL_PAGES, ids=[p.name for p in FULL_PAGES])
    def test_why_it_matters_no_bullets(self, page_path):
        """The 'Why it matters' section must not be a bullet list."""
        content = _read_page(page_path)
        section = self._get_section(content, "## Why it matters")
        if not section:
            pytest.skip(f"{page_path.name}: No 'Why it matters' section found")

        lines = [l.strip() for l in section.split("\n") if l.strip()]
        bullet_lines = [l for l in lines if l.startswith("- ") or l.startswith("* ")]
        prose_lines = [
            l for l in lines
            if not l.startswith("#") and not l.startswith("-") and not l.startswith("*")
            and not l.startswith("|") and len(l) > 40
        ]
        assert len(prose_lines) >= 1, (
            f"{page_path.name}: 'Why it matters' section is all bullet points, no prose. "
            f"Must be narrative."
        )


class TestAllStubPages:
    """Every page in docs/curriculum must have valid frontmatter — even stubs."""

    def _get_all_topic_pages(self) -> list[Path]:
        return [
            p for p in CURRICULUM_DIR.rglob("*.md")
            if p.name != "index.md"
        ]

    def test_all_pages_have_title_frontmatter(self):
        for page in self._get_all_topic_pages():
            content = page.read_text(encoding="utf-8")
            if content.startswith("---"):
                fm = _extract_frontmatter(content)
                assert "title:" in fm, f"{page.name}: Missing 'title:' in frontmatter"

    def test_no_page_has_banned_domains(self):
        for page in self._get_all_topic_pages():
            content = page.read_text(encoding="utf-8")
            links = _extract_links(content)
            for link in links:
                for banned in BANNED_DOMAINS:
                    assert banned not in link, (
                        f"{page.name}: Banned domain '{banned}' in link: {link}"
                    )
